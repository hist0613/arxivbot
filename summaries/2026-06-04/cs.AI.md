New uploads on arXiv(cs.CL)

### Streaming Communication in Multi-Agent Reasoning (https://arxiv.org/abs/2606.05158)
Comments:
          project page: this https URL

- **What's New**: 이번 논문에서는 Multi-agent reasoning systems의 효율성을 크게 향상시키는 StreamMA라는 새로운 시스템을 소개합니다. StreamMA는 reasoning 단계가 생성되자마자 하류 에이전트에게 전송되는 구조를 채택하여 latency를 줄이고, 동시에 각 단계의 신뢰성을 최대한 활용합니다. 이러한 방식은 하류 에이전트가 더 신뢰할 수 있는 초기 단계에 초점을 맞출 수 있게 하여 전체적인 오류를 감소시킵니다.

- **Technical Details**: StreamMA는 전통적인 'generate-then-transfer' 패러다임 대신, 단계별로 reasoning을 스트리밍합니다. 이 시스템은 pipeline을 통해 인접한 에이전트를 연결하여 latency를 줄이는 데 기여하며, 이는 수학, 과학, 코드 등 다양한 reasoning 벤치마크에서 효과적으로 입증되었습니다. 본 연구는 stream, serial, single 프로토콜의 장점을 정량적으로 분석하며, 효과성 순위, 속도 증가 최대 한계 및 비용 비율을 도출합니다.

- **Performance Highlights**: StreamMA는 Claude Opus 4.6 및 GPT-5.4와 같은 두 가지 선도적인 LLM을 사용하여 총 여덟 개의 reasoning 벤치마크에서 뛰어난 성능을 기록했습니다. 평균 7.3 포인트 향상, 최대 22.4 포인트 향상을 달성하여 기존 기법을 능가했습니다. 또한 에이전트당 단계 수를 늘리면 효과성과 효율성이 일관되게 개선된다는 새로운 'step-level scaling law'도 발견하였습니다.



### Activation-Based Active Learning for In-Context Learning: Challenges and Insights (https://arxiv.org/abs/2606.05134)
Comments:
          9 pages, 3 figures

- **What's New**: 이번 논문에서는 최근의 transformer 활성화 이해를 기반으로 한 깊이 있는 활성 학습(deep active learning) 방법을 대규모 언어 모델(Large Language Models, LLM)의 맥락에서 샘플 선택에 활용하는 방법을 탐구했습니다. MLP 활성화에 기반한 활성 학습 방법의 구체적인 분석을 제공하며, 다양한 주의 집중(masking) 전략들이 어떻게 활성 학습에 영향을 미치는지를 다룹니다. 연구 결과, MLP 출력이 예제 품질이나 과제 성능과 직접적인 상관관계를 가지지 않음을 발견하였습니다.

- **Technical Details**: 본 논문에서는 MLP 계층 활성화를 통해 맥락 예제 선택을 최적화하는 신호를 탐색했습니다. 다양한 정보 흐름 변형을 허용하기 위해 여러 가지 mascar를 시도하며, 주의 집중 및 causal masking 전략을 포함시킵니다. 활성화를 점수화하기 위해 평균, 분산, 왜도(skewness), 첨도(kurtosis)와 같은 네 가지 모먼트를 사용해 다양한 활성화 패턴을 포착합니다.

- **Performance Highlights**: 실험을 통해 우리는 Llama-3.2-3B 및 Qwen2.5-3B 모델을 사용하여 여러 분류 및 생성 데이터셋에서 다양한 성능을 평가했습니다. 하지만 MLP 활성화 기반의 활성 학습 방식이 과제 성능과 직접적 상관관계가 없다는 부정적인 결과를 도출하였습니다. 이러한 결과는 향후 연구를 위한 유용한 통찰과 도전 과제를 제공합니다.



### Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data (https://arxiv.org/abs/2606.05122)
- **What's New**: 이 논문에서는 대형 언어 모델이 다른 모델에 의해 평가되는 방식에 대한 새로운 통찰을 제시합니다. 특히, 모델이 자신의 결과에 대한 평가 점수를 예측할 수 있는지를 탐구하며, Self-Evaluation Elicitation (SEE)라는 방법론을 도입하여 성능을 개선합니다. 연구 결과, 모델이 주어진 기준 이상의 평가를 예측할 수 있는 능력이 이미 훈련 전에도 존재함을 발견하였습니다. 이를 통해 자가 평가(self-evaluation)의 관점을 변화시키고 있습니다.

- **Technical Details**: Self-Evaluation Elicitation (SEE)는 짧은 사이클로 구성된 과정으로, 보강 학습(reinforcement learning)과 마스크 증류(masked distillation) 단계를 포함합니다. 첫 번째 단계에서는 보강 학습을 통해 답변을 개선하고 평가자를 예측합니다. 두 번째 단계에서는 오직 자가 평가 토큰에 대해서만 실제 점수에 기반한 학습이 이루어져 모델의 답변은 변경되지 않도록 합니다. 이 두 단계를 반복하여 모델의 자가 평가 능력을 효과적으로 끌어내는 방식을 제시합니다.

- **Performance Highlights**: SEE 방법론은 기존의 보강 학습 접근법에 비해 적은 훈련 샘플로도 성능을 개선합니다. 연구에 따르면, SEE를 통해 모델의 자가 평가 오차가 평균 절대 오차(mean absolute error) 0.25∼0.66으로 감소하며, 각 벤치마크에서 평가 품질이 향상되었습니다. 이는 자가 평가 예측이 신뢰도 있게 진행되었음을 나타내며, 보강 학습을 통한 기존 방법과 비교하였을 때 매우 긍정적인 결과를 보였습니다.



### Evaluating Large Language Models in Dynamic Clinical Decision-Making with Standardized Patient Cases (https://arxiv.org/abs/2606.05112)
- **What's New**: 최근에 발표된 이 논문에서는 MedSP1000이라는 상호작용 기준을 도입하였습니다. 이 기준은 1,638개의 표준화된 환자 시나리오를 포함하고 있어, 더 정교하게 임상 에이전트를 평가할 수 있도록 합니다. 기존의 단일 턴(static, single-turn) 벤치마크에서는 제대로 평가할 수 없는 동적인 치료 제공 과정에서의 성능을 잘 반영하고 있습니다.

- **Technical Details**: MedSP1000은 기존의 표준화된 환자(teaching case) 사례를 실행 가능한 시나리오로 변환합니다. 각 시뮬레이션에서 임상 에이전트는 환자 에이전트(patient agent) 및 환경 제어기(environment controller)와 상호작용하며, 전문가 기준에 따라 평가됩니다. 이 평가 과정에서는 24,602개의 피어 리뷰(peer-reviewed) 루브릭(rubrics)을 활용하여 임상 성과를 측정합니다.

- **Performance Highlights**: MedSP1000을 일반 목적 및 의학 전용 대형 언어 모델(LLMs)에 적용한 결과, 정적인 벤치마크에서의 성능이 교육적 시나리오에서 신뢰할 수 없다는 것을 발견했습니다. 최상위 성과 모델인 GPT-5.5는 전문가 기준의 60.4%만 충족하였고, 의학 전용 모델은 40.0%에 그쳤습니다. 이는 현재의 LLM이 실제 임상에서 안전하게 통합되기에 충분히 신뢰할 수 없음을 시사합니다.



### Arithmetic Pedagogy for Language Models (https://arxiv.org/abs/2606.05106)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 연구는 인도네시아의 GASING 방식에 기반하여 언어 모델의 수학적 사고 방법을 훈련하는 데 사람이 사용하는 수학 교육 방법이 어떻게 적용될 수 있는지를 조사합니다. GASING 방법을 통해 기본적인 산술 문제를 해결하기 위한 절차적 접근 방식을 채택하고, 이러한 방법을 자연어 Chain-of-Thought (CoT) 감독과 결합합니다. 연구의 목표는 소규모 모델이 수학적으로 사고하는 능력을 어떻게 학습할 수 있는지를 탐구하며, 이는 인간의 수학 이해를 돕는 교육 방법과의 연계를 통해 이루어집니다.

- **Technical Details**: 연구에서는 GASING 방법을 사용하여 언어 모델의 훈련 및 추론 패턴을 수립합니다. 특정 수학적 관계를 내부적으로 형성하는 Transformer 모델의 훈련을 통해, 모델은 처녀 상태에서 초기 상태를 지나 최종 출력을 산출하는 일련의 과정을 통하여 발생하는 수학적 절차를 학습합니다. 모델 성능을 평가하기 위해, 동일한 문제를 푸는 다른 대규모 언어 모델과 비교하여 일반화 능력을 평가합니다.

- **Performance Highlights**: 훈련된 모델은 새로운 산술 문제에 대해 80% 이상의 정확도로 성과를 보이며, 더 큰 매개변수를 가진 모델과 비교하여 경쟁력 있는 성능을 달성했습니다. 특히, 이번 연구는 예측 목표 하에서 소규모 모델이 수학적 사고 패턴을 학습할 수 있음을 보여줍니다. 이러한 결과는 특정 교육적 방법이 적은 규모의 모델에서도 강력한 산술 능력을 지속적으로 발휘할 수 있다는 것을 입증합니다.



### Light or Full Verb? A Minimal-Pair Dataset for Probing Phraseological Competence in Language Models (https://arxiv.org/abs/2606.05087)
- **What's New**: 이 연구에서는 'have'와 'make'와 같은 자주 사용되는 영어 동사가 경량 동사 구성(light-verb constructions, LVCs) 및 풀 렉시컬 프레디케이트(full lexical predicates)로 기능할 수 있는지에 대한 여부를 다룹니다. 이를 위해 같은 동사가 경량 동사와 풀 동사로 사용된 영어 문장의 최소 변동 데이터셋을 소개하고, 두 가지 실험을 통해 언어 모델이 이들 사용을 구별할 수 있음을 보였습니다. 이 데이터셋은 재사용 가능한 자원으로 제공될 예정입니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 현재 LVC에서 자주 발생하는 다섯 개의 고빈도 영어 동사로 구성됩니다. 각 동사는 LVC 후보 리스트에서 검증되어야 하며, 후보들은 문맥의 의미적 일관성과 구문적 구조를 고려하여 선택됩니다. 이 데이터셋은 최소 쌍 기준을 확장하여 최소 문장을 시리즈로 결합하는 방법을 사용하여, 경량 동사 사용과 풀 동사 사용의 대조를 효율적으로 평가할 수 있게 설계되었습니다.

- **Performance Highlights**: 실험 결과, 최소한의 문맥만으로도 LVC와 전체 동사 간의 차별화된 효과가 관찰됨을 보여주었습니다. 언어 모델에서 경량/풀 동사 구별이 기대치 및 내부 표현에서 나타나는지를 평가하기 위해, 서프리잘 기반의 프로빙 및 대조 임베딩 분석이 수행되었습니다. 이로 인해 모델이 기대치를 구현할 수 있는지와 LVC 및 풀 동사 구성이 언어 모델의 내적 표현에 반영되는지를 확인할 수 있었습니다.



### Automatic Generation of Titles for Research Papers Using Language Models (https://arxiv.org/abs/2606.05085)
Comments:
          24 pages, 24 tables, 01 figure

- **What's New**: 이번 연구는 초록에서 연구 논문의 제목을 자동으로 생성하는 기술을 제안합니다. 특히, 사전 훈련된 언어 모델(PLMs)과 대형 언어 모델(LLMs)을 사용하여 제목 생성의 효과성을 평가합니다. 새로운 데이터셋 SpringerSSAT를 소개하고, 여러 모델의 성능을 비교하여 최적의 제목 생성 방법을 탐색합니다.

- **Technical Details**: 연구는 PLMs(예: T5, BART, PEGASUS)와 LLMs(예: LLaMA-3-8B, GPT-3.5-turbo)의 조합을 통해 이뤄지며, 모델 훈련은 CSPubSum, LREC-COLING-2024 및 새로 생성된 SpringerSSAT 데이터셋을 기반으로 합니다. ROUGE, METEOR와 같은 자동화된 평가지표를 사용하여 모델의 성능을 평가하며, ChatGPT의 창의적인 제목 생성 가능성도 탐구합니다.

- **Performance Highlights**: 실험 결과, PEGASUS-large 모델이 대부분의 지표에서 뛰어난 성능을 보인 것으로 나타났습니다. ChatGPT도 스타일 다양성을 인정받아 인간 작가의 제목과의 비교에서 긍정적인 평가를 받았습니다. 모든 훈련된 모델과 데이터셋은 Hugging Face에 공개되어 연구자들이 활용할 수 있도록 하고 있습니다.



### Fast & Faithful Function Vectors (https://arxiv.org/abs/2606.05079)
- **What's New**: 이번 연구에서는 기능 벡터(Function Vectors, FVs)의 정의가 지시 사항에 미치는 영향을 분석하며, 주목할 만한 두 가지 자유도를 제시합니다. 첫째, 주목 헤드 선택(attention head selection)에서 그래디언트 기반 방법인 Layer-wise Relevance Propagation (LRP)을 활용하면 효율성과 정확성이 크게 향상됩니다. 둘째, FVs의 적용 방식에서 분산적으로 적용할 경우, 단순 집계 방식보다 더 높은 정확도를 기록합니다. 연구의 코드 또한 공개되어 있습니다.

- **Technical Details**: 연구에서는 두 가지 방법을 사용하여 기능 벡터를 추출하기 위한 헤드를 선택합니다. 첫 번째는 Todd et al. (2024)에 의해 제안된 평균 간접 효과(Average Indirect Effect, AIE)와 Layer-wise Relevance Propagation (LRP)입니다. LRP의 경우, 특히 AttnLRP를 기반으로 하여 그래디언트 기반의 효율성이 뛰어난 방법으로 선택됩니다. FVs를 구축하는 과정에서는 지시문에 대한 헤드의 활성화를 평균하여 각 헤드에서 과제 표현(task representation)을 계산합니다.

- **Performance Highlights**: FVs를 직접적으로 각 헤드에서 적용함으로써 정확도가 상당히 향상됨을 보여줍니다. 특히, 헤드를 그래디언트 기반 방법으로 선택할 경우, 활성화 방식을 사용하는 경우보다 성능이 뛰어난 것으로 나타났습니다. 이러한 결과는 FVs 정의에 대한 방법론적 세부 사항의 민감성을 강조하는 동시에, 향후 FVs 선택에 대한 추천 자료로 활용될 수 있기를 희망합니다.



### Boosting Self-Consistency with Ranking (https://arxiv.org/abs/2606.05054)
Comments:
          16 pages, 13 figures, accepted at ACL Student Research Workshop 2026

- **What's New**: 이번 논문에서는 Ranking-Improved Self-Consistency (RISC)라는 새로운 방법을 소개합니다. RISC는 self-consistency 기법을 답변 선택 문제를 랭킹 문제로 재구성하여 성능을 향상시키는 접근 방식입니다. 이를 통해 기존의 다수결 방법을 넘어 보다 정확한 답변 선택이 가능해집니다. 특히 RISC는 후보 답변을 평가하기 위해 간소화된 LambdaRank 모델을 사용하여 다섯 가지 상호 보완적인 특성을 기반으로 점수를 매깁니다.

- **Technical Details**: RISC는 LLM이 여러 후보 답변을 샘플링하고 이를 랭킹하여 최종 답변을 선택하는 방식으로 작동합니다. 이를 위해, 다섯 가지 해석 가능한 특성(특징)을 설계하여 각 후보 답변의 빈도(frequency), 의미 중심성(semantic centrality), 추론 추적 일관성(reasoning-trace consistency)을 평가합니다. RISC는 샘플링된 후보 집합 내에서 답변의 상대적 점수를 매기는 것을 목표로 하며, 단일 확신(confidence) 신호에 의존하지 않고 리스트와 관련된(supervision) 점수 최적화를 통해 성능을 극대화합니다.

- **Performance Highlights**: RISC는 세 가지 데이터셋에서 테스트 결과, 기존의 다수결 방법과 유사한 강력한 기준선을 초과한 성능을 보여주었습니다. 특히 질문응답(QA) 벤치마크에서 유의미한 성능 향상을 기록하여 효율성과 정확성의 상생 관계를 잘 보여주었습니다. 각 특성들은 독립적으로 유용하며, 조합 시 추가적인 성능 향상을 가져오는 상호 보완적인 효과가 있다는 점이 특히 강조되었습니다.



### Imbuing Large Language Models with Bidirectional Logic for Robust Chain Repair (https://arxiv.org/abs/2606.05030)
Comments:
          25 Pages

- **What's New**: 최근 발표된 Teleological Reasoning Infilling (TRI) 프레임워크는 자동회귀 체인-오브-생각 (CoT) 훈련 방식을 개선합니다. 이 프레임워크는 디코더 전용 트랜스포머가 목표에 기반하여 결합할 수 있는 능력을 부여하는 방식으로, 오류가 포함된 추론 세그먼트를 가운데를 채우는 (Fill-in-the-Middle) 과제로 재구성합니다. TRI는 목표 논리를 연결하기 위해 미리 검증된 전제와 미래 이정표를 동시 고려할 수 있게 하여, 오류 확산 현상을 최소화합니다.

- **Technical Details**: TRI는 두 단계의 훈련 프로세스를 거치며, 첫 번째 단계에서는 형식적 수학 문헌에서 추출한 검증된 (Q,P,S,M) 쌍에 대한 감독된 미세 조정을 수행합니다. 두 번째 단계인 DPO (Direct Preference Optimisation)에서는 결정론적 기호 검증기(Lean 4/Python)를 통해 긍정적인 효율성을 극대화하고, LLM에 기반한 평가를 제거합니다. 추론 단계에서는 TRI가 오류가 포함된 세그먼트만 재생산하며 검증된 부분은 보존됩니다.

- **Performance Highlights**: TRI는 MATH, HumanEval-Fix, Lean-Workbook과 같은 세 가지 벤치마크에서 최신 성능 기록을 달성하며, CoT, CoT-SC, ToT보다 우수한 결과를 나타냅니다. 또한 TRI는 문제당 토큰 사용량을 31.2% 줄이면서 효율성을 증가시킵니다. 이 연구는 LLM의 문제 해결 능력을 크게 향상시키는 접근 방식으로, 정확하고 관련성 있는 추론을 가능하게 합니다.



### TaDA: Calibrated Probe Gating for Task-Domain LoRA Merging (https://arxiv.org/abs/2606.05016)
- **What's New**: 이번 연구에서는 Task LoRA 어댑터와 Domain LoRA 어댑터를 통합하는 새로운 접근 방식인 TaDA (Task-Domain LoRA Merging)를 제안합니다. TaDA는 프로브 신호를 기반으로 한 퍼-레이어 게이팅(per-layer gating)과 서브스페이스 인식을 통한 머징(merging) 기법을 이용하여 두 어댑터 간의 비대칭 구조를 활용합니다. 이 알고리즘은 추가 학습 없이도 작동하며, 각 레이어와 프로젝션 타입에 개별 가중치를 할당합니다.

- **Technical Details**: TaDA는 N=32개의 도메인 특화 입력을 통해 두 개의 어댑터에서의 액티베이션 노름을 측정하여 구조적 패턴을 발견합니다. 연구 결과, 도메인 지배는 레이어 깊이에 따라 증가하고, 얕은 레이어는 더 많은 작업 관련 신호를 유지하는 경향이 있습니다. TaDA는 이를 기반으로 한 스코어를 사용하여 개별 레이어와 모듈 유형에 대한 가중치를 동적으로 할당하며, SVD를 이용해 어댑터를 분해하고, 상충하는 방향을 제거한 후 남은 성분을 결합합니다.

- **Performance Highlights**: TaDA는 Llama-2-7B 모형을 기반으로 한 여섯 개의 과학적 QA 벤치마크에서 평균 정확도 0.452를 달성하여 DARE-TIES보다 3.6% 포인트 높은 성과를 보여주었습니다. 또한, ViT-L/16을 사용하는 여섯 개의 이미지 분류 벤치마크에서도 평균 85.9%의 정확도를 기록하며, 가장 강력한 벤치마크를 초과 달성했습니다. TaDA는 랜덤하게 어댑터 가중치를 드롭하는 기존 방법들과 달리 비대칭 구조를 활용하여 예측 성능을 향상시킵니다.



### Depth-Attention: Cross-Layer Value Mixing for Language Models (https://arxiv.org/abs/2606.05014)
Comments:
          21 pages, 4 figures, 9 tables

- **What's New**: 이 논문에서는 Self-attention 메커니즘을 통해 정보 선택을 자유롭게 하면서도, Transformer의 깊이에 걸쳐서는 각 레이어의 출력을 단순히 잔여 흐름에 추가하는 한계점이 있다는 점을 지적합니다. 최근의 교차 레이어(cross-layer) 방법들이 이러한 흐름을 개선하고자 했지만, 이들은 주로 hidden states에 접근해야 하므로 inference 과정에서 오버헤드가 발생합니다. 새로운 Depth-Attention 방법은 이 문제를 해결하며, 레이어 내에서 정보 선택을 가능하게 하여 추가적인 파라미터 없이 작동합니다.

- **Technical Details**: Depth-Attention은 self-attention 모듈 내에서 깊이 방향의 정보 선택을 수행합니다. 각 레이어는 먼저 이전 레이어의 키를 정렬하여 같은 토큰 위치에서 주의를 기울인 후, 해당 값들을 혼합하여 업데이트된 값 상태를 생성합니다. 이 과정은 standard causal self-attention으로 이어지며, 원래의 keys와 queries에는 아무런 변화도 주지 않습니다.

- **Performance Highlights**: 1.5B 및 3B 파라미터를 지닌 Qwen3 모델에서 Depth-Attention은 낮은 perplexity와 높은 평균 downstream accuracy를 달성했습니다. vanilla Transformer에 비해 최대 2.3 포인트의 accuracy 향상을 이루었으며, 추가 연산은 0.01% 미만입니다. 이 방식은 hidden-state 기반 방법들보다 메모리 오버헤드가 적어 효율적인 성능을 보여줍니다.



### DAR: Deontic Reasoning with Agentic Harnesses (https://arxiv.org/abs/2606.05009)
- **What's New**: 이 논문은 Deontic Agentic Reasoning (DAR)이라는 새로운 접근 방식을 소개합니다. DAR은 모델이 규정에 따라 요구에 맞게 법령을 동적으로 검색할 수 있도록 하는 에이전틱(Agentic) 추론 설정입니다. 기존의 추론 방식은 모델이 모든 규칙과 사례를 하나의 프롬프트에서 함께 제공받는 방식이지만, DAR은 규정 파일을 독립적으로 관리하여 보다 효율적인 추론을 가능하게 합니다.

- **Technical Details**: DAR에서는 법령이 프롬프트에 포함되지 않고, 별도의 파일(statute.txt)로 저장되어 요청할 때마다 필요한 정보를 조회할 수 있습니다. 모델은 사례와 질문을 받고, 도구 호출을 통해 법령의 관련 부분을 동적으로 읽어냅니다. 이러한 방식은 모델이 상태를 쌓아가며 규정을 효과적으로 활용할 수 있도록 돕습니다.

- **Performance Highlights**: DAR 평가를 위해 DeonticBench에서 다양한 모델을 테스트한 결과, 최상위(frontier) 모델은 DAR를 통해 성능 향상을 보였으나, 약한(open-source) 모델은 성능이 악화되기도 했습니다. 예를 들어, GPT-5.2는 SARA-Numeric에서 30%에서 60%로 개선된 반면, Qwen3.5는 34%에서 11%로 감소했습니다. 이러한 결과는 에이전틱 환경이 모델의 능력에 따라 다르게 작용함을 시사합니다.



### GARL: Game-Theoretic Reinforcement Learning for Multi-Agent Strategic Prioritisation (https://arxiv.org/abs/2606.05002)
- **What's New**: 본 연구에서는 다중 에이전트 전략적 우선순위 지정을 위한 게임 이론 기반 강화 학습 프레임워크인 GARL(GAme-theoretic Reinforcement Learning)을 제안합니다. GARL은 우선순위 지정을 두 단계의 게임으로 공식화하며, 첫 단계에서는 에이전트가 공유 후보 집합에서 전략적 자원을 할당하고, 두 번째 단계에서는 최종 순위를 도출하는 상위 중재자가 있습니다. 이를 통해 복잡한 상호작용 구조를 정리하여 정책 최적화를 가능하게 합니다.

- **Technical Details**: GARL은 다중 에이전트에서 전략적 우선순위를 정립하기 위해 두 단계의 게임 구조를 적용합니다. 첫 번째 단계(Agenda Allocation)에서는 에이전트들이 후보 집합에서 전략적 자원을 할당하고, 두 번째 단계(Arbitration)에서는 중재자가 해당 자원을 기반으로 최종 순위를 매깁니다. 이 구조는 에이전트 간 상호작용의 형태에 따라 보상을 정의하여 MARL(멀티 에이전트 강화 학습)에서 정책 최적화를 더욱 체계적으로 접근할 수 있게 합니다.

- **Performance Highlights**: GARL은 법적 문제의 순위 매기기 작업에 적용되었으며, 실험 결과 실제 순위 성능을 개선시키고 있습니다. 또한, 작은 오픈 소스 LLM이 강력한 폐쇄형 LLM과의 경쟁력을 가지게 함으로써 법적 영역에서의 적합성과 광범위한 전략적 의사결정 능력의 향상을 보여줍니다. GARL은 상호작용 구조를 게임 이론적으로 활용하여 정책 최적화에 있어서의 원칙적인 접근법을 제시하며, 결과적으로 전략적 우선순위 지정을 위한 효과적인 프레임워크임을 입증합니다.



### DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving (https://arxiv.org/abs/2606.04987)
- **What's New**: 이번 논문에서는 여러 참가자들이 협력적으로 체스 퍼즐을 해결하는 대화 데이터셋인 DeliChess를 소개하고 있습니다. 이 데이터셋은 107개의 대화로 구성되어 있으며, 참가자들은 개별적으로 퍼즐을 해결한 후, 다자간 토론을 통해 집단 응답을 수정합니다. 기존의 데이터셋들은 구조화된 복합적 사고 과제에 중점을 두지 않았지만, DeliChess는 이 과제를 해결하기 위한 새로운 기회를 제공합니다.

- **Technical Details**: DeliChess 데이터셋은 3.4명의 참가자로 구성된 그룹이 평균 72회의 대화를 통해 세 가지 유형의 체스 퍼즐(포지셔널, 전투, 엔드게임)을 다룹니다. 참가자들은 개별적으로 퍼즐을 시도한 후,.live multi-party chat에서 토론을 하고 답변을 수정합니다. 이 데이터셋은 토론 전후의 선택과 퍼즐 난이도, 이동 품질에 대한 메타데이터를 포함하고 있습니다.

- **Performance Highlights**: 분석 결과, 토론을 통해 그룹의 퍼포먼스가 크게 향상되는 것으로 나타났습니다. 질문을 자극하는 발화(probing utterances)는 결과의 변동성을 증가시키지만, 반드시 모든 경우에서 성과를 개선하는 것은 아닙니다. 이 데이터셋은 단체 의사 결정 및 다자간 상호작용의 다이내믹스를 연구하는 데 중요한 자원이 될 것입니다.



### Probing Outcome-Level Resemblance and Mechanism-Level Alignment in LLM Risk Decisions: Evidence from the St. Petersburg Gam (https://arxiv.org/abs/2606.04978)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)이 위험 결정에서 겪는 복잡한 기계적 일치 문제를 조사합니다. 특히, St. Petersburg 게임을 사용하여 모델의 출력이 인간의 결정 메커니즘과 실제로 일치하는지, 아니면 단지 겉모습만 유사한지를 분석합니다. 연구 결과, 모델들은 일반적으로 겉으로는 인간과 비슷한 결정을 보이지만, 내재된 결정 메커니즘은 상이할 수 있음을 보여주고 있습니다.

- **Technical Details**: 연구팀은 여러 모델을 평가하기 위해 구조화된 프롬프트 세트를 구성했습니다. 이 프롬프트는 원래 게임, 잘린 게임, 반복적인 게임, 자산 조작 및 직업 정체성 변형 등을 포함하여 모델의 응답을 비교합니다. 연구 질문은 LLM이 실제로 인간과 비슷한 제한된 응답을 생성하는지, 수정된 게임 구조에서도 일관되게 인간의 결정 메커니즘을 따르는지를 탐구합니다.

- **Performance Highlights**: 연구 결과, 대부분의 LLM은 원래의 St. Petersburg 게임에서 낮고 유한한 입찰가를 생성하며, 이는 겉으로는 인간과 비슷한 신중한 행동처럼 보이지만, 이러한 일치는 적절한 메커니즘 검사 아래에서 종종 무너집니다. 인지적으로 조정된 프롬프트와 명령어 조정은 일부 극단적인 비인간적 응답을 줄이지만, 대부분의 기계적 반응 패턴은 변하지 않고 유지됩니다. 이는 LLM 평가의 방법론적 도전을 강조하며, 단순한 결과의 유사성을 넘어서 기계적 일관성을 검토해야 함을 나타냅니다.



### SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding (https://arxiv.org/abs/2606.04974)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 SAID(Scaffold-Aware Iterative Decoding)라는 새로운 프레임워크를 제안하여 확산 기반 대형 언어 모델(DLLMs)의 성능을 향상시킵니다. 이 방법은 먼저 스캐폴드 토큰(scaffold tokens)에 대한 연산을 집중하고, 이후 예측 가능한 세부 표현(detail tokens)을 적은 단계로 생성함으로써 효율성을 높입니다. 또한 Confidence-Hierarchical Layered Generation (CHLG)을 도입하여 저신뢰(low-confidence) 토큰에 추가 단계를 할당하고 고신뢰(high-confidence) 토큰을 빠르게 확정하는 전략을 수립했습니다.

- **Technical Details**: SAID는 기존의 고르게 분배된 디노이징(de-noising) 과정을 재고하여 스캐폴드 구성에 더 많은 연산을 할당하고, 예측 가능한 세부 토큰을 위해 적은 단계를 사용합니다. 이를 통해 생성 과정에서 각 토큰의 위치에 따라 다르게 할당되는 연산을 최적화할 수 있습니다. 블록 기반(block-wise) 확산 디코딩에 적응할 수 있도록 SAID를 수정하여 각 블록 내에서 스캐폴드 생성과 세부 생성 과정을 적용합니다.

- **Performance Highlights**: 실험 결과, SAID는 LLaDA-8B와 LLaDA 1.5 모델을 기준으로 수학, 코드 작성 및 지식 기반 벤치마크에서 인퍼런스(inference) 속도를 최대 9.1배 향상시키면서도 경쟁력 있는 생성 품질을 유지합니다. 이를 통해 SAID의 효율성을 입증하며, 기존의 디노이징 절차보다 현저히 낮은 지연 시간을 기록하였습니다.



### SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs (https://arxiv.org/abs/2606.04964)
Comments:
          Code: this https URL

- **What's New**: 이 논문은 SemBlock이라는 새로운 동적 블록 디코딩 프레임워크를 제안합니다. SemBlock은 고정 블록 크기 또는 구분 기호 기반의 신호를 사용하지 않고, 의미 경계(prediction of semantic boundaries)를 기반으로 블록을 동적으로 구성합니다. 이를 통해 DLMs의 텍스트 생성 과정을 더욱 효율적으로 개선할 수 있습니다.

- **Technical Details**: SemBlock은 Frozen LLaDA 상태에서 학습된 경량의 예측기를 사용하여 각 후보 위치에서 현재 의미 단위가 완성되었는지를 추정합니다. SemBound라는 데이터셋을 구축하여 언어, 수학, 코드 작업의 담론 단위, 추론 단계, 구현 범위를 통해 경계 레이블을 파생합니다. 주어진 후보 블록 윈도우에서 예측된 경계 확률을 사용하여, 각 동적 블록의 끝 위치를 선택합니다.

- **Performance Highlights**: GSM8K, IFEval, MATH, HumanEval 데이터셋에 대한 실험 결과, SemBlock은 기존의 고정 블록 디코딩 및 AdaBlock보다 일관되게 더 나은 성능을 보였습니다. 특히 LLaDA-1.5에서는 SemBlock이 HumanEval에서 최대 11.60 pass@1 포인트 향상된 결과를 기록했습니다. 이는 경계 위치를 SEMANTIC UNIT에 더 가깝게 설정하여 향상된 성능을 보여줍니다.



### Can Crowdsourcing Survive the LLM Era? A Community Survey on Human Data Collection (https://arxiv.org/abs/2606.04924)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 사용이 크라우드 소싱 데이터의 유효성에 미치는 영향을 조사하기 위해 자연어 처리(NLP) 및 관련 분야의 155명의 연구자들에게 설문 조사를 실시하였습니다. 데이터 수집 시 크라우드 워커들이 LLM을 사용하는 경우가 많으며, 이를 통해 수집된 데이터의 질에 대한 우려가 커지고 있습니다. 연구자들은 LLM 사용을 탐지하기 위한 다양한 전략을 보고했으며, 이러한 문제를 해결하기 위한 조치를 취하고 있지만 현재의 노력은 불충분하다는 점을 강조하였습니다.

- **Technical Details**: 크라우드 소싱은 큰 집단의 개인으로부터 다양한 데이터를 수집하는 데 사용되는 방법으로, NLP, 행동 과학 및 심리학 등 다양한 분야에서 활용됩니다. 그러나 LLM의 광범위한 사용은 크라우드 소싱 데이터의 질과 해석 가능성에 중대한 도전을 제기하고 있으며, 크라우드 워커들이 생성 도구를 사용하여 고품질의 텍스트를 빠르게 생산할 수 있게 되면서 이 문제가 가중되고 있습니다. 저자들은 LLM의 사용이 크라우드 소싱 연구에서 제공하는 진정한 인간 관점을 제한할 수 있다고 경고하였습니다.

- **Performance Highlights**: 설문 조사에 따르면, 응답자의 44%는 자신이 수집한 데이터에서 LLM 사용을 관찰했으며, 93%가 이를 예상했음에도 불구하고 적절한 예방 조치를 모르고 있었습니다. 응답자들은 독특한 텍스트 스타일 패턴과 비정상적으로 빠른 완료 시간을 주요 탐지 전략으로 평가하였습니다. 연구자들은 LLM이 크라우드 소싱 데이터의 질에 미치는 영향을 인지하고 있으나, 이 문제를 해결하기 위한 구체적인 실행 가능 방안에 대한 합의는 부족한 상황입니다.



### Caliper: Probing Lexical Anchors versus Causal Structure in LLMs (https://arxiv.org/abs/2606.04915)
- **What's New**: 이번 논문에서는 Caliper라는 새로운 방법론을 소개합니다. 이 방법론은 언어 모델이 구조적 인과 추론(structural causal reasoning)을 수행하는지, 아니면 단순히 레퍼런스 패턴을 매칭하는지 분석하기 위해 설계되었습니다. Caliper는 의미 있는 변수 이름을 자리 표시자 토큰으로 변경하면서 인과 구조를 보존하는 통제된 변형을 제공합니다.

- **Technical Details**: Caliper는 내부 변수(됴내적 변수)와 외부 노이즈(외생 노이즈), 구조적 방정식, 그리고 노이즈 분포를 포함하는 구조적 인과 모델(SCM)을 기반으로 합니다. 이 모델은 Pearl의 인과 계층 구조를 따르고, 인과 질의의 맥락에서 일관된 평가를 가능하게 합니다. Caliper를 통해 각 질의의 인과 그래프를 유지하면서도 표면적인 어휘(content)를 제거함으로써, 모델의 인과적 추론 능력을 측정할 수 있습니다.

- **Performance Highlights**: 시험 결과, 14개의 instruction-tuned LLM을 사용하는 동안, 표면 어휘가 제거되면 인과적 정확도에서 유의미한 감소가 발견되었습니다. 특히 CRASS와 e-CARE 등 여러 벤치마크에서 정확도 저하가 두드러졌으며, 이는 LLM들이 구조적 인과 추론 대신 레퍼런스 기반으로 작동하고 있다는 것을 시사합니다. 이 연구 결과는 대규모 언어 모델의 인과적 성능을 평가하는 데 중요한 통찰을 제공합니다.



### 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions (https://arxiv.org/abs/2606.04906)
- **What's New**: 이번 논문은 AI가 생성한 텍스트가 발생시키는 광범위한 사회적 위험성을 다루며, AI 생성 텍스트 탐지(AITD)에 대한 명확한 정의의 필요성을 강조합니다. 현재의 데이터셋과 접근법들이 각각의 고유한 기준을 정의하고 있지만, 이는 실제 적용 가능성과는 거리가 멀다는 점에 착안했습니다. 이 연구는 AI 생성 텍스트에 대한 다양한 개념과 그 특성을 체계적으로 정의하고, AITDNA라는 새로운 벤치마크 데이터셋을 구축했습니다.

- **Technical Details**: AITDNA는 99명의 인간 저자가 공동으로 작성한 350개 이상의 텍스트로 구성되어 있으며, 이 과정에서 발생하는 인간과 AI 간의 상호작용을 상세히 기록합니다. 이러한 데이터셋은 특정 환경에서 발생하는 인간-AI 공동 저작의 복잡성을 반영하며, 기존 문헌에 기반한 AITD 개념을 확장하여 내용 기반 및 저자 ID 기반의 새로운 개념을 추가했습니다. 이 연구는 다양한 AI 탐지기와 AITD 관련 데이터셋의 미세 조정을 통해 성능 평가의 일관성을 강화할 수 있는 방법을 제시합니다.

- **Performance Highlights**: AITDNA 데이터셋을 사용하여 기존 AITD 데이터셋의 숨겨진 가정과 인간-AI 공동 저작과의 정렬을 평가했습니다. 기존 AITD 데이터셋들은 자주 비현실적인 가정을 내포하고 있으며, 특정 개념의 탐지가 어렵다는 사실이 드러났습니다. 연구 결과, 각 개념별로 탐지기의 성능을 비교하고, 이러한 가정들이 성능 평가에 미치는 영향을 실증적으로 추정하여 AITD에 대한 더 정교한 접근법의 필요성을 제안합니다.



### GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning with Verifiable Rewards (https://arxiv.org/abs/2606.04889)
- **What's New**: 이 논문에서는 Reinforcement Learning (강화 학습)에서의 보상(Rewards)에 대해 논의하고, 기존의 방법들이 모든 토큰에 동일한 시퀀스 수준의 이점을 배포하거나 프로세스 보상 모델(Reward Models)을 사용해야 했던 점의 한계를 설명합니다. 제안된 방법인 GRAIL(Gradient-Reweighted Advantage)은 각 토큰의 가치에 따라 이점을 재조정함으로써 논리적 추론 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: GRAIL은 gradient-activation saliency를 활용하여 최종 답변에 더 민감한 토큰에 더 많은 가중치를 부여합니다. 이러한 접근은 모든 토큰이 동등하게 최종 보상에 기여한다고 가정하는 전통적인 방식과 차별화됩니다. GRAIL은 각 모델에서의 성과 평가를 통해 효과적인 토큰 단위 이점 조정 메커니즘을 구현합니다.

- **Performance Highlights**: 실험 결과, GRAIL은 Qwen3, R1-distilled 및 OctoThinker 모델을 포함한 5개 모델에서 GRPO보다 일관되게 우수한 성능을 보였습니다. GRAIL은 정확도에서 평균 3.60% 및 Pass@3에서 3.05%의 향상을 달성하여, 프로세스 수준의 감독 없이도 세밀한 추론 정렬이 가능함을 입증하였습니다.



### Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean (https://arxiv.org/abs/2606.04883)
- **What's New**: 이 논문은 Lean에서의 정리 증명을 위한 동작 라우팅 에이전트를 제안합니다. 이 에이전트는 데이터 평면(data plane)과 제어 평면(control plane)으로 구성되어 있으며, 이전의 실패한 증명 시도를 관찰하여 성공 가능성과 비용을 추정합니다. 기존의 고정 단계 정책(fixed-step policies)은 비효율적이며, 이 에이전트는 불필요한 시도를 줄여 비용과 품질 간의 균형을 잘 맞추는 스킬을 가지고 있습니다.

- **Technical Details**: 이 논문의 데이터 평면은 자연어로 된 렘마(decomposition)로 문제를 분해하고, 해당 렘마를 Lean에서 정식으로 표현하여 증명 시도를 생성합니다. 제어 평면은 이전의 증명 경로를 관찰하여 다음 시도의 성공 확률과 비용을 평가합니다. 이 라우팅 방법을 통해 장기적인 비용-품질(tradeoff) 결정을 동적으로 내릴 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 에이전트는 PutnamBench의 문제 집합에서 평균적으로 25.8%의 비용 절감을 달성했습니다. 또한 동일한 비용에서 7.8%의 정확도 향상을 이뤘습니다. 이러한 결과는 고정 단계 정책(fixed-step policy)의 비효율성을 극복하고 성능을 유지하면서도 계산 비용을 줄일 수 있음을 보여줍니다.



### Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents (https://arxiv.org/abs/2606.04874)
- **What's New**: 이번 논문에서는 새로운 진단 벤치마크인 	extbf{Agent Planning Benchmark (APB)}를 소개하여, 4,209개의 다중 모드 사례를 22개의 도메인과 5개의 설정으로 나누어 평가합니다. APB는 계획 수립에 중점을 두어 전체 계획, 피드백 조건 하의 단계별 계획 및 도구의 신뢰성을 평가합니다. 이 벤치마크는 실패 원인을 명확히 분석하며, 이전의 평가 방식과는 다른 접근 방식을 제공합니다.

- **Technical Details**: APB는 4,209개의 테스트 사례를 사용하여 Holistic Planning과 Step-wise Planning을 통해 계획 능력을 다양한 방면에서 평가합니다. Holistic Planning은 모델이 전체 솔루션을 구성하는 능력을 요구하며, Step-wise Planning은 실행 경과에 따라 추론을 기반으로 한 다음 행동을 예측하도록 조건을 부여합니다. APB는 도구 사용 시의 다양한 상황을 반영하여, 고장 난 도구나 해결할 수 없는 문제에 대한 저항력을 강조합니다.

- **Performance Highlights**: 12개의 다중 모드 대형 언어 모델(MLLM)에서의 평가 결과, 긴 수평 계획에 대한 체계적인 약점이 발견되었고, 도구 노이즈와 같은 외부 요인에도 취약함을 알 수 있었습니다. APB를 활용한 정제된 계획은 실행 성과 지표를 일관되게 개선하여, 계획 수립-실행 과정 간의 연결성을 보여줍니다. 이러한 결과는 APB가 실행 벤치마크의 상위 진단 신호로서 기능함을 입증합니다.



### Large Language Models in K-12 Education: Alignment with State Curriculum Standards and Student Personas (https://arxiv.org/abs/2606.04846)
- **What's New**: 이번 연구는 미국 내 여러 주의 역사 커리큘럼 차이를 탐색하고, 대규모 언어 모델(LLMs)이 이러한 차이를 어떻게 반영하는지를 분석합니다. 특히 LLM의 응답이 학생의 지역, 학년, 성별 및 인종 같은 인구통계적 특성에 따라 어떻게 변화하는지를 평가합니다. 이러한 맥락에서 LLM의 사용이 교육적 결과에 미치는 잠재적 영향을 고려하고 있습니다.

- **Technical Details**: 이 연구에서는 RAG(Retrieval-Augmented Generation) 기반 프레임워크를 설계하여 각 주의 커리큘럼을 분석하고, LLM의 응답이 특정 주의 커리큘럼 기준과 얼마나 일치하는지를 평가하기 위한 메트릭스를 설정합니다. 연구의 첫 번째 단계로, 각 주가 다루는 주제를 파악하고, 두 번째 단계로는 다양한 LLM이 주별 커리큘럼에 얼마나 잘 부합하는지를 측정합니다.

- **Performance Highlights**: 실험 결과, LLM은 학생의 학년 수준에 맞춰 정보를 조정할 수 있지만, 인종이나 성별에는 최소한의 민감성을 보였습니다. 그러나 LLM의 응답은 주의 정치적 성향에 따라 변화할 수 있으며, 이는 실제 커리큘럼 내용과 반드시 일치하지 않습니다. 이러한 결과는 LLM이 교육적 기준과 얼마나 잘 일치하는지에 대한 더 강력한 정렬 기법의 필요성을 강조합니다.



### A French Corpus Annotated for Multiword Expressions with Adverbial Function (https://arxiv.org/abs/2606.04828)
- **What's New**: 이 논문에서는 부사적 기능을 가진 다중 단어 표현(Multiword Expressions, MWEs)을 주석으로 추가한 프랑스어 말뭉치(corpus)를 소개합니다. 이 말뭉치는 정보 검색(information retrieval) 및 추출(extraction), 그리고 깊은(deep)과 얕은(shallow) 구문 파싱(syntactic parsing)에 대한 연구를 위해 설계되었습니다.

- **Technical Details**: 논문에서는 주석을 추가한 MWEs의 종류를 구분하고, 주석 작업에 사용한 자원(resources)과 방법(methods)을 설명합니다. 또한, 주석 결과에 대해 간략하게 언급하고 있습니다. 이 주석이 추가된 말뭉치는 LGPLLR 라이센스 하에 제공됩니다.

- **Performance Highlights**: 이 연구의 결과는 정보 검색 및 구문 분석 분야에서 다중 단어 표현의 중요성을 강조합니다. 주석된 말뭉치의 사용으로 연구자들은 MWEs의 처리에 있어서 더 나은 성능을 기대할 수 있습니다.



### PersonaTree: Structured Lifecycle Memory for Person Understanding in LLM Agents (https://arxiv.org/abs/2606.04780)
- **What's New**: 최근의 LLM(Large Language Model) 에이전트는 장기간 동안 같은 사용자와 상호작용하는 것을 요구받고 있습니다. 이를 위해서는 기억이 단순한 사실의 회상을 넘어 사용자의 행동과 선호를 추적하고 안정적인 판단을 형성할 수 있어야 합니다. 기존의 기억 연구는 정보를 보유하고 활용하는 데 초점을 맞추고 있지만, 이 논문은 지속적인 상호작용을 통한 사용자 이해를 위한 새로운 메모리 프레임워크인 PersonaTree를 제시합니다.

- **Technical Details**: PersonaTree는 사용자의 프로필을 세 가지 수준의 persona tree로 구성하여 기억을 구조화합니다. 각 잎(leaves) 노드는 특정 시점의 상호작용 증거를 저장하고, 중간(mid) 노드는 반복되는 행동 패턴을 캡처하며, 루트(root) 노드는 안정적인 개인 수준의 주장을 표현합니다. 이러한 구조를 통해 사용자의 행동과 선호에 대한 더 깊은 이해를 가능하게 하며, 메모리는 증거로부터 추론할 수 있는 지원 경로를 제공합니다.

- **Performance Highlights**: PersonaTree는 여러 가지 사용자 이해 및 지속적 기억 벤치마크에서 뛰어난 성능을 보이며, 18개의 컴팩트 점수 중 12개에서 1위를 차지했습니다. 구조적 계층화가 사용자의 추론력을 향상시키는 동시에, 지원 경로의 검색이 특정 질문에 대한 효과적인 답변을 도와줍니다. 이러한 분석을 통해 PersonaTree의 메모리가 장기간의 사용자 경험을 효과적으로 지원함을 입증하였습니다.



### TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration (https://arxiv.org/abs/2606.04743)
- **What's New**: 본 논문은 TIDE라는 새로운 프레임워크를 소개하며, 이는 사용자의 명시적 요청에 의해 드러나지 않은 숨겨진 문제를 발견하는 것을 목표로 합니다. TIDE는 두 가지 보완적 메커니즘으로 구성되어 있으며, 반복적인 발견(iterative discovery)과 사고 템플릿(thought templates)을 통해 효과적인 문제 해결을 지원합니다. 이 프레임워크는 개인 작업 공간과 소프트웨어 저장소 등 두 가지 현실적 설정에서 검증되었으며, 단일 예측(single-shot) 및 병렬 다중 에이전트(parallel multi-agent) 기준보다 현저한 성과를 보였습니다.

- **Technical Details**: TIDE 프레임워크는 반복적인 발견과 사고 템플릿을 결합하여 작동합니다. 반복적인 발견은 매 라운드마다 소규모 후보군을 제시하며, 이전에 발견한 내용을 기준으로 후속 라운드가 진행되어 문제 범위를 확대합니다. 사고 템플릿은 이전에 해결된 사례에서 추출된 재사용 가능한 형식(schema)으로, 문제 유형에 대한 구체적인 지침을 제공합니다.

- **Performance Highlights**: TIDE는 네 가지 LLM 모델 백본을 사용하여 개인 작업 공간과 소프트웨어 저장소에서 검증되었으며, 작업 범위, 문제 식별, 문제 해결에서 일관되게 더 나은 성과를 나타냈습니다. 이 결과들은 TIDE가 기존의 단일 요청 기반 접근 방식보다 더 효과적인 발견 과정을 통해 사용자에게 보다 유용한 지원을 제공할 수 있음을 시사합니다.



### Multilingual Long-Form Speech Instruction Following: KIT's Submission to IWSLT 2026 (https://arxiv.org/abs/2606.04730)
Comments:
          9 pages main paper, IWSLT 2026 Instruction Following track

- **What's New**: 이 논문은 최근 대규모 언어 모델의 발전에 따라 IWSLT 2026의 장단기 지시 수행 트랙에 대한 KIT의 기여를 소개합니다. 새로운 데이터 증강 기법을 통해 단기 음성 데이터셋을 장기로 변환하고, 다양한 태스크와 언어에 대한 확장 가능성을 보여줍니다. 특히 1백만 이상의 데이터를 생성하여 실제 작업에 대한 성능을 향상시킴을 선언하고 있습니다.

- **Technical Details**: KIT는 AMR 기반의 레이블 생성과 다국어 참조 번역을 활용하여 인식된 음성을 다양한 태스크에서 사용할 수 있도록 하는 데이터 증강 프레임워크를 구현하였습니다. 이 논문에서는 기계 번역 모델을 통해 전이된 라벨을 사용하여 단기 음성 데이터를 장기 지시 수행에 맞게 조정하는 방법을 설명합니다. 또한, 데이터 간 혼합 방식에 대한 실험을 통해 멀티모달 음성 지시 수행에서 최적의 전략을 찾아냈습니다.

- **Performance Highlights**: KIT는 ASR(Automatic Speech Recognition) 및 SQA(Spoken Question Answering)와 같은 다양한 태스크에서의 성능을 비교하였으며, 새로운 리랭킹 전략을 적용하여 의미 기반 태스크에서의 성능 감소를 효과적으로 줄였습니다. 긴 오디오 데이터를 학습시킴으로써 이전 모델에 비해 성능을 크게 향상시킬 수 있음을 보였고, 이는 특히 QA와 요약 작업에서도 긍정적인 결과를 가져왔습니다.



### Query-based Cross-Modal Projector Bolstering Mamba Multimodal LLM (https://arxiv.org/abs/2606.04719)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 본 논문에서는 Mamba라는 Selective Scan Structured State-Space Model(SSM)을 제안하여 기존 Transformer의 한계를 극복합니다. 입력 길이에 따른 제곱 복잡성 문제를 해결하기 위한 쿼리 기반의 크로스 모달 프로젝터를 소개하여 비전-언어 모델링의 효율성을 강화합니다. 이 프로젝터는 기존 이미지 특징을 입력 시퀀스로 변환할 때 2D 스캔 순서를 수동으로 설계할 필요를 제거하며, 실험 결과는 Mamba 기반의 다중 모달 LLM이 성능 및 처리량을 향상시키는 데 기여함을 보여줍니다.

- **Technical Details**: MLLM(Multimodal Large Language Models)은 텍스트와 이미지를 포함한 다양한 모달리티의 기능을 확장하기 위해 설계되었습니다. Mamba는 입력 의존적 게이팅 메커니즘과 하드웨어 인식을 기반으로 한 알고리즘을 통합하여 선택적 스캔을 가능하게 합니다. 이 연구는 Mamba 아키텍처를 기반으로 한 쿼리 기반의 크로스 모달 프로젝터를 통해 미리 훈련된 비전 인코더와 Mamba 모델 간의 효과적인 연결을 제안합니다.

- **Performance Highlights**: 제안된 모델은 비전-언어 이해 벤치마크에 대한 철저한 실험 평가를 통해 Mamba 기반의 LLM이 기존 모델에 비해 성능과 강건성을 향상시킨다는 것을 입증합니다. Mamba는 타당한 훈련 및 추론 속도를 유지하면서도 고급 Transformer와 어깨를 나란히 하거나 오히려 능가하는 성능을 보여줍니다. 자세한 비교 내용은 논문 내 그림에서 확인할 수 있습니다.



### Rethinking Continual Experience Internalization for Self-Evolving LLM Agents (https://arxiv.org/abs/2606.04703)
Comments:
          10 pages, 8 figures

- **What's New**: 이번 연구에서는 경험 내재화(experience internalization)가 LLM(대형 언어 모델)의 지속 가능한 학습에 대한 새로운 통찰을 제공하고자 한다. 기존 방법론은 단일 반복(iteration)에 집중된 반면, 다중 반복 경험 학습(multi-iteration experience learning)에서의 성능 붕괴 문제를 발견하였다. 이를 기반으로 경험 내재화를 강화하기 위한 세 가지 주요 차원을 체계적으로 분석하였다.

- **Technical Details**: 연구에서는 경험의 세분화(Experience Granularity), 경험 주입 패턴(Experience Injection Pattern), 내부화 체계(Internalization Regime)라는 세 가지 측면을 통해 경험 내재화의 문제를 분석하였다. 첫째, 원칙 수준(principle-level) 경험이 사례 수준(instance-level) 경험보다 더 지속 가능하다는 것을 확인하였고, 둘째, 단계적인 주입(step-wise injection)이 전반적인 주입(global injection)보다 우수하다는 점을 강조하였다. 마지막으로, 고품질의 교사 경로(teacher trajectories)를 활용한 오프 정책(context-distillation)이 더 안정적인 학습 신호를 제공함을 보였다.

- **Performance Highlights**: 연구 결과는 지속 가능한 경험 내재화의 간단하면서도 효과적인 방법론을 제공하며, LLM이 스스로 발전(self-evolve)할 수 있는 가능성을 제시한다. 이러한 성과는 경험 기반의 자기 진화(self-evolution)를 통해 LLM이 반복(iteration)을 거쳐 더 나은 성능을 발휘할 수 있도록 만들어준다. 또한, 실험을 통해 얻은 이론적인 통찰은 향후 LLM 에이전트 설계에 실질적인 지침이 될 것으로 기대된다.



### DuDi: Dual-Signal Distillation with Cross-Lingual Verbalizer (https://arxiv.org/abs/2606.04694)
- **What's New**: 본 논문에서는 DuDi라는 새로운 다중 신호 다국어 증류 프레임워크를 소개합니다. DuDi는 온라인 시퀀스 수준 신호와 오프 정책 및 온 정책의 토큰 수준 신호를 결합하여 언어 모델의 다국어 성능을 개선합니다. 이 프레임워크는 다양한 모델 크기와 설정에서 최대 1000만 명 이상의 사용자에게 제공되는 동남아시아어 성능을 향상시키도록 설계되었습니다.

- **Technical Details**: DuDi는 시퀀스 신호, 토큰 신호, 그리고 크로스-링구얼(다국어) 비서로 구성되어 있습니다. 시퀀스 신호는 학생 정책을 실제 정답으로 유도하는 온라인 시퀀스 수준 목표를 제공합니다. 토큰 신호는 학생 생성 응답에서 나오는 온 정책 신호와 훈련 말뭉치에서 나오는 오프 정책 신호를 사용하여 추가적인 학습 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, DuDi는 Qwen2.5-0.5B 설정에서 가장 우수한 성능을 달성하였고 대부분의 동남아시아 언어에서 성과 향상을 보였습니다. DuDi 구성 요소 중 어느 하나라도 제거 시 성능이 감소하는 것을 확인하였으며, 이는 시퀀스 수준 목표, 이중 정책 토큰 신호 및 크로스-링구얼 비서의 공동 최적화 필요성을 강조합니다. 또한 DuDi 비서는 학생-교사 증류에서 더욱 풍부한 학습 신호를 제공합니다.



### SMADE-IE: Sparse Multi-Agent Framework with Evidence-Driven Debate for Zero-Shot Information Extraction (https://arxiv.org/abs/2606.04691)
Comments:
          21 pages, 9 figures

- **What's New**: 이번 연구는 SMADE-IE라는 새로운 sparse multi-agent framework를 제안합니다. 이 프레임워크는 zero-shot 정보 추출(IE)을 위해 설계되었으며, 입력의 복잡도에 따라 Global Extraction Mode와 Type-Centric Extraction Mode로 분기하여 불필요한 선택과 추론을 줄입니다. 이 연구는 또한 Toulmin-style 구조의 Evidence-Driven Debate 메커니즘을 도입하여 논쟁의 조직을 개선합니다.

- **Technical Details**: SMADE-IE는 Adaptive Mode Selector를 통해 각 입력을 동적으로 분류하고, Type-Centric Extraction Mode 내에서 발생하는 크로스-타입 충돌을 해결하기 위해 Evidence-Driven Debate 모듈을 활용합니다. SMADE-IE는 NER, RE, JERE 작업에서 성능 일관성을 유지하기 위해 엔티티와 관계 간의 관계를 동기화하는 Iterative Entity–Relation Alignment 단계도 포함합니다. 이 과정을 통해 각 엔티티와 관계가 올바르게 연결됩니다.

- **Performance Highlights**: 실험 결과 SMADE-IE는 9개의 벤치마크 데이터세트에서 기존 zero-shot IE 모델들보다 일관되게 우수한 성능을 보였으며, 토큰 효율성 또한 향상되었습니다. 다양한 복잡도의 입력 상황에서 SMADE-IE는 성능과 효율성의 균형을 효과적으로 유지하며, 이를 통해 다양한 엔티티와 관계 유형을 정확하게 추출할 수 있음을 입증하였습니다.



### CRAFT: Cost-aware Refinement And Front-aware Tuning of Prompts (https://arxiv.org/abs/2606.04661)
- **What's New**: CRAFT(비용 인식 정제 및 프론트 인식 조정)는 정확도와 프롬프트 토큰 비용 간의 균형을 최적화하는 새로운 접근 방식을 제안합니다. 이 방법은 여러 후보의 정확도를 평가하기 위한 target-LLM 검증 호출을 자원으로 고려하고, 이를 통해 후보들이 보다 낙관적인 후보 프론트에 가까워지도록 합니다. 기존의 단순 가중치 합성 방법과는 달리 CRAFT는 비용과 정확도 간의 상호작용을 보다 깊이 이해하고 직접적으로 최적화합니다.

- **Technical Details**: CRAFT는 세 가지 주요 단계로 구성된 최적화 루프를 통해 작동합니다. 첫 번째 단계에서는 구조 인식을 기반으로 한 정제기(refiner)와 응축기(condenser)가 새로운 후보를 제안합니다. 두 번째 단계에서는 Pareto-gap 획득(Pareto-gap acquisition) 기능이 최적 후보 프론트에 가까운 후보에게 예산을 할당하고, 세 번째 단계에서는 다양한 후보를 유지하기 위한 선택 기능이 작동합니다.

- **Performance Highlights**: CRAFT는 여섯 개의 분류 및 추론 벤치마크에서 높은 정확도와 낮은 비용을 동시에 유지하며, 기존의 정확도 중심, 비용 중심, 가중합 중심의 방법들이 좁은 영역에 집중하는 것과 대비되는 장점을 보여줍니다. 이로 인해 CRAFT는 다수의 유효한 프롬프트를 보존하며, 각 테스트에 대해 높은 정확도와 낮은 비용을 달성할 수 있었습니다.



### LifeSide: Benchmarking Agents as Lifelong Digital Companions (https://arxiv.org/abs/2606.04660)
Comments:
          28 pages, 23 figures, 7 tables

- **What's New**: 이번 논문에서는 기존의 평가 방식이 인간과의 지속적인 관계를 구축하는 디지털 동반자(Companions)의 본질적인 요구 사항을 포착하지 못하고 있다는 점에 착안하여, 새로운 벤치마크인 LifeSide를 제안합니다. LifeSide는 다중 세션(Multi-session) 메모리-감정-환경(Memory-Emotion-Environment) 루프에 중점을 두고 구성되며, 사용자 경험을 지속적으로 업데이트하고 적응할 수 있는 능력을 평가하는 것을 목표로 합니다. 이는 디지털 동반자가 과거의 기억뿐만 아니라 현재의 정서와 환경을 동시에 통합하여 사용자에게 개인화된 지원을 제공해야 함을 강조합니다.

- **Technical Details**: LifeSide는 2,000개의 인구 통계 기반 페르소나(Personas)와 111K 개의 작업(Task)을 포함하는 대규모 데이터셋으로 구성되어 있습니다. 각 페르소나는 사용자 프로파일(User profile), 사건의 이력(Event trajectory), 사회적 관계(Social relations), 장기 목표(Long-term goals), 그리고 외부 환경 조건(Exogenous environmental conditions) 등을 포함한 사용자 세계(User world)를 형성합니다. 벤치마크된 에이전트는 이 세계와 시간에 따라 상호 작용하면서, Partially Observable Markov Decision Process(POMDP)를 설정하여 정보의 격차를 효과적으로 처리합니다.

- **Performance Highlights**: 실험 결과는 현재의 메모리 벤치마크를 초과하는 모델조차도 장기적으로 사용자 이해를 유지하고 진정한 동반 관계를 형성하는 데 실패하고 있음을 나타냅니다. 즉, 장기 메모리에 대한 낮은 한계는 약 52%의 평균을 보여주며, 정보 검색을 통한 통합은 공감과 조절을 22%-44%까지 저하시킬 수 있습니다. 이러한 한계들은 디지털 동반자가 직면하고 있는 다양한 도전 과제를 명확히 드러냅니다.



### QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples (https://arxiv.org/abs/2606.04646)
Comments:
          14 pages

- **What's New**: 이 논문에서는 데이터베이스 스타일 쿼리와 유사한 자연어 질문에 대한 응답을 평가하기 위한 진단 벤치마크인 QO-Bench를 소개합니다. 기존의 검색 보강 생성(Retrieval-Augmented Generation, RAG) 시스템은 주로 의미론적 관련성에 최적화되어 있었으나, 실제 쿼리 실행의 정확성을 보장하지 않는다는 문제점을 지적합니다. QO-Bench는 특정 이벤트의 쿼리 연산자를 보존하는 검색을 목표로 하며, 22,984개의 뉴스 기사와 614개의 기업 사건을 바탕으로 785개의 질문을 평가합니다.

- **Technical Details**: QO-Bench는 이벤트 튜플에 대한 설정에서 쿼리-연산자 질문 응답(QO-QA)을 재정립합니다. 각 gold answer는 이벤트 튜플에서 결정적으로 계산되며, 템플릿 특정 리콜에 따라 점수를 매깁니다. 이 디자인은 각 실패를 특정 연산자로 귀속시킬 수 있도록 하여, RAG, ReAct RAG, GraphRAG, 정보 추출-SQL 시스템을 비교 분석합니다. 또한, 두 축 프레임워크인 인덱스-시간 보존과 쿼리-시간 실행을 통해 각 패러다임의 실패 요인을 파악합니다.

- **Performance Highlights**: 실험 결과 각 패러다임은 특정 연산자에서 우월성을 보이지 않으며, 유사성 검색은 필터/프로젝트에서 강점을 보이는 반면, 정보 추출-SQL 시스템은 교차 사건 조인에 약점을 보입니다. QO-Bench는 연산자 실행이 핵심 병목임을 시사하며, 이는 단순한 검색 대답 생성만으로는 개선되지 않습니다. 또한, Gold evidence를 이용한 긴 컨텍스트 오라클조차도 한계에 도달하지 못하였고, 이는 연산자 실행의 중요성을 부각시킵니다.



### CYGNET: Cypher Gate for Neural Execution Triage and Cost Containmen (https://arxiv.org/abs/2606.04645)
- **What's New**: 이번 연구에서는 Cypher 쿼리를 생성하는 언어 모델이 데이터베이스에서 구조적 또는 의미적 오류를 발생시키지 않도록 하기 위한 사전 실행 게이트(pre-execution gate)를 도입하였습니다. 이 게이트는 쿼리의 구조를 검증하고, 마지막 단계로 미러 그래프(mirror graph)에서 실행됩니다. 연구팀은 구조적 오류를 탐지하고 교정하여 쿼리 생성 과정에서 발생할 수 있는 문제를 사전에 방지하는 시스템을 구상하였습니다.

- **Technical Details**: 제안된 게이트는 네 개의 백엔드(backend) 체계를 사용하여 쿼리를 검증합니다. 이 체계에는 정규 표현식(regex) 필터, ANTLR(킥) 기반의 추상 구문 트리(ANTL) 파서, Neo4j EXPLAIN 백엔드가 포함됩니다. 이러한 체계는 구조적 오류를 검출하고, 오류의 종류를 분류하며, 각 오류에 대한 정보를 구조화된 형식으로 리턴하여 후속 시스템이 이를 활용할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, 이 시스템은 1135개의 쿼리를 통해 100%의 구문 오류, 제약 조건 위반 및 스키마 참조 오류를 성공적으로 탐지하였습니다. 구조적 오류 교정의 성공률은 81%에서 95%에 이를 정도로 뛰어나며, 평균 89%의 성과를 기록했습니다. 이 시스템은 7개의 CypherBench 스키마에서 테스트한 모든 모델의 생성 정확도를 유지하며 안전한 방어층으로서 기능함을 입증하였습니다.



### RAMPART: Registry-based Agentic Memory with Priority-Aware Runtime Transformation (https://arxiv.org/abs/2606.04628)
- **What's New**: RAMPART는 LLM 기반 에이전트를 위한 컴파일 타임 메모리 모델과 순수 인-램( in-RAM) 블록 레지스트리입니다. 이 시스템은 프로그래머블 런타임 작업을 통해 명시적인 정책에 따라 내용을 조정하여 메모리를 최적화합니다. 다섯 가지 조합 가능 원시(primitives)인 promote, gate, write, evict, rollback은 프롬프트 토큰 비용 없이 블록에 작용하여 성능 향상을 추구합니다.

- **Technical Details**: RAMPART의 핵심 단위인 Instruction Block(IB)은 Behavior directive, Tool schema, 학습된 휴리스틱을 포함한 자연어 문자열로 구성됩니다. 각 블록은 고유한 식별자와 함께 출처 및 저자를 기록하며, 제어 가능성을 통해 블록의 삭제 여부를 결정합니다. 블록 레지스트리는 일반적으로 사용되는 데이터베이스로부터 독립적이며, 디스크 I/O가 발생하지 않아 시스템 성능을 크게 향상시킵니다.

- **Performance Highlights**: RAMPART의 제어된 프로브는 컴파일 타임 위치 조정이 작업 성공에 영향을 미친다는 것을 보여줍니다. 또한, relevance gating을 통해 프롬프트 비용을 67.8% 줄이고 성공률을 83% 회복하는 결과를 나타내었습니다. 크로스 모델 복제 실험에서 RAMPART의 성능은 다양한 모델에서 일관성을 보여주었으며, 블록 그룹화는 Mistral의 평균 통과율을 약 5배 증가시켰습니다.



### Hybrid Adversarial Defence for Natural Language Understanding Tasks (https://arxiv.org/abs/2606.04612)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 환각(hallucination)과 적대적 공격(adversarial manipulation)이 서로 밀접하게 연결되어 있다는 점에 주목하며, 이를 해결하기 위한 하이브리드 방어 프레임워크를 제안합니다. 이 프레임워크는 엔트로피 기반 모델(Entropy-based models), 불확실성 모델(Uncertainty-based models) 및 기하학적 모델(Geometric-based models)을 결합하여 모델의 견고성을 높입니다. 연구 결과, 제안된 하이브리드 모델은 NLU 데이터셋에서 최고의 성능을 보였으며, 기존 모델들에 비해 향상된 정확성과 강력한 방어력을 나타냈습니다.

- **Technical Details**: 하이브리드 적대적 방어 프레임워크는 엔트로피 기반 모델, 불확실성 모델, 기하학적 모델의 세 가지 개별 방어 기법으로 구성됩니다. 각 입력 쿼리는 이 세 가지 모델에 전달되고, 모델들은 각각의 특성을 활용하여 라우팅 알고리즘을 훈련시킵니다. 이를 통해 입력 쿼리에 대한 적절한 방어 결정을 내리며, 해당 결정은 특성의 다양성과 관련된 정보를 기준으로 이루어집니다. 또한, 하드 전문가 선택(hard expert selection)과 소프트 확률적 집계(soft probabilistic aggregation) 방법을 모두 탐색합니다.

- **Performance Highlights**: 실험 결과, 제안된 하이브리드 모델은 NLU 작업에서 최대 43.34%의 정확도 개선과 최대 64.92%의 적대적 강인성 향상을 보였습니다. 정의된 다양한 데이터셋에서, 하이브리드 모델은 단독 모델보다 우수한 성능을 보여, 특히 OUT-OF-DISTRIBUTION 데이터 및 프롬프트 주입(prompt injection) 공격에 대해서도 강력한 성능을 기록했습니다. 전반적으로 이 논문은 엔트로피, 불확실성 및 기하학적 특성을 결합하여 효과적인 방어 전략을 제공한다는 점을 강조합니다.



### A Systematic Evaluation of Positional Bias in Multi-Video Summarization with MLLMs (https://arxiv.org/abs/2606.04596)
- **What's New**: 이 논문은 멀티모달 대형 언어 모델(MLLMs)이 멀티 비디오 вход에서의 신뢰성이 잘 알려져 있지 않음을 강조합니다. 멀티 비디오 요약에서 위치 편향(positional bias)을 연구하고, 비디오의 입력 슬롯에 따라 요약의 질이 어떻게 달라질 수 있는지를 탐구합니다. 또한, ActivityNet과 뉴스 비디오 자료를 바탕으로 베이스라인을 구축하고 다양한 MLLMs의 성능을 평가하였습니다.

- **Technical Details**: 연구에서는 요약의 품질을 평가하기 위해 Coverage, Directional Positional Bias (DPB), Middle-Edge Gap (MEG)의 세 가지 보충 지표를 사용합니다. 실험은 Cooking, Domestic, Leisure 및 News 카테고리를 포함하여 두 개와 네 개의 비디오 입력에 대해 진행되었습니다. 각 비디오의 위치가 미치는 영향이 도메인과 모델에 따라 다름을 발견했습니다.

- **Performance Highlights**: 평가 결과, 위치 효과가 도메인-모델 의존적임을 확인했습니다. 특정 설정에서는 DPB가 거의 없지만 MEG가 부정적인 결과를 보였습니다. 이러한 결과는 위치에 따른 서로 다른 편향이 요약의 질에 미치는 영향을 제시하며, 멀티 비디오 요약 모델을 더 강력하게 만들기 위한 향후 연구의 필요성을 강조합니다.



### Fine-grained Fragment Retrieval in Multi-modal Long-form Dialogues (https://arxiv.org/abs/2606.04591)
- **What's New**: 본 논문에서는 Fine-grained Fragment Retrieval (FFR)라는 새로운 작업을 도입하여, 다중 모드(long-form multi-modal) 대화에서 의미적으로 일관된 발화-이미지 조각을 직관적으로 찾아내는 방법을 제안합니다. 기존 다이얼로그 검색 방식은 일반적으로 개별 발화나 이미지를 선택하는 데 초점을 맞추는 반면, FFR은 여러 대화 턴과 모드를 아우르는 일관된 의미 조각을 찾아 사용자 쿼리에 응답하려고 합니다. 이를 통해 복잡한 다중 모드 대화에서 유용한 정보를 보다 효율적으로 검색할 수 있게 됩니다.

- **Technical Details**: FFR을 지원하기 위해 MLDR이라는 대규모 다중 모드(long-form multi-modal) 대화 검색 데이터셋을 구축하였으며, 이는 평균 25.45턴으로 가장 긴 다이얼로그를 담고 있습니다. FFR은 단일 대화에서의 검색뿐만 아니라, 대화 코퍼스에서의 검색을 평가하기 위한 두 가지 설정을 탐구합니다. 이 과정에서 Fragment Embedding Model (FEM)을 사용하여 각 대화를 최소한의 의미적 단위로 분해하고, 쿼리에 대한 응답을 빠르게 검색할 수 있도록 구조화된 인덱싱을 적용합니다.

- **Performance Highlights**: F2RVLM과 FFRS는 MLDR과 실제 WeChat 기반 테스트 세트에서 모두 탁월한 성능을 입증하며 단일 대화 및 대화 코퍼스의 FFR 접근 방식에서 높은 정확도를 달성하였습니다. FFRS는 효율적인 검색과 의미적 정밀도 간의 균형을 적절히 유지하며, 실제 대화 시나리오에서도 효과적임을 보여줍니다. 이러한 성과는 FFR이 실제 응용 프로그램에서 유용하게 활용될 가능성을 시사합니다.



### VCIFBench: Evaluating Complex Instruction Following for Video Understanding (https://arxiv.org/abs/2606.04588)
- **What's New**: 본 논문에서는 비디오 이해를 위한 새로운 벤치마크인 VCIFBench를 소개합니다. 기존의 벤치마크가 단순한 프롬프트에 의존하고 있는 반면, VCIFBench는 복잡한 지시사항을 평가하는 데 초점을 맞춥니다. 이 벤치마크는 콘텐츠, 형식, 스타일 및 구조 요구사항을 포함한 다양한 제약 조건을 가진 지시사항을 생성합니다.

- **Technical Details**: VCIFBench는 306개의 만족 가능한 테스트 지시사항, 540개 쌍의 DPO(Direct Preference Optimization) 선호 데이터셋, 그리고 30개 항목의 충돌 진단 하위 집합을 포함하고 있습니다. 이를 통해 모델 출력은 하이브리드 검증 파이프라인(hybrid verification pipeline)을 사용하여 평가됩니다. 또한, 10개의 다중모달 대형 언어 모델(MLLMs)에 대한 실험 결과, 제약 조건 동시 충족이 여전히 어려운 문제임을 보여줍니다.

- **Performance Highlights**: VCIFBench 데이터에 대한 DPO 학습이 지시사항 수행 성능을 향상시킬 수 있음을 확인했습니다. 이는 다중모달 대형 언어 모델이 보다 복잡한 지시사항을 보다 효과적으로 따를 수 있게 하는 중요한 진전을 보여줍니다. 향후 이러한 연구 결과가 비디오 이해 분야의 발전에 기여할 것으로 기대됩니다.



### Cartridges at Scale: Training Modular KV Caches over Large Document Collections (https://arxiv.org/abs/2606.04557)
Comments:
          21 pages, 5 figures, 17 tables

- **What's New**: 본 연구는 Cartridges at Scale (CAS)라는 새로운 훈련 프레임워크를 소개합니다. CAS는 다수의 카드리지(cartridge) 학습을 가능하게 하여 문서 크기가 수백 개에 달하는 경우에도 저렴하게 다루고, 동적 방해 요소 혼합과 메모리 효율적인 예산 관리 기능을 제공합니다. 또한 CAS는 기존의 한정적인 카드리지 접근법의 성능 한계를 극복하여 효율적인 데이터 처리를 보장합니다.

- **Technical Details**: CAS는 카드리지 훈련을 위해 동적 혼합 및 회전 시스템을 도입하고, GPU와 지속적 저장소 간의 스왑을 지원합니다. 이러한 방식으로 수십 개의 카드리지를 고정된 GPU 메모리 예산 내에서 동시에 훈련할 수 있으며, 다양한 질문 생성을 위한 더 나은 데이터 생성 방법도 제안합니다. 또한, 독립적으로 훈련된 카드리지가 충돌을 일으키는 문제를 해결하고, 집중적인 훈련을 통해 카드리지 간의 결합 성능 향상을 보장합니다.

- **Performance Highlights**: CAS는 다수의 벤치마크에서 단일 카드리지보다 30점 이상 성능이 향상되었습니다. 또한, CAS는 기존의 검색 보강 생성(RAG) 방식보다 4배 적은 프롬프트 토큰을 소모하면서도 유사하거나 높은 정확도로 결과를 도출할 수 있습니다. 이 연구는 카드리지 기반의 학습 모델이 상업적 및 실용적 환경에 얼마나 잘 적응하는지를 보여줍니다.



### Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents (https://arxiv.org/abs/2606.04555)
- **What's New**: 이번 연구에서는 'Segment Tree Memory'(SegTreeMem)라는 새로운 기억 아키텍처를 제안합니다. SegTreeMem은 대화의 역사(history)를 주어진 시간순으로 분류하여 관리하며, 실시간으로 새로운 발언(utterance)을 삽입할 수 있도록 설계되었습니다. 기존의 기억 시스템이 주제적 유사성에 중점을 두는 것과 달리, 이 시스템은 발언의 시간 순서를 유지합니다.

- **Technical Details**: SegTreeMem은 발언을 메모리 구조에 통합하기 위해 오른쪽 단말 노드(rightmost frontier)에서 소규모의 노드를 업데이트합니다. 이 구조는 시간적 순서를 보존하고 계층적인 기억 세그먼트를 형성할 수 있도록 하며, 정보 검색(retrieval) 시에도 계층적 시간적 문맥(hierarchical temporal context)을 활용합니다. 이 메모리 아키텍처는 세 개의 장기 기억 벤치마크 및 두 개의 대형 언어 모델(LLM) 백본을 통해 검증되었습니다.

- **Performance Highlights**: SegTreeMem은 기존의 평면적 검색(flat retrieval), 그래프 구조 메모리(graph-structured memory), 트리 구조 메모리(tree-structured memory) 벤치마크에서 응답 품질을 향상시키는 성과를 보여주었습니다. 추가적인 시간적 순서(permutation analysis) 분석에서는 기억의 구조를 설정할 때 시간적인 순서가 성능 향상에 결정적인 요소임을 지적하였습니다. 대화형 에이전트의 효과를 증진시키기 위해 매우 중요한 기여를 하는 메모리 구조로 평가받고 있습니다.



### LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling (https://arxiv.org/abs/2606.04552)
- **What's New**: 최근 유전체학을 위한 LDARNet 모델이 발표되었습니다. 이 모델은 기존의 고정된 토큰화 방식 대신 H-Net 스타일의 동적 청킹(dynamic chunking)을 적용하여 더 적응력 있는 토큰 경계를 생성합니다. 120M개의 파라미터를 가진 LDARNet은 다양한 유전체 작업에서 뛰어난 성능을 보여주며, 일부 작업에서는 최대 20배 큰 모델보다도 우수한 성능을 발휘합니다.

- **Technical Details**: LDARNet은 BiMamba-2 상태공간 계층(state-space layers)과 지역적 주의(local attention), 양방향 라우팅(bidirectional routing), 비율 기반 정규화(ratio-based regularizer)를 결합하여 지도 없이 적응형 토큰 경계를 유도합니다. 이 모델은 27개의 다양한 유전체 작업에서 fine-tuning을 통해 11/18의 성과를 거두었으며, 고정 그리드 경계보다 learned boundary가 최대 14.3% 더 나은 성능을 발휘하는 것으로 나타났습니다.

- **Performance Highlights**: LDARNet은 Nucleotide Transformer 및 Genomic Benchmarks에서 공공 벤치마크 작업을 기반으로 평가되었습니다. 5개의 히스톤 변형 작업에서 현재 최고의 성능을 기록하였으며, 2.5B 파라미터 경쟁 모델과 비교할 때 유의미한 우위를 점하고 있습니다. 이 연구는 학습된 라우팅이 성과의 핵심 요인이라는 점을 명확히 밝혀내어, 생물학적 구조와 일치하는 경계 학습의 중요성을 강조합니다.



### Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models (https://arxiv.org/abs/2606.04535)
Comments:
          Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)

- **What's New**: 이번 논문에서는 Dynamic Infilling Anchors (DIA)라는 혁신적인 방법을 제안합니다. 이 방법은 생성 길이를 조정하는데 도움을 주며, 고정된 앵커(anchor) 사용의 비효율성을 극복합니다. DIA는 훈련 없이 동적으로 앵커 포지션을 추정하여 형식 제약이 있는 작업에서의 품질과 신뢰성을 높입니다.

- **Technical Details**: DIA는 두 단계로 구성됩니다. 첫 번째 단계에서는 끝 앵커의 포지션을 추정하여 생성 길이를 조정합니다. 두 번째 단계에서는 고정된 앵커 간의 콘텐츠를 반복적으로 생성하는 과정을 통해 구조적 일관성을 확보합니다. 이 접근법은 모델이 충분한 생성 공간을 확보하도록 하여 중복 방지 및 불필요한 컴퓨팅을 최소화합니다.

- **Performance Highlights**: 실험 결과, DIA는 GSM8K 및 MATH 데이터셋에서 0-샷(0-shot) 접근 방식으로 형식 정확도를 각각 58.83%에서 72.63%, 29.10%에서 76.82%로 크게 향상시켰습니다. 답변 정답률도 GSM8K에서 14.86%에서 46.78%로 개선되었습니다. 이러한 성과로 DIA는 신뢰성 있고 구조적으로 인식할 수 있는 생성으로 나아가는 확고한 경로가 됩니다.



### GENEB: Why Genomic Models Are Hard to Compar (https://arxiv.org/abs/2606.04525)
- **What's New**: 이번 논문은 GENEB라는 대규모 진단 벤치마크를 소개하며, 40개의 유전체 기반 모델을 100개의 작업에서 평가합니다. 이러한 평가 작업은 13개의 기능 범주로 나뉘어 있으며, 통합된 probing 기반 프로토콜에 따라 수행됩니다. 이는 모델의 스케일, 아키텍처, tokenization 및 사전 훈련 데이터 간의 상대적 비교를 가능하게 하여 기존 평가 관행의 한계를 보완하는 역할을 합니다.

- **Technical Details**: GENEB는 frozen representations를 평가하기 위해 embedding-based probing 프로토콜을 사용하여 다양한 유전체 예측 작업을 커버합니다. 각 작업에 대해 frozen embeddings를 로지스틱 회귀 모델의 특징으로 사용하며, 1-shot, 10-shot 및 전체 데이터 레짐에서 평가됩니다. 평가 메트릭으로는 클래스 불균형에 강한 Matthews Correlation Coefficient (MCC)를 사용하고, 10^5 시퀀스를 초과하는 작업은 서브샘플링하여 분석합니다.

- **Performance Highlights**: 모델의 크기와 성능 간에는 통계적으로 유의미한 상관관계가 있으며, 이는 모델 규모가 증가함에 따라 성능이 향상된다는 것을 나타냅니다. 그러나 상위 36개의 도메인 모델 가운데 31개의 경우, 5배 더 작은 모델이 더 큰 모델보다 평균 MCC에서 더 나은 성능을 기록하는 경우도 있습니다. 이는 GENEB의 평가 프레임워크가 모델 선택에 있어 단순히 파라미터 수로 환원될 수 없음을 시사합니다.



### SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inferenc (https://arxiv.org/abs/2606.04511)
- **What's New**: 이번 논문에서는 SparDA라는 새로운 희소 주의(attention) 아키텍처를 제안합니다. 이 아키텍처는 Query, Key, Value 외에 네 번째 레이어 프로젝션인 Forecast를 도입하여 다음 레이어에 필요한 KV 블록을 예측할 수 있게 합니다. 이를 통해 CPU와 GPU 간의 데이터 전송을 겹쳐 처리할 수 있어 성능 향상이 가능합니다.

- **Technical Details**: SparDA의 핵심은 Forecast가 희소 선택 스텝의 복잡도를 줄여준다는 점입니다. 기존의 선택 메커니즘에 비해 선택 오버헤드를 감소시키고, GQA(Generalized Query Attention) 구현에서 GQA 그룹당 하나의 Forecast 헤드를 사용합니다. SparDA는 전체 파라미터가 0.5% 미만으로 늘어나고, 원래의 주의 분포에 맞춰 Forecast 프로젝션만 훈련합니다.

- **Performance Highlights**: SparDA는 두 개의 희소 사전 훈련된 8B 모델에서 정확도를 유지하거나 약간 개선시키며, 최대 1.25배의 prefill 속도 향상과 1.7배의 디코드 속도 향상을 보여 줍니다. 또한, 하나의 GPU에서 더 큰 배치 크기를 허용함으로써 비오프로드 희소 기준보다 최대 5.3배 높은 디코드 처리량을 달성합니다.



### Self-Evolving Deep Research via Joint Generation and Evaluation (https://arxiv.org/abs/2606.04507)
- **What's New**: 이 연구에서는 기존의 고정된 평가자가 성능 향상에 따라 평가 기준을 동적으로 조정할 수 없는 한계를 극복하기 위해, 평가자와 해결자가 동시에 진화하는 자기 발전 코 진화(training framework for co-evolution) 방식인 SCORE를 제안합니다. 이 방식은 생성 및 평가를 독립된 모듈로 취급하는 대신, 두 가지의 내재적 연결성을 활용하여 성능을 공동으로 개선하도록 합니다. 이론적인 분석을 통해 공유 매개변수(shared-parameter) 하에서의 SCORE의 역할과 평가자의 일관성의 중요성을 규명합니다.

- **Technical Details**: SCORE는 주어진 쿼리(q)와 환경(ℰq)에 대해 후보 보고서(r)의 정책(π)을 학습하는 프레임워크로 구성됩니다. 이 방법은 쿼리에 특정한 증거와 평가 환경에 기반하여 긴 보고 내용을 생성하는 데 필요한 질적 특성을 평가합니다. 또한, 고정된 외부 메타 하네스를 통해 쿼리 특이적인 평가 환경을 형성하며, 평가자는 쿼리 기반의 루브릭을 구성하고 구조화된 보고서를 평가하는 역할을 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 SCORE는 기존의 연구 벤치마크에서 보고서 생성을 향상시키는 데 있어 일관된 성과 개선을 보여주었습니다. 우리의 방법은 평가 차원에 걸쳐 에이전트 성능을 향상시키고, 다양한 생성 작업에서 안정적인 학습 압력을 유지합니다. 코 진화 평가와 생성이 열린 연구 에이전트 훈련의 유망한 방향임을 입증하였습니다.



### SANE Schema-aware Natural-language Evaluation of Biological Data (https://arxiv.org/abs/2606.04500)
Comments:
          5 pages, 3 figures, submitted but not yet reviewed by BMT2026

- **What's New**: SANE(Schema-Aware Natural-language Evaluation)라는 새로운 패러다임이 도입되었습니다. 이 프레임워크는 약물 응답 데이터베이스에서 자동으로 생성된 평가 쿼리를 통해 텍스트를 SQL로 변환하는 데 도움을 줍니다. SANE은 특정 도메인에 맞춘 평가를 가능하게 하여 데이터베이스 접근성을 향상시킵니다.

- **Technical Details**: 고속 마이크로스코프를 사용한 약물 스크리닝 과정에서 생성되는 데이터는 계층적 실험 설계를 따릅니다. 이 연구에서는 4비트 양자화된 LlaMA 3.1 모델을 활용해 PostgreSQL 데이터베이스를 질의하며, 훈련이나 파인 튜닝 없이도 몇 가지 샷(few-shot) 모드로 SQL 쿼리를 생성합니다.

- **Performance Highlights**: 572개의 자동 생성 테스트 케이스를 통해 전체 정확도가 97.2%에 달하며, 단순한 질문에서 100%, 복잡한 질문에서 98.4%, 스키마 관련 질문에서 96.9%의 높은 정확도를 기록했습니다. 단, 문맥이 부족하거나 오류가 있는 질문에서는 성능 저하가 나타났고, 모델의 훈련 없이도 신뢰할 수 있는 데이터베이스 접근이 가능함을 입증했습니다.



### Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs (https://arxiv.org/abs/2606.04483)
Comments:
          23 pages

- **What's New**: 이 논문은 기존의 jailbreaking 기술이 특정한 프로프트(프롬프트)에 국한되어 있는 단점이 있다고 주장합니다. 연구진은 자연어 작문의 전체 레지스터(register)에서 발생하는 취약점을 발견하였고, 이를 바탕으로 팬픽션 서브장르를 활용한 새로운 jailbreak 공격 방식인 SAGA-A4를 소개합니다. 이 공격 방식은 창의적인 글쓰기 메타를 기반으로 하여 해로운 행동을 클라이맥스에서 삽입하는 방식으로, 각 모델에 대한 친화성을 유지합니다.

- **Technical Details**: 제안된 방법은 Archive of Our Own(AO3)에서 수집한 12개의 팬픽션 서브장르를 활용하여 사전 훈련된 LLM의 안전 필터를 우회하는 메타를 구성합니다. 이 메타는 특정 공격자 LLM없이 생성되며 전통적인 공격 방법보다 성공률이 3배 이상 높습니다. 연구는 HarmBench와 JailbreakBench를 통합하여 290개의 해로운 행동을 평가하고, SAGA-A4 공격 파이프라인은 평균 ASR 0.924를 기록하며 기존 다중 턴 방법보다 현저히 우수한 성과를 보였습니다.

- **Performance Highlights**: SAGA-A4는 4명의 심사위를 통해 타당성을 평가받고 평균 ASR을 0.731로 향상시키며, 두 개의 활성 방어 기법이 기존의 비율을 넓히는 결과를 보였습니다. 이 논문이 제안하는 팬픽션 레지스터의 jailsbreak 방식은 LLM의 안전성 문제를 해결하는 데 있어 새로운 방법론을 제시하며, 더욱이 기법이 모델에 대해 긍정적인 효과를 지속적으로 유지하는 것을 강조합니다.



### Entity Binding Failures in Speech LLM Reasoning: Diagnosis and Chain-of-Thought Intervention (https://arxiv.org/abs/2606.04474)
- **What's New**: 이 논문은 Speech Large Language Models (SLLMs)이 기존의 텍스트 모델에 비해 복잡한 추론(Reasoning)에서 성능이 떨어진다는 점을 지적합니다. 특히, SLLMs의 성능 격차는 단순한 인지 결함(cognitive deficit)이 아님을 드러내고 있습니다. 세 가지 다양한 SLLMs를 평가한 결과, speech-to-text (S2T)가 특정 작업에서는 text-to-text (T2T)를 초과하는 성능을 보였습니다.

- **Technical Details**: 논문에서는 S2T가 spatial, syntactic, factual 작업에서 T2T와 유사하거나 더 나은 성능을 보이지만, 논리적 작업에서는 성능이 무작위(random) 수준으로 떨어진다고 언급합니다. 이는 엔터티 바인딩 실패(entity binding failure)로 진단되며, 주의 깊은 단어 간의 연관성이 유지되지 않아서 발생합니다. 이를 해결하기 위해, 저자들은 Entity-Aware Chain-of-Thought (EA-CoT) 접근 방식을 제안하며, 이는 모델이 추론 전에 엔터티를 명시적으로 열거하고 주장(claims)과 바인딩하도록 합니다.

- **Performance Highlights**: EA-CoT를 도입하면, 음성 인식 오류가 발생하더라도 성능 격차를 효과적으로 줄일 수 있으며, 최대 24.4%의 절대 정확도(absolute accuracy) 향상을 가져옵니다. 실험 결과는 이러한 성과가 모두 명시적 의미 바인딩(explicit semantic binding)에서 기인함을 입증하며, 이에 따라 SLLMs의 성능 격차는 해결 가능한 병목 현상으로 재구성됩니다.



### Learning What to Learn: Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning (https://arxiv.org/abs/2606.04466)
Comments:
          25 pages, 12 figures

- **What's New**: 이 연구에서는 Reasoning을 위한 Post-training Small Language Models (SLMs)에 대한 새로운 접근법을 제안합니다. 기존의 SFT-then-RL 파이프라인에서는 데이터 학습 단계를 고려하지 않았습니다. 새로운 기법인 difficulty-aware SFT-then-RL 프레임워크는 각 단계에 맞춰 훈련 데이터를 구성하여 SLM의 Reasoning 능력을 향상합니다.

- **Technical Details**: 우리는 데이터 난이도에 따라 SFT와 RL 단계의 기능적 역할을 구분합니다. SFT는 아직 마스터되지 않은 Reasoning 기술을 습득하는 데 적합하고, RL은 모델이 부분적으로 접근할 수 있는 기술을 강화하는 데 더 효과적입니다. 새로운 Bridge 메커니즘은 어렵고 복잡한 reasoning trace를 학습하기 쉽게 변환합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 SLM에서 다섯 개의 Reasoning 벤치마크를 기반으로 실험한 결과, 기존의 SFT, distillation, RL 기준선에 비해 일관되게 더 좋은 성능을 보였습니다. 이 결과는 SFT와 RL 간 데이터 난이도 조율이 효과적인 SLM Reasoning에 중요함을 강조합니다.



### SePO: Self-Evolving Prompt Agent for System Prompt Optimization (https://arxiv.org/abs/2606.04465)
Comments:
          26 pages. Code: this https URL

- **What's New**: 이 논문에서는 Self-Evolving Prompt Optimization (SePO)이라는 새로운 방법을 제안합니다. 이는 기존의 시스템 프롬프트 최적화 방식을 넘어서, 프롬프트 에이전트의 시스템 프롬프트도 최적화 대상으로 삼습니다. 한 개의 프롬프트 에이전트가 태스크 에이전트의 시스템 프롬프트와 자신을 모두 향상시키며, 이는 오픈 엔디드(evolutionary search) 진화 검색을 통해 가능합니다.

- **Technical Details**: SePO의 구조는 자기참조(self-referential) 디자인을 따릅니다. 이 방법은 두 단계로 나뉘며, 첫 번째 단계는 여러 태스크에 대한 프리트레이닝(pre-training)이고, 두 번째 단계는 특정 태스크에 대한 파인튜닝(fine-tuning)입니다. 이를 통해 프롬프트 최적화 기술이 특정 태스크에 한정되지 않고, 다양한 태스크에 대해 일반화됩니다.

- **Performance Highlights**: SePO는 수학(AIME'25), 추상적 추론(ARC-AGI-1), 과학(GPQA), 코드 생성(MBPP), 논리 퍼즐(Sudoku) 등 다양한 벤치마크에서 검증되었습니다. Manual-CoT에 비해 평균 정확도가 4.49 포인트 개선되었으며, 모든 태스크에서 최상의 정확도를 달성했습니다. 프리트레이닝과 파인튜닝 단계로 나누는 접근은 향후 여러 태스크에 대해 반복적으로 활용될 수 있는 프롬프트 에이전트의 효율적인 개선을 가능하게 합니다.



### Stepwise Reasoning Enhancement for LLMs via External Subgraph Generation (https://arxiv.org/abs/2606.04454)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 논리적 일관성과 사실적인 근거, 해석 가능성 문제를 해결하기 위한 새로운 접근법인 SGR(Stepwise Reasoning enhancement framework)를 제안합니다. SGR은 입력 질문을 바탕으로 구조화된 스키마를 만들어 외부 지식 그래프에서 관련 서브그래프를 생성하여 LLM과 결합합니다. 이 과정은 LLM이 단계적인 추론을 하도록 지원하며, 결과적으로 정확도와 해석 가능성을 향상시킵니다.

- **Technical Details**: SGR은 먼저 입력 질문에서 핵심 엔터티, 관계 및 제약 조건을 추출하여 구조화된 스키마를 생성합니다. 이후, Cypher 쿼리를 통해 Neo4j에서 지식 그래프로부터 작은 서브그래프를 검색합니다. 이 서브그래프는 LLM의 추론을 뒷받침하는 명시적인 관계 증거를 제공하며, 위치에 따라 검증된 반복 가능한 경로를 따라 단계별로 추론할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, SGR은 CWQ, WebQSP, GrailQA 및 KQA Pro와 같은 벤치마크 데이터 세트에서 표준 프롬프트 방법 및 여러 지식 강화 기준선보다 Hits@1 성능과 정확도가 향상된 것으로 나타났습니다. 추가적인 연구에서는 스키마 안내와 Neo4j 기반 검색이 이 프레임워크의 효과에 모두 중요하다는 것을 확인하였습니다. 이러한 결과는 동적으로 생성된 외부 서브그래프가 LLM 기반 추론의 정확도, 내구성, 그리고 해석 가능성을 향상시킬 수 있음을 시사합니다.



### Listening to the Workforce: Measuring Construction Worker Safety Attitudes from Social Media Discourse Using LLMs (https://arxiv.org/abs/2606.04450)
- **What's New**: 이번 연구에서는 건설 현장에서의 안전 태도를 측정하기 위한 새로운 구조인 Construction Safety Attitude Framework (CSAF)를 개발하고 검증하였습니다. CSAF는 여덟 개의 차원으로 안전 태도를 설명하는 이론적인 구조와 이를 작업자 자연 담화에서 측정하기 위한 운영적 코드북을 통합한 것입니다. 이 프레임워크는 대규모 말뭉치에 적용할 수 있도록 대규모 언어 모델(LLM)을 통해 실제적으로 운영됩니다.

- **Technical Details**: CSAF는 250개의 Reddit r/Construction 게시물과 댓글에서 훈련된 코더들이 높은 일치를 보인 결과(크리핀도르프의 알파 = 0.85)를 기반으로 하여 구축되었습니다. CSAF는 450개의 r/Construction 기여에 적용된 후 전문적 인간 코딩과 재현성을 보였으며, Cohen의 카파 값 0.90, 정밀도 0.98, 재현율 0.98을 기록했습니다. 이것은 다른 무역 커뮤니티로의 전이 후에도 비슷한 정확도를 유지했습니다.

- **Performance Highlights**: 이번 연구의 경우 연구자는 10,346개의 r/Roofing 기여를 사용하여 CSAF가 안전 주제별로 다차원적 태도를 구분하고 시간에 따른 변화 추적을 가능하게 함을 입증하였습니다. CSAF는 안전 관행을 결정짓는 태도를 분석하고 이를 기반으로 위험한 관행을 해결할 수 있는 목표 간섭을 위한 이론적으로 기반이 있는 도구로 작용합니다. 이를 통해 정부 기관이나 산업 협회가 정책 및 교육 우선 순위를 inform하는 데 사용할 수 있는 기반이 마련되었습니다.



### MemoryDocDataSet: A Benchmark for Joint Conversational Memory and Long Document Reasoning (https://arxiv.org/abs/2606.04442)
Comments:
          17 pages, 2 figures, 8 tables. Submitted for peer review

- **What's New**: MemoryDocDataSet는 AI 시스템의 두 가지 중요한 기능, 즉 긴 대화 기록 탐색과 긴 문서에 대한 깊은 독해 능력을 동시에 평가할 수 있는 새로운 벤치마크 데이터셋입니다. 이 데이터셋은 50개의 마이크로 월드(micro-worlds)와 1,000개의 QA 쌍으로 구성되어 있으며, 각 인스턴스는 3-5명의 페르소나(personas), 다수의 시간적 사건 그래프(temporal event graph), 그리고 20,000-50,000 토큰으로 구성된 긴 문서를 포함합니다. 특히, 기계가 대화 기록을 탐색하여 관련 문서를 찾고 그 문서에서 답변을 추출해야 하는 'Hybrid' 질문이 75.1%를 차지하는 것이 특징입니다.

- **Technical Details**: MemoryDocDataSet은 서로 다른 6개의 기본 구성(베이스라인 설정)을 평가하며, 이들은 잘라낸 컨텍스트, 긴 컨텍스트 LLM(long-context LLM), 회수 증강 생성(retrieval-augmented generation, RAG), 메모리 시스템을 포함합니다. 데이터셋의 품질은 LLM을 사용한 자기 일관성 분석을 통해 측정되며, 50개의 마이크로 월드에서 중위값 Cohen's κ는 0.634입니다. 본 연구의 목표는 메모리 기반 시스템과 긴 문서 내비게이션을 통합하는 새로운 시스템 구조를 동기화하는 것입니다.

- **Performance Highlights**: Baseline 설정에서 RAG-Both 모델이 0.358의 전체 F1 스코어를 달성하였으나, Hybrid 질문에서는 0.342에 불과하여 여전히 인간 성능에 비해 낮은 성과를 보였습니다. Document-only retrieval 방식인 RAG-Doc은 오히려 Hybrid 질문에서 0.267로 저조한 성과를 기록하며 이러한 조합 접근법에서의 명확한 성능 격차를 강조합니다. 이러한 격차는 대화 메모리와 긴 문서 탐색을 통합하는 아키텍처에 대한 필요성을 더욱 부각시킵니다.



### Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion Language Models (https://arxiv.org/abs/2606.04396)
Comments:
          19 pages, 10 figures, 7 Tables

- **What's New**: 이번 논문에서는 차세대 언어 모델인 dLLMs(Diffusion Large Language Models)의 응답 생성 방식을 개선하기 위한 새로운 방법인 CAPR(Cached-Amortized Path Refinement)를 소개합니다. CAPR은 기존의 강화 학습 기술에서 약하게 사용되었던 denoising trace를 활용하여 트리 수준의 계산 없이도 트리와 같은 감독(supervision)을 제공할 수 있는지에 대해 질문합니다.

- **Technical Details**: CAPR 알고리즘은 denoising trace를 간결한 경로 상태로 요약하고, 캐시된 경로 상태를 이용하여 저렴한 형제( sibling) 연속체를 생성합니다. 또한, 블록 수준 감독을 위한 가치 헤드(value head)를 훈련시킵니다. 블록 단위의 언마스킹 일정에 따라 CAPR은 최종 보상을 각 블록에서 드러난 토큰에 따라 재분배하는 방식으로 동작합니다.

- **Performance Highlights**: CAPR은 기존의 flat rollout보다 약 0.75배, tree rollout보다 약 0.6배 낮은 비용으로 롤아웃 생성 비용을 줄입니다. 4x4 스도쿠, Countdown, GSM8K, Math500과 같은 다양한 테스트에서 CAPR은 256 및 512 토큰 예산으로 RL 튜닝된 dLLMs의 새로운 최첨단 성능을 보여주며, 스도쿠 문제에서는 가장 강력한 트리 구조 베이스라인에 비해 1/3에 불과한 계산 비용으로 성능을 발휘합니다.



### When Clients Stop Following: A Cognitive Conceptualization Diagram-driven Framework for Strategic Counseling (https://arxiv.org/abs/2606.04389)
- **What's New**: 이번 논문에서는 심리 상담에 대한 기존의 평가가 협조적인 고객 시뮬레이터에 과도하게 의존하고 있음을 지적하며, 상담 모델의 평가 불일치를 해결하기 위한 인지 행동 치료(CBT) 기반의 저항 인식 프레임워크를 제안합니다. 저자는 CARS라고 불리는 클라이언트 시뮬레이터를 소개하는데, 이는 클라이언트의 역동적인 저항을 모델링하는 Cognitive Conceptualization Diagrams (CCDs)를 사용합니다. 또한, 전략적 사고를 분리하고 강화 학습을 통해 최적화하는 STREAMS라는 이중 모듈 프레임워크를 소개합니다.

- **Technical Details**: 이 연구의 프레임워크는 CARS와 STREAMS 두 개의 상호 작용 에이전트로 구성됩니다. CARS는 클라이언트 저항을 비이상적으로 발달시키고, 클라이언트의 신념 시스템을 표현하는 CCD를 사용합니다. STREAMS는 전략적 사고(Thinker)와 응답 생성(Presenter)을 분리하여 저항이 있는 상호 작용에서의 상담자 반응성을 평가하기 위해 EWTS-MI라는 엔트로피 가중치 메트릭으로 대화의 질을 평가합니다.

- **Performance Highlights**: 실험을 통해 저항과 비저항 상담 환경에서의 평가 불일치에 대한 발견을 검증하고, 저항 인식 훈련이 도전적인 상담 상호작용에서 전략적 강인성을 개선하는 데 효과적임을 입증하였습니다. 이는 실제 상담에서의 저항적 상호작용 및 상담 모델의 강인성을 높이기 위한 중요한 기초자료를 제공합니다.



### DLLG: Dynamic Logit-Level Gating of LLM Experts (https://arxiv.org/abs/2606.04378)
- **What's New**: 새로 제안된 DLLG(Dynamic Logit-Level Gating) 프레임워크는 여러 전문화된 LLMs를 효과적으로 통합할 수 있는 동적 logit-level ensembling 방식입니다. 기존의 라우팅 방법이 조기 결정으로 유연성을 희생하는 문제를 해결하기 위해, DLLG는 토큰 수준 전문가 융합을 학습하여 적은 양의 응답 수준 감독을 활용합니다. 이 접근법은 전문가 파라미터를 고정시켜 간섭을 피하고, 재훈련을 요구하지 않으며 효율적인 전문가 통합을 가능하게 합니다.

- **Technical Details**: DLLG는 자율 회귀 소프트 융합(autoregressive soft fusion)을 수행하며, 각 디코딩 단계에서 경량 게이트가 프롬프트와 부분 접두사 및 모든 전문가의 진행 상태(hidden states)를 바탕으로 단계별 혼합 가중치를 생성합니다. 또한, 응답의 정확도 레이블(teacher forcing)을 이용하여 훈련된 게이트는 응답 수준의 정확성 레이블을 모든 토큰에 전송하여 융합 규칙을 배웁니다. 이 방법은 토큰 수준의 라벨링 없이도 전문가 활용을 세밀하게 조정할 수 있게 합니다.

- **Performance Highlights**: DLLG는 다양한 추론 및 코드 벤치마크(GSM8K, Minerva Math, MATH 등)에서 우수한 성능을 보였습니다. 기존의 라우팅, 휴리스틱 앙상블 및 파라미터 병합 기법을 초과하는 성능을 보여주며, 모델 규모에 상관없이 일관된 향상을 기록했습니다. 이러한 결과는 DLLG가 학습된 logit-level 융합이 전문화된 전문가들을 통합하는 데 있어 견고하고 확장 가능한 패러다임이 될 수 있음을 보여줍니다.



### GlossAssist -- A Tool to Simplify Corpus Creation and Study the Effect of NLP Models in Low-Resource Documentation Settings (https://arxiv.org/abs/2606.04367)
Comments:
          6 pages, 3 figures

- **What's New**: 이번 논문에서는 언어 문서화를 위한 새로운 자동 주석 도구인 GlossAssist를 소개합니다. 이 도구는 CWoMP의 검색 기반 아키텍처를 사용하여 언어 분석가의 수정 입력을 실시간으로 반영하고, 예측 결과를 학습된 형태소 표현의 변형 가능한 어휘에 기반하여 제공합니다. 문제는 기존의 도구들이 평가를 위한 설계로 인해 실용적인 사용이 어렵다는 점입니다.

- **Technical Details**: GlossAssist는 두 가지 통합 구성 요소로 이루어져 있습니다: (1) 문장 단위의 IGT 생산을 위한 주석 인터페이스와 (2) 모델 성능을 분석하기 위한 연구자 대시보드입니다. 이 도구는 기존의 NLP 모델을 활동적인 문서화 워크플로우에 통합하고, 모델의 효율성을 평가하며, 새로운 코퍼스를 구축하는 세 가지 사용 모드를 지원합니다.

- **Performance Highlights**: 각 주석 세션에서는 수용된 예측이 검증된 어휘 항목으로 추가되고, 거부된 예측은 모델 실패의 신호로 간주됩니다. 대시보드는 주석의 패턴과 모델 개선이 필요한 지점을 연구자가 이해하는데 도움을 주는 KPI를 제공합니다. GlossAssist는 이러한 구조를 명확하고 실행 가능하게 만들어, 현장에서 작업하는 언어 분석가와 코퍼스를 평가하는 연구자 모두에게 유용합니다.



### Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs (https://arxiv.org/abs/2606.04360)
Comments:
          ICML 2026

- **What's New**: 이 논문에서는 기호 회귀(symbolic regression) 분야에서 기존의 LLM(대형 언어 모델) 기반 진화 방법의 샘플 비효율성을 해결하기 위해 Deliberate Evolution (DE)라는 새로운 프레임워크를 제안합니다. DE는 후보 제안(candidate proposal)과 탐색 안내(search guidance)를 분리하여 LLM이 표현을 발전시키는 방법을 추론할 필요가 없도록 설계되었습니다.

- **Technical Details**: DE는 적응형 연산자(adaptive operators)를 사용하여 탐색 방향을 안내하고, 구조 진단을 위한 분석 도구(analytical tools), 경로 수준 경험(reflective memory)을 제공하여 기호 생성을 탐색 제어로부터 분리합니다. 이 접근 방식은 LLM이 단일 점수로부터 과거 경험을 재사용하는 데서 발생하는 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, LLM-SRBench에서 DE는 다양한 과학 분야의 대표적인 LLM 기반 기호 회귀 기준선 대비 일관되게 더 나은 성능을 보이며, 표준 샘플 예산의 40%만을 사용했습니다. 이는 DE가 기호 회귀의 효율성을 높이는 데 있어 중요한 기여를 한다는 것을 보여줍니다.



### Noisy memory encoding explains negative polarity illusions (https://arxiv.org/abs/2606.04340)
Comments:
          21 pages, 5 figures, submitted for journal publication

- **What's New**: 이번 연구에서는 부정적 양극성 착각(negative polarity illusion)이라는 현상을 조명하고, Hahn et al. (2022)의 손실 컨텍스트 서프리살 이론(lossy context surprisal theory)을 기반으로 이 현상이 어떻게 발생하는지를 제안합니다. 연구자들은 사람들이 주요 절(main clause)과 포함 절(embedded clause)에서 결정자(determiner)에 대한 기억 표현이 좋지 않다는 가설을 세우고, 유사한 결정자가 있을 경우 착각 효과가 더 강해질 것이라고 주장합니다.

- **Technical Details**: 연구는 특히 'ever'와 같은 부정적 양극성 표현이 특정 언어적 환경에서만 발생한다는 점에 주목하며, 'no'가 결정자의 허가자(licensor) 역할을 할 때 'ever'가 문법적으로 허가된다는 점을 설명합니다. 대학생들을 대상으로 한 직접적인 수용성 실험에서, 송신자가 공급하는 결정자들 간에 유사성이 클수록 결정자들의 기기억을 서로 바꾸어 이끌 수 있다는 점을 실험적으로 증명했습니다.

- **Performance Highlights**: 수용성 태스크에 대한 결과는 유사한 결정자 쌍이 있는 문장이 기존의 문장보다 더 강한 착각 효과를 발생시킨다는 것을 보여줍니다. 예를 들어, 'Many authors that few critics recommended have ever received acknowledgment for a best-selling novel.'과 같은 형태는 전통적인 문장보다 수용성에서 더욱 강한 착각을 유도함을 입증했습니다. 이는 인간의 언어 처리 과정이 불완전하다는 이론적 근거를 제공하며, 적은 작업 기억(resource-rational)을 활용해 손상된 언어 입력을 바탕으로 합리적으로 복원하는 경향을 나타냅니다.



### Parameter-Efficient Fine-Tuning with Learnable Rank (https://arxiv.org/abs/2606.04325)
Comments:
          In Submission

- **What's New**: 이 연구에서는 사용자가 지정한 고정된 랭크가 매개변수 효율적인 미세 조정을 위한 최적의 귀납적 편향인지에 대한 의문을 제기합니다. 이를 해결하기 위해 'Learnable Rank LoRA (LR-LoRA)'라는 새로운 PEFT 방법을 제안합니다. LR-LoRA는 학습 과정 중에 어댑터 랭크( rank)를 학습하게 하여 각 레이어에 적합한 랭크를 최적화할 수 있게 합니다.

- **Technical Details**: LR-LoRA는 저차원 서브스페이스에서 최적화를 실행하여 고정된 저차 랭크 바이어스를 도입하는 기존 LoRA 접근법을 개선합니다. LR-LoRA는 각 레이어의 어댑터 업데이트에 비선형성( nonlinearity)을 적용하여, 고정된 랭크 제약을 이완하며, 최적화기에 각 레이어의 업데이트 차원을 조정할 수 있도록 합니다. 이를 통해 변환기 모델의 주의(attention) 및 다층 퍼셉트론(MLP) 레이어에서 학습된 랭크의 중요한 변화가 관찰되었습니다.

- **Performance Highlights**: LR-LoRA는 언어 이해 및 상식 추론 벤치마크에서 우수한 성능을 발휘하며, 강력한 PEFT 기준을 일관되게 초월합니다. 이 방법은 125M에서 13B 파라미터에 이르는 77개의 아키텍처와 1919개의 작업에서 새로운 최첨단 결과를 달성하였습니다. 이러한 결과는 고정된 랭크 제약이 매개변수 효율적인 미세 조정을 위한 최적의 접근법이 아님을 시사합니다.



### LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding (https://arxiv.org/abs/2606.04302)
Comments:
          ICML 2026

- **What's New**: 본 연구에서는 LazyAttention 이라는 새로운 주의(attention) 메커니즘을 소개합니다. 이는 기존의 Key-value (KV) 캐싱 방식의 한계를 극복하며, 임의의 위치에서 제로 복사(zero-copy) 및 위치 불가지(bin-agnostic) KV 재사용을 가능하게 합니다. LazyAttention은 주의 커널(attention kernels) 내에서 positional encoding을 동적으로 조정하여 메모리 물질화(materialization) 병목 현상을 해결합니다.

- **Technical Details**: LazyAttention은 deferred positional encoding을 커널화(kernelize)하여 캐시 내에서 정보의 재사용을 극대화합니다. 이는 기존의 KV 캐싱 방식이 위치 정보를 캐시에 직접 포함시키는 문제를 해결합니다. 제안된 시스템은 특정 문서 배포에 따라 조정된 attention kernels를 사용하여 KV 복사가 여러 논리적 요청에 동시에 사용될 수 있도록 합니다.

- **Performance Highlights**: LazyAttention은 기존의 Block-Attention과 비교하여 효율성을 크게 개선하였습니다. skewed document distributions 하에서 시간-첫 토큰(time-to-first-token, TTFT)을 1.37배 감소시키고, 추론(throughtput) 성능을 1.40배 증가시켰습니다. 이러한 성능 개선에도 불구하고, 출력 품질(output quality)은 유사한 수준을 유지합니다.



### Using Text-Based Causal Inference to Disentangle Factors Influencing Online Review Ratings (https://arxiv.org/abs/2606.04286)
Comments:
          HLT/NAACL 2025

- **What's New**: 이 논문은 온라인 리뷰에서 추출된 다양한 요소들이 전체 평가에 미치는 영향을 이해하기 위한 새로운 방법론을 소개합니다. CausalBERT를 기반으로 한 이 접근법은 각 요소의 영향을 분리하고, 이를 보다 정확하게 평가하기 위해 세 가지 주요 개선사항을 포함합니다. 이러한 방법론은 600,000개의 미국 K-12 학교 리뷰에 대한 실증적 검증을 통해 유효성을 입증하였습니다.

- **Technical Details**: 연구에서는 Neyman의 잠재적 결과 프레임워크를 적용하여 각 학교의 리뷰에서 텍스트 변수(X)와 평균 리뷰 점수(Y)가 주어졌을 때, 특정 주제의 존재 여부(T)에 따른 영향을 분석합니다. CausalBERT는 BERT의 확장을 통해 텍스트 표현을 학습하고, 이를 기반으로 경향성 점수와 조건부 기대 출력을 예측합니다. 혼란 변수 조정을 위해 CausalBERT에서 온도 스케일링과 하이퍼파라미터 최적화 같은 방법이 도입되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론을 통해 학교 행정 문제와 학업 성취도가 학교에 대한 전반적인 인식에 중요한 영향을 미친다는 것을 발견했습니다. 이 연구는 교육 분야의 리뷰 분석에 중요한 의미를 가지며, 실제 교육 품질과 관련된 요소들이 어떻게 평가에 반영되는지를 명확히 밝혀냈습니다. 궁극적으로, 이 연구는 기업들이 개선해야 할 주요 영역을 식별하는 데 도움을 주는 통찰력을 제공합니다.



### Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Redd (https://arxiv.org/abs/2606.04274)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 잘못된 정보의 분류를 수행하는 능력을 지속적으로 검토하고 있다. 우리가 수행한 실험은 900개의 Reddit 댓글을 기반으로 하여, 잘못된 정보에 대한 반응을 belief(믿음), fact-check(사실 확인), 또는 other(기타)로 분류하는 능력을 테스트했다. 이 연구는 현재 상용화된 LLM 모델들, 특히 Claude Haiku 4.5와 DelBERT에 대해 Zero-shot 성능을 평가하며, 전체적인 성능과 비용 측면에서도 주목할 만한 결과를 보여준다.

- **Technical Details**: 이 연구에서는 BART-MNLI, 세 가지 Llama 변형 및 여러 상업적 LLM 모델들을 비교하면서, 각 모델의 성능을 검증하기 위해 Stratified 5-fold cross-validation와 permutation tests를 사용하였다. 각 모델의 성능은 보편적 라벨과 주제별 라벨 두 가지 조건 아래에서 평가되었다. 특히, fine-tuned RoBERTa는 0.62의 macro-$F_1$ 점수를 기록하여 다른 zero-shot 모델들보다 월등한 성과를 보였고, belief 감지에 있어서는 특히 높은 성능을 보였다.

- **Performance Highlights**: 결과적으로, 기존의 zero-shot 모델들에서는 belief 관련 콘텐츠를 잘 감지하지 못하는 경향이 발견되었다. Llama-3-8B가 Llama-3-70B와 동등한 성능을 보이는 반면, Claude Sonnet 4.6는 Claude Haiku 4.5에 비해 저조한 성능을 발휘하였다. 특히, fine-tuned RoBERTa는 다른 zero-shot 모델들보다 높은 정확도를 기록하며, large generative models의 빠른 성장이 무색할 정도로 task-specific fine-tuning이 여전히 유효한 전략임을 보여주었다.



### Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA (https://arxiv.org/abs/2606.04262)
Comments:
          16 pages, 7 figures

- **What's New**: 이 연구에서는 DOSEBENCH라는 새로운 벤치마크를 도입하여 OTC 약물인 아세트아미노펜과 이부프로펜에 대한 81개 시나리오를 평가합니다. 이 벤치마크는 사용자가 안전하게 또 다른 복용을 할 수 있는지에 대한 건강 질문을 다루며, 최근 LLM(대형 언어 모델)의 성능을 평가할 수 있도록 설계되었습니다. 기존의 의료 질문-응답 벤치마크는 임상 지식이나 일반적인 의료 추론에 중점을 두었지만, DOSEBENCH는 OTC 복용 결정을 중심으로 하여 보다 안전 관련한 문제에 초점을 맞춥니다.

- **Technical Details**: DOSEBENCH는 아세트아미노펜과 이부프로펜 사용에 관한 실제 성인 소비자 질문을 기반으로 한 OTC 복용 시나리오로 구성되어 있습니다. 연구에서는 네 가지 LLM을 평가하였고, 의사 결정의 정확성, 일관성, 설명 가능성, 실패 패턴 및 신뢰도 관련 행동을 측정했습니다. 평가 결과, 모델이 종종 시간적 추론(temporal reasoning) 및 애매모호한 경우(ambiguity-sensitive cases)에서 어려움을 겪으며, 안정적이거나 신뢰감 있는 답변이 복용 제한을 위반할 수 있음을 발견했습니다.

- **Performance Highlights**: DOSEBENCH의 결과는 현재의 LLM이 소비자 지향적인 OTC 복용 결정을 내리는 데 있어 신뢰감을 주지만, 여전히 문제를 야기할 수 있음을 보여줍니다. 안전 관련 복용 규제를 준수하지 않는 잘못된 사례가 많아, 이 벤치마크는 LLM의 안전 지향적 신뢰성(safety-oriented reliability)을 평가하는 데 유용한 테스트베드(testbed)를 제공함을 시사합니다. 이러한 발견은 OTC 복용 QA가 시간적 추론, 제약 준수 및 안전 관련 불확실성 처리와 같은 중요한 요소를 평가하는 데 효과적임을 보여줍니다.



### Supportive Token Revealing for Fast Diffusion Language Model Decoding (https://arxiv.org/abs/2606.04236)
- **What's New**: 본 논문에서는 AXON이라는 새로운 모듈을 제안합니다. AXON은 기존의 parallel decoding 전략에 추가하여 사용할 수 있는 training-free 모듈로, 마스크된 토큰에 대해 신뢰도 및 의존성을 기반으로 정보가 풍부한 토큰을 선택하여 추가적인 문맥을 제공합니다. 이를 통해 디퓨전 언어 모델의 품질-지연(time latency) 트레이드오프를 개선할 수 있습니다.

- **Technical Details**: AXON은 마스크된 토큰들을 모니터링하고 필요할 때만 개입하는 방식으로 동작합니다. 이 모듈은 유용한 문맥을 제공할 수 있는 확신 있는 마스크된 토큰을 선택하기 위해 주의(attention), 불확실성(uncertainty), 신뢰도(confidence) 신호를 사용하여 후보 토큰에 점수를 매깁니다. AXON의 선택 과정은 서브모듈러(submodular) 함수 목표를 따릅니다.

- **Performance Highlights**: 다수의 디퓨전 언어 모델과 패러럴 디코더를 대상으로 한 실험에서 AXON이 품질-지연 트레이드오프를 개선하는 데 성공했음을 보여주었습니다. AXON은 기능 평가 횟수를 줄이면서 정확도를 유지하거나 개선하여 기존의 강력한 기준선보다 나은 성능을 발휘합니다.



### MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A (https://arxiv.org/abs/2606.04231)
Comments:
          Accepted at ACL 2026 (Industry Track)

- **What's New**: 본 논문은 MM-RAG(다중 모달 검색 보강 생성)의 최근 발전에 대한 논의에서 출발한다. 기존 방법들이 복잡한 기업 문서의 구조적 정보를 명시적으로 처리하지 않고, 사전 훈련된 임베딩 모델에 의존한 데 반해, 본 연구에서는 문서 구조 인식 스플릿을 통해 문서 구조를 능동적으로 추출하고 표현하는 방식을 제안한다. 이를 통해 MM-BizRAG는 다양한 문서 유형에 대해 최적화된 파이프라인을 구현하며, 수직 구조와 수평 구조에 각각 적합한 처리를 통해 효율성을 높인다.

- **Technical Details**: MM-BizRAG는 문서 처리에 있어 독특한 접근 방식을 채택한다. 각 문서는 LLM 기반 분류기를 통해 수직 구조(V)와 수평 구조(H)로 구분된 후, 레이아웃 인식 파싱(layout-aware parsing)이 적용된다. 이 과정은 페이지 내 텍스트 블록, 테이블 및 그림을 정렬된 형태로 추출하여 자연스러운 독서 순서를 유지하는 동시에 재구성된 표현으로 변환한다.

- **Performance Highlights**: MM-BizRAG는 대규모의 이질적인 기업 데이터셋과 SlideVQA, FinRAGBench-V 같은 공공 벤치마크에서 기존 비전 중심 기법들보다 최대 32% 더 높은 성능을 보여준다. 연구진은 FastRAGEval이라는 새로운 메트릭을 도입하여 생성적 회수의 정밀도를 높이면서도 비용을 절반으로 줄이는 효과를 얻었다. 이러한 결과는 재활용하지 않고도 풍부한 맥락 기반 생성을 가능하게 하여 다중 모달 생성을 한층 강화한다.



### Cross-Prompt Generalization in Detecting AI-Generated Fake News Using Interpretable Linguistic Features (https://arxiv.org/abs/2606.04199)
- **What's New**: 이번 연구에서는 다양한 프롬프트(prompt) 전략 하에서 AI 생성 가짜 뉴스(fake news) 탐지 모델의 강 robustness을 평가하기 위한 새로운 접근 방식을 채택하였습니다. 세 가지 데이터셋을 사용하여 각기 다른 프롬프트에서 생성된 AI 기사와 실제 뉴스 기사를 결합하여 가짜 뉴스 탐지의 일반화 가능성(cross-prompt generalization)을 탐구하였습니다. 연구 결과는 단일 프롬프트에서 훈련된 모델이 다른 프롬프트에서 테스트할 때에도 우수한 성능을 유지함을 보여줍니다.

- **Technical Details**: 이 연구는 각기 다른 스타일과 언어적 특성이 부여된 세 가지 프롬프트를 사용하여 AI 생성 가짜 뉴스를 평가하기 위해 구성된 데이터셋을 활용합니다. 이러한 데이터셋은 감정적 어조, 서사 일관성 및 구조적 프레임을 보존하면서도 각기 다른 생성 방식을 반영하여, 가짜 뉴스 탐지의 프롬프트에 따른 변동성을 분석합니다. 이 연구에서는 랜덤 포레스트(random forest) 분류기를 활용하여 설명 가능한 언어적 특징(linguistic features)을 추출하고, 이를 통해 AI 생성 텍스트의 통계적 특성(statistical properties)을 분석합니다.

- **Performance Highlights**: 모든 훈련-테스트 조합에서 AUC(Area Under Curve) 값은 0.988에서 1.000으로 일관되게 높은 성능을 기록하였습니다. AI 생성 텍스트는 더 높은 어휘 다양성(lexical diversity), 낮은 가독성(readability), 그리고 상대적으로 낮은 감정 강도를 보입니다. 프롬프트에 따른 특징 분포 변화를 분석한 결과, 이러한 특징들이 서로 다른 프롬프트에 걸쳐 안정적인 특성을 포착하고 있음을 보여주며, 이는 실제 환경에서의 신뢰할 수 있는 AI 생성 가짜 뉴스 탐지 방법 개발에 기여할 것으로 기대됩니다.



### ACAT: A Collaborative Platform for Efficient Aspect-Based Sentiment Dataset Annotation (https://arxiv.org/abs/2606.04189)
Comments:
          Accepted at The 28th International Conference on Big Data Analytics and Knowledge Discovery (DaWak 2026)

- **What's New**: ACAT(Aspect-based sentiment analysis Collaborative Annotation Tool)는 고품질의 ABSA 데이터 세트를 지원하는 웹 기반 플랫폼으로, 기존의 주석 도구가 처리하던 단점들을 해결합니다. 네 가지 ABSA 워크플로(Aspect-Category Sentiment Analysis, Clause-Level Segmentation 등)를 원활하게 지원하며, 자동화된 Extract, Transform, Load (ETL) 파이프라인을 통해 데이터 협업과 IAA(Inter-Annotator Agreement) 측정을 자동으로 수행할 수 있습니다.

- **Technical Details**: ACAT은 Docker 컨테이너(E.g., PostgreSQL, Python Flask)로 배포되어 빠른 설치가 가능합니다. 모든 주석은 각기 다른 레벨(Aspect Category, Sentiment Polarity 등)로 구성되며, CSV, JSON 및 XML 포맷으로 내보낼 수 있는 구조를 갖추고 있습니다. 또한, Implicit Toggle기능을 통해 명시적 용어가 없을 경우에도 숨겨진 의미를 포착할 수 있도록 설계되었습니다.

- **Performance Highlights**: ACAT의 사전 검증 결과 1,002개의 레스토랑 리뷰를 대상으로 각각의 전문성이 다른 두 개의 주석자가 참여하여, 중간 주석 시간 31.58초, IAA 지수 0.78에서 0.86까지 다양하게 기록되었습니다. 직접적인 플랫폼 성능 비교가 난해하지만, ACAT는 데이터 엔지니어링 비용을 줄이면서 목록원 비율이 높은 주석 효율성을 보여줍니다.



### A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models (https://arxiv.org/abs/2606.04177)
Comments:
          preprint

- **What's New**: 이번 연구는 AI 생성 텍스트를 설명하기 위한 해석 가능한 언어적 특성(interpretable linguistic features)의 신뢰성을 평가하는 대규모 실증 연구입니다. 27개의 LLM과 10개의 텍스트 도메인에서 284개의 언어적 특성을 분석하여, 비전문가 사용자도 이해할 수 있도록 기계 생성 텍스트를 구분할 수 있는 신뢰할 수 있는 신호를 제시합니다.

- **Technical Details**: 연구에서는 다양한 모델(27 LLM)과 도메인(10 text domains)에서 생성된 텍스트의 언어적 특성을 284개에 걸쳐 분석하였으며, 이를 통해 범모델(cross-model) 및 범도메인(cross-domain) 일반화(settings) 하에서 신호의 강력함을 평가했습니다. 분석을 통해, 언어적 특성만으로도 AI 생성 텍스트와 인간이 작성한 텍스트를 구분할 수 있는 기준(classifiers)을 확립하였습니다.

- **Performance Highlights**: 결과적으로, 많은 기존의 지표(indicators)는 특정 문맥(context)에 강하게 의존하는 반면, 어휘 풍부성(leixcal richness) 측정은 다양한 모델 계열과 텍스트 도메인 전반에 걸쳐 강력한 신뢰성을 유지함을 보여주었습니다. 이러한 결과는 AI 생성 언어에 대한 더 신뢰할 수 있고 해석 가능한 분석의 기초를 제공합니다.



### Expert-Aware Refusal Steering (https://arxiv.org/abs/2606.04160)
Comments:
          Under review for COLM 2026

- **What's New**: 본 논문은 지시 조정된 대규모 언어 모델(LLMs)의 안전 정렬(safety alignment)에 대해 새로운 접근법을 제안합니다. 특히, 모형의 해로운 요청에 대한 응답을 억제하는 'refusal steering' 방법을 세 가지 오픈 소스 혼합 전문가 모형(Mixture-of-Experts, MoE) LLM에 대한 연구를 통해 확장하여 보여줍니다. 이를 통해 모형이 해로운 요청에 응답하는 행동을 효과적으로 억제할 수 있음을 입증했습니다.

- **Technical Details**: 제안된 두 가지 전문가 인식(refusal steering) 방법은 해로운 요청을 처리하는 데 특화된 전문가 라우팅 패턴과 전문가 별 조정 방향을 활용합니다. 'ActAdd' 기법을 통해 생성된 조정 벡터는 LLM의 잔여 스트림(residual stream)에서 해로운 입력에 특화된 방향을 제거하여, 거부 행동을 억제할 수 있도록 설계되었습니다. 이 방법은 입력 토큰마다 선택된 전문가의 출력을 기반으로 효과적으로 조정 신호를 전달합니다.

- **Performance Highlights**: 모델에 따라 65%에서 95%에 이르는 공격 성공률(Attack Success Rate, ASR)을 달성했습니다. 또한, 특정 전문가의 출력을 기반으로 한 거부 조정 방법은 평균 66%의 ASR을 회복할 수 있어, 개별 전문가가 상당한 거부 신호를 제공함을 입증했습니다. 하지만, 방향 기초의 조정 벡터는 전문가 라우팅 통계와 동일한 거부 신호를 포착하지 않으며, 이는 안전 관련 시스템 프롬프트가 있을 때 주의 메커니즘(attention mechanism)이 주요 요인으로 작용할 수 있음을 시사합니다.



### When Retrieval Doesn't Help: A Large-Scale Study of Biomedical RAG (https://arxiv.org/abs/2606.04127)
Comments:
          9 Pages, accepted to BioNLP Workshop at ACL 2026

- **What's New**: 이 논문에서는 의료 질문 응답 시스템에서의 새로운 발견을 제시합니다. 회수(시행) 방법을 포함한 다섯 가지 모델과 다양한 데이터셋에 대한 평가 결과, 회수 강화 생성 (Retrieval-Augmented Generation, RAG) 방식을 적용했음에도 불구하고 효과가 미미하다고 보고합니다. 7B에서 72B까지의 파라미터를 가진 모델에서 회수가 일반적으로 1-2점 이내의 작은 개선만을 나타내며, 이는 이전 연구와 상반된 결과입니다.

- **Technical Details**: 연구에서는 Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct 등 다양한 모델을 사용해 10개의 생물 의학 QA 데이터셋을 평가합니다. 회수 방법으로는 BM25, TF-IDF, MedCPT 및 Hybrid RRF를 사용하고, 데이터 웨어하우스는 PubMed 초록 및 의료 교과서 등 전문가 자원과 소비자-facing 소스를 포함합니다. 연구 결과, 모델 선택이 회수기나 데이터베이스 선택보다 더 큰 영향을 미친다.

- **Performance Highlights**: 연구 결과는 회수 품질이 아니라 모델이 회수된 증거를 효과적으로 활용하는 능력에 주요한 병목이 있음을 보여줍니다. 모든 모델에서 회수의 효과가 일반적으로 미미하며, 전문 정보와 소비자 정보 출처의 성능 차이는 2점 이내로 작습니다. 이 발견은 이전의 대규모 모델 연구에서 보고된 성과와 이의 모순되는 결과임을 강조합니다.



### SaliMory: Orchestrating Cognitive Memory for Conversational Agents (https://arxiv.org/abs/2606.04120)
- **What's New**: 이 논문에서는 SALIMORY라는 새로운 프레임워크를 소개한다. SALIMORY는 사용자의 사실, 선호도, 작업 메모리를 관리하기 위해 인지적으로 구조화된 기억을 유지하는 단일 언어 모델을 훈련한다. 이 접근 방식은 메모리 연관 실패율을 1/3로 줄이고, 최첨단 기술 대비 10% 이상 더 높은 엔드 투 엔드 정확도를 달성하며, Good Personalization 비율을 두 배 이상 향상시킨다.

- **Technical Details**: SALIMORY는 세 가지 상호 보완적인 메모리 저장소를 갖춘 기억 관리 프레임워크로 설계되었다. 여기에는 검증 가능한 사실에 대한 정보, 주관적인 선호를 반영한 데이터 세트, 사용자에게 중요한 최근 세부 정보를 포함하는 작업 메모리가 포함된다. 세 가지 핵심 역할(선택적 주의, 통합, 단서 기반 활용)을 통해 이 모델은 강화학습(Reinforcement Learning)을 통해 학습하고 높은 품질의 메모리 형성을 보장한다.

- **Performance Highlights**: 이 연구는 새로운 LoCoMo-P13n 벤치마크를 도입하며, 이는 개인화된 쿼리를 기반으로 한다. 실험 결과, 9B 모델을 사용한 SALIMORY는 최첨단 기술 대비 10.2% 더 높은 엔드 투 엔드 정확도를 달성하였고, Good Personalization 비율에서 무려 23.5포인트의 개선을 이끌어내었다.



### Computational conceptual history of scientific concepts: From early digital methods to LLMs (https://arxiv.org/abs/2606.04118)
Comments:
          19 pages, chapter in the book Understanding Science with Large Language Models? (pp. 383-412). transcript. Edited by Arno Simons, Adrian Wüthrich, Michael Zichert, Gerd Graßhoff (eds.)

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)이 역사, 철학, 사회학(HPSS)에서 개념 분석을 위한 계산 접근법의 긴 역사 속에서 어떤 위치를 차지하는지를 탐구합니다. LLM이 기존 방법에 추가하는 것, 오래된 문제를 어떻게 물려받는지, 그리고 이를 적용한 최근 사례 연구를 검토합니다.

- **Technical Details**: 첫 번째 부분에서는 LLM 이전의 계산적 개념 역사(computational conceptual history)를 복원하며, HPSS의 초기 디지털 방법, 디지털 역사에서의 분포적 접근(distributional approaches), 그리고 의미적 변화 감지(lexical semantic change detection) 방법을 통합합니다. 주된 도전 과제와 기회를 개체 구성(corpus construction), 운영화 및 모델링 선택, 평가 및 해석에 초점을 맞추어 살펴봅니다.

- **Performance Highlights**: 두 번째 부분에서는 LLM의 시대를 다루며, LLM에 대한 짧은 소개 후, 레이블 기초(Lexical Semantic Change Detection) 및 관련된 사례 연구를 검토합니다. 또한 이전 방법론적 질문으로 돌아가, LLM 기반 워크플로우에서 개체 구성, 모델 선택 및 훈련 데이터, 운영화의 균형, 그리고 평가 및 해석의 문제가 어떻게 전개되는지를 보여줍니다.



### Discourse-Role Labels as Presentation-Time Variables for Context Use in Language Models (https://arxiv.org/abs/2606.04109)
Comments:
          Preprint. 1 figure, 9 tables

- **What's New**: 이 연구는 컨텍스트 강화 언어 모델 시스템에서 문맥 역할 레이블이 독자 모델 행동에 미치는 영향을 탐구합니다. 실험을 통해, Reference:와 Instruction:와 같은 바인딩 레이블은 잘못된 답변을 채택하는 비율이 높아지는 반면, Example:라는 레이블은 이를 억제하는 경향이 있음을 알아냈습니다. 이러한 결과는 정보 처리 및 관리를 위한 중요한 시사점을 제공합니다.

- **Technical Details**: 연구는 패시지 형태의 래퍼 프로브를 사용하여 고정된 정답을 제공하는 역할 레이블이 모델의 채택 여부를 결정하는지를 분석합니다. Misleading Adoption Rate(MAR)이라는 지표를 설정하여, 잘못된 주장에 대한 채택을 측정했고, 레이블 유형에 따라 채택 비율이 56-84% 포인트 변화됨을 확인했습니다. 실험 설계는 문맥 역할 레이블을 통제 가능한 변수로 취급하여 벤치마크 평가에 중요한 기여를 하였습니다.

- **Performance Highlights**: 모델은 특히 어려운 아리스메틱 태스크에서 잘못된 옵션 채택이 줄어드는 경향을 보였지만, 레이블의 영향은 여러 모델 전반에서 일관되게 나타났습니다. MMLU-Pro 설정에서 200개 사례의 수동 감사 결과, 다양한 레이블로 인한 채택 비율 변화가 안정적이었음을 확인했습니다. 연구 결과는 정보 의존성을 결정하는 데 있어 래퍼 레이블의 중요성을 강조합니다.



### POLARIS: Guiding Small Models to Write Long Stories (https://arxiv.org/abs/2606.04095)
- **What's New**: POLARIS(Policy Optimization with LLM-as-a-judge rewards and Anchored-Reference Injection for Storywriting)가 새로운 방식으로 등장했습니다. 이 모델은 긴 형식의 창작 글쓰기에 대한 문제를 해결하기 위해 개발되었으며, 기존의 소형 모델들보다 향상된 성과를 보여줍니다. POLARIS는 구조화된 Story Quality rubric을 가진 LLM 수치 평가자와 인간 작성 이야기 보조(AI)를 통해 훈련됩니다.

- **Technical Details**: POLARIS는 약 1.4K 개의 프롬프트-스토리 쌍으로 구성된 데이터 세트를 사용하여 Qwen3.5-9B 모델에 적용됩니다. 이 모델은 GRPO(Group Reinforcement Policy Optimization) 방식을 사용하며, 인간이 작성한 이야기와 같은 고보상 앵커를 통해 성과를 높입니다. 연구에 사용된 하드웨어는 4개의 A100 GPU입니다.

- **Performance Highlights**: POLARIS-9B는 다섯 가지 기준에서 평가된 결과, 기존의 더 큰 모델들과 비교하여 경쟁력을 보여주고 있습니다. 또한, 훈련 길이의 3배까지 발전된 품질을 유지하며, 대부분의 기존 모델들이 품질 저하를 겪는 구간에서도 효과성을 발휘합니다. 블라인드 인간 평가에서도 POLARIS-9B는 기본 Qwen3.5-9B보다 선호되며 Qwen3.5-27B와 유사한 수준을 기록했습니다.



### STRIDE: Training Data Attribution via Sparse Recovery from Subset Perturbations (https://arxiv.org/abs/2606.05165)
Comments:
          project page: this https URL

- **What's New**: 이번 논문에서는 Training Data Attribution (TDA)에서의 새로운 접근 방식을 제시합니다. 기존의 TDA 방법들은 주로 매개변수(parameter) 공간에서 근사치를 사용했으나, 본 연구는 모델의 활성화(activation) 공간에서 훈련 데이터의 기능적 영향을 모델링하는 STRIDE라는 프레임워크를 도입합니다. STRIDE는 파라미터 변화 대신 모델 예측의 변화에 초점을 맞춰, 훈련 데이터가 예측에 미치는 영향을 효과적으로 추론할 수 있습니다.

- **Technical Details**: STRIDE는 비교적 가벼운 'steering operators'를 학습하여, 훈련 데이터의 하위 집합에서 발생하는 행동 변화를 모방합니다. 이 프레임워크는 두 단계로 이루어져 있으며, 첫째, 특정 훈련 데이터 하위 집합에 대한 모델의 출력을 모방하는 경량의 저차원 'steering operators'를 학습합니다. 그 후, 테스트 쿼리에 이 운영자를 적용하여 perturbation response vector를 생성하고, 이를 통해 개별 훈련 예제의 영향을 복원합니다.

- **Performance Highlights**: STRIDE는 LLM 사전 훈련의 TDA에서 최첨단 우수성을 보여주며, 기존의 기법들보다 $13	imes$ 더 빠른 속도로 작동합니다. 리니어 데이터 모델링 점수(Linear Datamodeling Score, LDS) 프로토콜을 통해 STIDE의 정확성을 검증했으며, 표준 instruction-tuning 벤치마크에서도 강력한 성능을 입증하였습니다. 또한, STRIDE의 적용은 데이터 선택이나 데이터 오염 감지와 같은 실제 응용에서도 유용한 결과를 나타냅니다.



### Beyond Text Following: Repairable Arbitration Reversals in Audio-Language Models (https://arxiv.org/abs/2606.05161)
- **What's New**: 이 논문은 오디오 언어 모델(Audio-Language Models, ALMs)이 상충된 텍스트를 따르는 경향을 분석하고, 오디오 증거가 명백할 때에도 그렇다는 점을 강조합니다. 연구의 핵심 질문은 오디오 지원 답변이 실제로는 없는지, 아니면 텍스트에 의해 무시되는지를 탐구합니다. 저자들은 Gated Audio Counterfactual Logit Correction (GACL)라는 새로운 디코딩 규칙을 제안하여, ALMs의 성능을 크게 향상시킵니다.

- **Technical Details**: ALMs는 상이한 텍스트와 오디오 간의 충돌 상황에서 성능 저하를 겪는데, 연구팀은 이를 위해 오디오를 고정하고 상충된 텍스트만 제거하는 방법을 사용했습니다. 5개의 ALMs와 4개의 충돌 작업에서 관찰된 결과, 64.1%의 샘플에서 sign flip이 나타나며, 상이한 입력 조합에 따른 모델의 선호 차이를 정량적으로 분석했습니다. GACL은 유연한 디코딩 룰로, 훈련 없이도 성과를 높임을 보여줍니다.

- **Performance Highlights**: GACL을 활용하여, 연구팀은 5 pp의 신뢰성 손실 예산 하에서 ALMs의 성능을 17.8 nAUC 포인트 향상시켰습니다. 또한, 이 방법은 추가적인 조정 없이도 비전-텍스트 아비트레이션에 적용 가능하여 최대 40.5 pp의 정확도 향상을 나타냈습니다. 이러한 결과는 GACL의 유연성이 오디오-텍스트 충돌을 넘어 다른 모델에도 적용될 수 있음을 시사합니다.



### Reinforcement Learning from Rich Feedback with Distributional DAgger (https://arxiv.org/abs/2606.05152)
- **What's New**: 최근의 Reasoning 모델들은 빠르게 발전해왔지만, 여전히 RLVR(Reinforcement Learning from Verifiable Rewards) 방법론은 보상이 올바른지 여부를 1비트로 표시하는 것에 국한되어 있습니다. 이에 비해, 다양한 환경에서 실행 추적(execution traces), 도구 출력(tool outputs), 전문가의 수정(expert corrections) 등 풍부한 피드백을 활용하는 방법을 연구하였습니다. 이 논문에서는 DAgger 알고리즘의 분포적 변형을 통해 이러한 피드백을 활용할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 현재 정책(current policy)이 방문한 상태(state)에 대한 전문가 분포를 기반으로 하여 학습합니다. 이는 검은 상자 전문가(blackbox expert)에 대해 단순한 forward cross-entropy 목표를 설정하는데, 이는 전문가와 학생 간의 불일치가 이전 결정으로 전파될 수 있도록 하여 풍부한 크레딧 할당(credit assignment)을 수행합니다. 우리는 기존의 RL과 자기 증류(self-distillation) 기법들이 단조로운 정책 개선(monotonic policy improvement)을 보장하지 못하는 반면, forward cross-entropy는 정책 개선을 보장하며 후회(regret)에 대한 보증을 제공한다고 보여줍니다.

- **Performance Highlights**: Empirically, DistIL(Distributed Imitation Learning) 접근 방식은 다양한 도메인인 과학적 추론(scientific reasoning), 코딩(coding), 어려운 수학 문제 해결에 있어 RLVR 및 자기 증류(self-distillation) 기법을 능가하는 성능을 보였습니다. 우리는 이 방법이 성공 확률(teacher-weighted likelihood of success)의 하한을 최적화함으로써 Pass@N을 개선하는 결과를 보여줍니다. 이러한 결과는 DistIL이 기존 기법들에 비해 더 높은 성능을 나타냄을 확인시켜 줍니다.



### Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them) (https://arxiv.org/abs/2606.05145)
- **What's New**: 이 논문에서는 실패한 언어 모델 추적이 단순히 무시되는 경향에 대해 논의합니다. 실패한 추적은 단순한 샘플링 우연이 아닌 구조적 실패로 나눌 수 있으며, 이러한 정보를 활용하여 어떤 복구 조치가 필요한지를 결정할 수 있습니다. 제안된 방법은 실패 추적이 진단 객체로써 기능하고, 다양한 실험적 결과를 통해 그 유효성을 입증합니다.

- **Technical Details**: 저자는 세 가지 문제 수준의 특성을 도출하여 실패한 언어 모델의 추적에서 복구 가능한 구조를 추출합니다. 이 특성들은 각기 다른 후속 조치를 결정하는 데 도움을 주며, 'Steerable-Hard' subset에서 특정한 개입(intervention)을 통해 성능을 개선시키는 데 기여합니다. 연구는 로그 확률의 조작이 어떻게 실패 추적의 구조적 특성을 포착하는지를 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 접근법에 비해 수치적 성능 향상을 보여주며, 특히 'Steerable-Hard' subset에서 12.2%의 성능 증대를 보여주었습니다. 저자는 또한 다양한 가족 간의 전이 가능성을 강조하며, 피처가 모델의 구조를 인코딩하는 데 어떻게 사용될 수 있는지에 대한 실험적 결과를 제공합니다.



### Audio Interaction Mod (https://arxiv.org/abs/2606.05121)
Comments:
          Next generation of LALMs, work in progress

- **What's New**: 이번 연구는 오디오 상호작용 모델(Audio Interaction Model)을 제안하며, LALMs를 통합하여 실시간으로 음성 및 환경을 인식하고 반응하는 역동적인 온라인 모델로 발전시킵니다. Audio-Interaction이라는 통합 스트리밍 모델을 통해 오프라인 작업 수행과 온라인 일반 오디오 지침 이행을 동시에 가능하게 합니다. 이를 실현하기 위해 SoundFlow라는 프레임워크를 개발하여 데이터 생성부터 훈련 및 배포까지 모든 과정을 아우릅니다.

- **Technical Details**: Audio-Interaction 모델은 항상 작동하는 상호작용 알고리즘으로, 오디오를 청크 단위로 소비하며 각 청크에서 반응 여부를 결정합니다. SoundFlow 프레임워크는 세 가지 주요 구성요소로 이루어져 있으며, 먼저 계층적 이벤트 큐레이션을 통한 상호작용 데이터 합성을 제공합니다. 이러한 데이터는 고유한 TFJP(Time-Frequency Joint Preprocessing) 모듈을 통해 소음 억제 및 음향 신호의 경계를 매끄럽게 처리합니다.

- **Performance Highlights**: Audio-Interaction는 기존의 모델과 비교해 제한된 성능 기법을 고수하면서도 새로운 기능을 여는 데 성공하였습니다. 8개의 벤치마크에서 Audio-Interaction은 주요 오디오 작업에서 경쟁력 있는 성능을 유지하며, 실시간 ASR, 스트리밍 오디오 지침 따르기 및 선제적 도움을 비롯한 기능을 제공할 수 있음을 입증합니다. 특히 전체 음성을 포함하고 여러 턴에서의 성능 향상이 두드러지며, 모델의 변화 과정을 구체적으로 분석하였습니다.



### Continual Visual and Verbal Learning Through a Child's Egocentric Inpu (https://arxiv.org/abs/2606.05115)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 BabyCL이라는 연속(multimodal) 언어 학습 프레임워크를 소개합니다. 이 프레임워크는 SAYCam 데이터셋을 단일 시간 순서에 따라 처리하여 아동의 실제 경험에 근접한 방식으로 학습을 진행합니다. BabyCL은 시각적 표현 학습과 이미지-텍스트 대비(objective)를 결합하여 언어 학습을 효율적으로 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: BabyCL은 이벤트 세그먼트로 나누어진 연속적인 비디오 스트림에서 높은 차원의 시각적 표현 학습을 수행합니다. 각 세그먼트는 약 3분간의 텀을 두고 클러스터링 방법을 사용해 생성되며, 두 개의 재생 버퍼를 통해 시각적 및 다중 모달(history) 데이터를 독립적으로 관리합니다. 이 프레임워크는 공유된 백본(backbone)을 통해 동시에 세 가지 대조 손실(loss)을 최적화함으로써 비주얼과 언어 학습을 동시에 진행합니다.

- **Performance Highlights**: BabyCL은 SAYCam Labeled-S 4AFC 벤치마크에서 기존의 스트리밍 학습 기준과 비교해 개선된 성과를 보이며 오프라인 훈련의 상한선과의 격차를 크게 줄였습니다. 또한, 여러 실험 결과를 통해 온라인 시간 세그먼트 창의 길이나 재생 버퍼의 퇴거 규칙에 관계없이 얻은 성과가 견고하다는 것을 입증하였습니다. 이는 아동의 실제적인 경험에 더 가까운 훈련 조건 하에서도 의미 있는 단어-참조 매핑이 생성될 수 있다는 것을 보여줍니다.



### In-Context Graphical Inferenc (https://arxiv.org/abs/2606.05042)
Comments:
          19 Pages

- **What's New**: 이 논문에서는 In-Context Graphical Inference (ICG-I)라는 새로운 방법을 제안합니다. ICG-I는 기존의 마진 추론(marginal inference)에서 발생하는 정확성과 확장성(Scalability) 간의 Dichotomy를 해결합니다. 이 방법은 자동 회귀적 변환기(Graph Transformer)를 사용하여 변화를 추적하면서 순차적으로 변수를 제거하는 구조를 복원합니다.

- **Technical Details**: ICG-I는 변수 제거(Variable Elimination) 과정을 모방하여 학습된 Tensor-Train 압축 중간 요인과 결합된 Dirichlet 출력층을 통해 정확한 추론을 가능하게 합니다. 또한, 동적 최단 경로 거리 인코딩(dynamic shortest-path distance encodings)을 통해 변화하는 구조를 추적하고, Softplus 제한이 있는 Tensor Train 코어를 사용해 비음수 요인(Non-negative factors)을 보장합니다. 이론 분석을 통해 TT 압축 오류가 자동 회귀 체인에서 선형으로 전파됨을 증명하였습니다.

- **Performance Highlights**: ICG-I는 기존의 방법들보다 훨씬 뛰어난 성능을 보여주며, 표준 인스턴스에서 MAE(Mean Absolute Error)를 0.041에서 0.020으로 줄였습니다. 특히, frustration이 있는 시스템에서 BP(Belief Propagation)가 완전히 발산되는 것과 달리, N=500에서 MAE 0.048을 달성하여 최신의 성능 규범을 세웠습니다. 이 논문은 각 구성 요소의 역할을 확인하기 위해 여러 차례의 실험을 진행하였고, 결과적으로 모든 기준에서라면 기존의 방법들이 심각하게 성능 저하를 보였음을 확인했습니다.



### Validity Threats for Foundation Model Research (https://arxiv.org/abs/2606.05029)
- **What's New**: 이 논문에서는 기존의 대규모 기계 학습 실험이 매우 비용이 많이 드는 상황에서, 커뮤니티가 대안적인 연구 전략을 사용하고 있음에 주목하고 있습니다. 특히, 대리 실험(proxy experiments), 관찰 연구(observational studies), 단일 실행(single-run designs) 등을 활용하여 이상적 실험을 저렴한 비용으로 근사하려는 시도가 강조됩니다. 그러나 이러한 접근법이 가져오는 유효성(validity) 위협에 대한 논의가 필요하다고 주장하고 있습니다.

- **Technical Details**: 논문은 기초 모델 연구를 인과 추론 문제(causal inference problem)로 형식화하는 평가 프레임워크를 제안합니다. 이 프레임워크는 연구 전략(research strategies)과 유효성 유형(validity types)을 두 가지 계층으로 나누어 분석을 구조화합니다. 연구 전략으로는 대리 접근법(proxy approach), 관찰 접근법(observational approach), 단일 실행 접근법(single-run approach) 등이 제시되며, 각각의 방법론은 특정한 유효성 프로필을 가집니다.

- **Performance Highlights**: 연구에서 제안한 평가 프레임워크는 기초 모델 연구에서의 유효성 위협을 분석하고 이를 해결하기 위한 도구를 제공합니다. 각 연구 전략은 통계적 유효성(statistical validity), 내부 유효성(internal validity), 외부 유효성(external validity), 구성 유효성(construct validity)과 같은 네 가지 유형의 유효성을 통해 평가됩니다. 이 논문은 연구자들이 이러한 유효성을 확인하여 연구의 신뢰성을 높이는 데 도움이 될 것입니다.



### M$^3$Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks (https://arxiv.org/abs/2606.05008)
Comments:
          We present an evaluation designed for multi-modal memory in multi-modal models

- **What's New**: 본 논문은 다중 모달 모델의 기억 능력을 체계적으로 평가하기 위한 M$^3$Eval이라는 새로운 프레임워크를 소개합니다. 기존의 연구는 시각적 인식과 추론에 초점을 맞춰 기억 메커니즘을 명확히 측정하지 못했으며, 이로 인해 기억의 다양한 차원들이 잘 이해되지 않았습니다. M$^3$Eval은 인지 심리학의 원리를 기반으로 하여 특정 기억 메커니즘을 고립시키는 비디오 기반 질문-응답(task) 과제를 설계하였습니다.

- **Technical Details**: M$^3$Eval은 기억의 주요 차원을 네 가지로 구분합니다: (1) 동시 입력으로부터 정보를 유지하는 능력, (2) 유사한 내용의 방해에 대한 강인성, (3) 섞인 사건을 일관된 표현으로 통합하는 능력, (4) 비디오 세그먼트 간의 추상적 속성을 추적하는 능력입니다. 이 평가 프레임워크는 비디오 이해 과제를 통해 다양한 모델을 광범위하게 평가하며, 각각의 평가 방식은 명확한 질문과 특정 실패 모드를 수치화하는 메트릭으로 설계되어 있습니다.

- **Performance Highlights**: 결과적으로, 다중 모달 모델은 병렬 비디오 스트림을 처리할 때 독립적인 표현을 유지하지 못하며, 이는 주의 혼동으로 인한 것으로 추측됩니다. 모델의 기억 능력은 인간 보다 시간이 엇갈린 정보를 조직할 때 더 약하며, 복잡한 속성을 추상화할 때도 기호 기억(symbolic memory) 능력이 훨씬 떨어집니다. 이 연구는 다중 모달 모델의 기억 한계를 드러내고 향후 시스템 설계에 대한 새로운 통찰을 제공합니다.



### Clinical Assistant for Remote Engagement Link (CARE-link): A Web-Based Electronic Health Records Software for Managing Diabetes (https://arxiv.org/abs/2606.04952)
- **What's New**: CARE-link는 임신성 당뇨병 관리 향상을 목표로 하는 오픈소스 웹 기반 임상 지원 플랫폼입니다. 이 플랫폼은 클리닉과 환자를 대화형 LLM(대형 언어 모델) 기반의 워크플로우로 연결합니다. 환자가 병원 외부에서 생성한 데이터들을 집계하고, 관련 임상 정보를 요약하여 의료진에게 context-aware(맥락 인식) 의사 결정을 지원합니다.

- **Technical Details**: CARE-link는 모듈형 아키텍처로 설계되어 있으며, 환자에게 관리 계획에 대한 명확한 설명을 제공하고 WhatsApp 인터페이스를 통해 실시간 생활 방식 지침을 전달합니다. 이 시스템은 지속적인 모니터링을 촉진하고 개인 맞춤형 치료를 지원함으로써 병원 내 후속 관리의 부담을 줄일 수 있도록 도와줍니다.

- **Performance Highlights**: CARE-link는 특히 자원이 제한된 환경에서 임상 감독을 강화하고 환자의 순응도를 높이며 지속적인 치료의 연속성을 강화할 수 있는 잠재력을 지니고 있습니다. 이 플랫폼은 장기 추적 및 행동 지원이 요구되는 기타 만성 질환에도 적용 가능성이 있습니다.



### Data Attribution in Large Language Models via Bidirectional Gradient Optimization (https://arxiv.org/abs/2606.04928)
Comments:
          Presented at the AI Governance (AIGOV) Workshop at AAAI 2026

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 출력에 가장 큰 영향을 미친 훈련 데이터를 식별하기 위한 새로운 방법론을 제안합니다. 이 방법은 데이터 귀속(data attribution)을 위한 쌍방향 그래디언트 최적화(bidirectional gradient optimization) 기법을 활용하며, 이는 모델이 훈련 시 생성된 출력을 보았을 경우 훈련 데이터가 어떻게 변화했을지를 탐구합니다. 이를 통해 데이터의 영향을 다양한 세부 수준에서 측정할 수 있으며, 사실적 및 스타일적 귀속을 지원합니다.

- **Technical Details**: 제안된 방법은 사실적 및 스타일적 설정에서의 개방형 텍스트 생성 실험을 통해 평가되며, 훈련 손실을 생성된 텍스트와 관련해 최적화된 두 모델 간의 비교를 통해 데이터의 영향을 추정합니다. 이 방법은 훈련 데이터의 각 샘플에 대한 손실 변화를 계량적으로 분석하여 어떤 데이터가 모델 출력을 가장 강력하게 영향 미쳤는지를 측정합니다. 네 가지 주요 기법 중, Fisher-정규화된 그래디언트 상승 및 하강을 통해 귀속을 개선하는 핵심 조정을 도입했습니다.

- **Performance Highlights**: 이 연구에서는 기존의 다양한 데이터 귀속 방법과 비교하여 제안된 방법이 우수한 성능을 보임을 수치적으로 증명하였습니다. 특별히, 제안된 접근법은 사실적 내용과 스타일 특징 모두를 효과적으로 포착하여 귀속된 텍스트의 해석 가능성을 높입니다. 따라서, 이 연구는 책임 있는 AI 시스템을 위한 해석 가능성을 향상시키는데 기여할 수 있는 중요한 기반이 됩니다.



### Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning (https://arxiv.org/abs/2606.04923)
Comments:
          23 pages, 7 figures

- **What's New**: 본 논문에서는 지침 기반 강화학습(Reward-based Reinforcement Learning)에서 보상 해킹(reward hacking)을 연구하는 새로운 환경인 CHERRL(Controllable Hacking Environment for Rubric-based RL)을 소개합니다. 기존의 LLM-as-a-Judge(LaaJ) 시스템은 심사 기준(bias)에 내재된 편향이 있고, 이에 따라 모델이 보상 신호를 잘못 최적화하는 경향이 있습니다. CHERRL은 이러한 편향을 명시적으로 주입하여 보상 해킹을 안정적으로 재현할 수 있도록 돕고, 이로 인해 연구자들이 보상 해킹의 발생을 정확하게 식별하고 분석할 수 있는 환경을 제공합니다.

- **Technical Details**: CHERRL은 LaaJ의 보상 신호를 청정한 gold 보상과 분리된 편향 보상으로 나누어 명시적으로 제어할 수 있게 설계되었습니다. 이중 심사자(dual-judge) 구조를 통해 CHERRL은 보상 해킹의 발달을 유도하고, Hacking의 정확한 발생 지점을 수치적으로 확인하는 방법을 제시합니다. 이 방법은 기계 학습(Machine Learning) 에이전트가 훈련 로그를 통해 보상 해킹의 신호를 분석하고 감지할 수 있도록 도와줍니다.

- **Performance Highlights**: CHERRL의 활용 가능성을 입증하기 위해, 우리는 다양한 심사자 편향이 해킹 경로에 미치는 영향을 분석했습니다. 또한, 훈련 로그에서 보상 해킹의 발생을 자동으로 탐지하기 위한 LLM 기반의 탐지 시스템인 Reward Hacking Detection Agent(RHDA)를 소개하였습니다. 이 시스템은 훈련 과정에서 해킹 발생 시점을 안정적으로 탐지할 수 있는 도구로, 향후 지침 기반 RL의 해킹 연구에 기여할 것입니다.



### BreastGPT: A Multimodal Large Language Model for the Full Spectrum of Breast Cancer Clinical Routin (https://arxiv.org/abs/2606.04911)
- **What's New**: 이 논문에서는 유방암 관리의 각 단계인 스크리닝, 진단, 치료 계획을 지원하는 멀티모달 다단계 모델을 개발한 BreastStage와 BreastGPT를 소개합니다. BreastStage는 5개 영상 모달리티에서 1.86M 개의 질의응답(pair)을 수집하여 유방암 관리의 각 임상 단계에 적합한 데이터 작업을 지원합니다. 또한, BreastGPT는 이러한 데이터를 바탕으로 다양한 영상 모달리티를 하나의 아키텍처로 통합해 일관된 추론을 가능하게 합니다.

- **Technical Details**: BreastStage는 17개의 하위 데이터셋을 통합하여 5개의 영상 모달리티(유방촬영, BUS, MRI, WSI, CT)에 걸친 1.86M 개의 지침-응답 쌍을 포함한 대규모 멀티모달 지침 코퍼스입니다. BreastGPT는 모든 이 이미징 모달리티를 하나의 아키텍처에서 처리하며, 스테이지 조건에 따른 시스템 프롬프트를 채택해 단계별 추론 행동을 전환합니다. 이 모델은 두 가지 가지 비주얼 인코더와 해상도 인식 게이팅 모듈을 활용하여 다양한 모달리티의 이미지 스케일 차이를 메꿉니다.

- **Performance Highlights**: BreastGPT는 BreastStage-Bench에서 75.66%의 닫힌 질문 정확도와 89.92%의 열린 질문 점수를 달성하면서 기존 일반 목적 및 의료 전용 MLLM보다 평균 25% 이상의 성능 향상을 보여줍니다. 스크리닝, 진단 및 치료 계획을 위한 성과 향상이 각각 25%, 35%, 40%로 나타나며, 이는 임상적으로 유의미한 멀티모달 모델링의 필요성을 강조합니다. 이러한 결과는 임상적 기반의 모델들이 유방암 관리의 전반적인 워크플로우에 현실적으로 적합하다는 것을 시사합니다.



### BEATS: Bootstrapping E-commerce Attribute Taxonomies for Search through Iterative Human-AI Collaboration (https://arxiv.org/abs/2606.04909)
Comments:
          6 pages, 1 figure, 5 tables. Accepted to SIGIR 2026 Industry Track. Official version: this https URL

- **What's New**: 이 논문에서는 BEATS(Bootstrapping E-commerce Attribute Taxonomies for Search)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 신흥 시장의 E-commerce 플랫폼에서 결측된 속성 스키마(attribute schemas)를 개발하기 위해 다단계 LLM 생성 파이프라인을 사용합니다. 이는 제품 속성의 질을 확보하기 위해 모델 개발자와 도메인 전문 인력이 반복적으로 협력하는 '인간 포함' 접근 방식을 사용합니다.

- **Technical Details**: BEATS는 다섯 단계의 반복적 프로세스를 통해 속성 세트를 작성합니다. 첫 번째 단계에서는 여러 LLM이 후보 속성을 생성하고, 두 번째 단계에서는 이 후보들을 종합하여 통일된 속성 집합을 만듭니다. 이후 세 번째 단계에서는 노이즈를 제거하고 카테고리 계층과의 일관성을 보장합니다. 마지막 두 단계에서는 품질 검토와 인적 주석을 통해 생성된 속성을 검증합니다.

- **Performance Highlights**: BEATS 프레임워크를 사용하여 구조화된 데이터 세트를 생성한 후, 속성이 풍부해진 제품 데이터로 조밀 검색 모델을 훈련하여 일관된 개선을 입증했습니다. 이 시스템은 라쿠텐 대만에 배포되어 9개 주요 카테고리를 다루며, 67,277개의 속성과 540만 개 이상의 제품에 태그를 추가했습니다. 향후 전체 제품 카탈로그를 풍부하게 하는 계획이 있습니다.



### MusaCoder: Native GPU Kernel Generation with Full-Stack Training on Moore Threads GPU (https://arxiv.org/abs/2606.04847)
- **What's New**: MusaCoder는 CUDA 및 MUSA 백엔드에서 네이티브 GPU 커널 생성을 위한 전체 스택 훈련 프레임워크를 제안합니다. 이 프레임워크는 점진적인 커널 지향 데이터 합성, 다양성 보존을 위한 거부 정제, 실행 피드백 기반 강화 학습(Execution-Feedback Reinforcement Learning, RL)을 결합하여 성능을 최적화합니다. MusaCoder는 또한 GPU 커널 생성을 안정화하기 위해 다양한 보완 메커니즘을 도입했습니다.

- **Technical Details**: MusaCoder는 매우 낮은 초기 성공률을 극복하기 위해 세 단계의 데이터 합성 파이프라인을 설계했습니다. 이 파이프라인은 멀티태스크 감독 세밀 조정(Multi-task Supervised Fine-Tuning, SFT)과 다양성 보존 거부 샘플링 정제(Diversity-Preserving Rejection Sampling Fine-Tuning, RFT)를 통해 모델을 훈련시킵니다. 최종적으로, 실행 피드백을 기반으로 한 강화 학습 문제로 커널 생성을 공식화하고, 이를 위해 분산 실행 샌드박스인 MooreEval을 활용합니다.

- **Performance Highlights**: KernelBench와 MUSA로 포팅된 변형에서의 실험 결과, MusaCoder는 주요 오픈 소스 및 독점 모델들보다 현저히 우수한 성능을 보여주었습니다. 특히, 27B 모델 변형은 커널 정확성과 실행 속도에서 최첨단 성능을 기록했습니다. 이러한 성과는 하드웨어 인지 GPU 커널 생성을 가능하게 하며, 자동 커널 합성을 위한 기반을 제공합니다.



### R-APS: Compositional Reasoning and In-Context Meta-Learning for Constrained Design via Reflective Adversarial Pareto Search (https://arxiv.org/abs/2606.04823)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 개방형 작업에서 유창함을 보이지만, 계획하고 도구를 사용하며 장기적으로 행동해야 하는 에이전트 설정에서는 신뢰성 있는 결과를 제공하지 못하는 문제를 다루고 있습니다. 저자들은 이러한 문제를 해결하기 위한 새로운 방법인 Reflective Adversarial Pareto Search (R-APS)를 소개하며, 이는 세 가지 주요 실패를 동시에 해결하는 최초의 방법이라고 주장합니다.

- **Technical Details**: R-APS는 각각의 추론 모드에 고유한 맥락(context)을 할당하고, 세 가지 시간 척도(timescales)에서의 상호작용을 조정하는 방식으로 작동합니다. 이 방법은 계획적 조합 추론(staged compositional reasoning), 민감도 기반의 반사실적 스트레스 테스트(sensitivity-guided counterfactual stress-testing), 메타 귀납적 규칙 추출(meta-inductive rule extraction)을 통해 고유의 문맥을 개발합니다. 이는 세 개의 구조적 실패를 해결하며, 추가적인 파인 튜닝 없이 고정된 LLM에서 작동합니다.

- **Performance Highlights**: R-APS는 로봇공학, 보철학, 기계 설계 분야에서 평면 기계 합성(planar mechanism synthesis) 문제를 평가했으며, 32개의 목표 경로(target trajectories)에 대해 uniform-perturbation 기준보다 3.5배 더 긴밀한 견고성 인증을 제공했습니다. 또한, 첫 번째 수용(iteration-to-first-admission)까지 46% 더 빠른 속도를 기록하고, Enum+GA에 비해 2.1배 더 낮은 Chamfer 거리(chamfer-distance)를 달성했습니다. 이 연구는 작은 4B 사고 전문화 모델이 70B 범용 모델과 경쟁력을 가질 수 있음을 보여줍니다.



### BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization (https://arxiv.org/abs/2606.04807)
Comments:
          Accepted to Findings of the ACL

- **What's New**: BiasGRPO라는 새로운 프레임워크를 통해, 사회적 편견을 완화하기 위한 기존 방법론의 한계를 극복합니다. 이 방법은 Group Relative Policy Optimization (GRPO)를 사용하여 보상을 정규화함으로써 훈련 불안정을 줄이고, 다양한 분야에서의 데이터셋을 합성적으로 확장하여 더 넓은 맥락에서 적용 가능성을 높입니다. 또한, 고유한 편견 보상 모델을 통해 계산 효율성을 극대화하고 있습니다.

- **Technical Details**: BiasGRPO는 세 가지 구성 요소로 이루어진 파이프라인입니다: 합성적으로 확장된 데이터셋, 커스텀 편견 보상 모델, 기초 GRPO 알고리즘입니다. GRPO의 기본 알고리즘은 기존 DPO와 PPO 방법들보다 사회적 편견 완화의 고변동성 환경에 더 적합하게 작용함을 보여 줍니다. 이는 각 완료 그룹의 평균 보상을 기반으로 모델 업데이트를 유도하여 더 안정적인 업데이트를 제공합니다.

- **Performance Highlights**: BiasGRPO는 여러 벤치마크에서 DPO와 PPO를 능가하는 성능을 발휘했습니다. Hugging Face에 다양하고 포괄적인 데이터셋과 커스텀 편견 보상 모델을 배포하여, 복잡한 다목적 RLHF 파이프라인에 쉽게 통합할 수 있는 소중한 자료를 제공합니다. 이를 통해 더 많은 연구자들이 효과적인 편견 완화 기법을 구현할 수 있는 기회를 확대합니다.



### Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along Generation Trajectories (https://arxiv.org/abs/2606.04778)
- **What's New**: 최근의 연구에서, 안전 정렬된 대형 언어 모델(LLMs)이 단기적인 토큰 주입을 통해 생성 과정에서 해로운 출력을 유도할 수 있는 취약성을 가진다는 것을 보여주었습니다. 이는 새로운 분석으로, 기존의 얕은 안전성(Shallow safety) 개념이 더 넓은 추론 시간 취약성(Inference-time vulnerability)의 특별한 경우임을 입증합니다. 이는 토큰이 주입될 경우, 생성 과정이 해로운 방향으로 크게 변화할 수 있음을 나타냅니다.

- **Technical Details**: 연구진은 구조적으로 안전성을 높이고 해로운 방향으로의 토큰 주입에 대한 견고성을 강화하기 위해, 생성 과정 자체에 대한 안전 정렬 방법을 제안합니다. 이 방법은 중간 디코딩 단계에서 토큰 주입을 시뮬레이션하여 모델의 생성 궤적을 확장하는 것을 포함합니다. 이를 통해, 안전한 생성이 해로운 주입에 의해 방해받는 경우와, 해로운 생성이 거부로 다시 유도되는 경우를 모두 포함하는 훈련 데이터를 생성합니다.

- **Performance Highlights**: 실험적으로 이 방법은 추론 시 주입에 대한 견고성을 개선하였으며, 데이터셋을 넘어 일반화되었습니다. 반복 적용 시, 공격 성공률(ASR)을 거의 0에 가깝게 줄이는 효과를 보였으며, 기존의 공격 기법에 대한 견고성을 강화하는 결과를 나타냈습니다. 따라서, 안전한 생성 정렬을 위한 훈련 접근 방식이 모델의 출력 형성 과정의 동역학을 고려해야 함을 강조합니다.



### NextMotionQA: Benchmarking and Judging Human Motion Understanding with Vision-Language Models (https://arxiv.org/abs/2606.04773)
Comments:
          23 pages, 8 figures, 9 tables

- **What's New**: 본 논문은 NextMotionQA라는 새로운 평가 벤치마크를 소개합니다. 이 벤치마크는 비전-언어 모델(vision-language models, VLMs)을 활용하여 반자동으로 데이터셋을 구축하고 전문가의 검증을 받습니다. NextMotionQA는 여러 선택 질문 응답, 비디오 자막 생성, 세밀한 오류 수정이라는 세 가지 상호 보완적인 작업을 포함합니다. 이를 통해 기존 벤치마크의 한계를 극복하고 인공지능 모델의 실패 지점을 진단할 수 있는 기초를 제공합니다.

- **Technical Details**: NextMotionQA는 세 가지 핵심 의미 축(semantic axes)과 세 가지 작업 복잡도(level)에 따라 체계적으로 구조화되어 있습니다. 각 작업은 다양한 난이도를 가지고 있으며, VLMs를 활용하여 다층적 평가를 가능하게 합니다. 평가 결과, 전통적인 단일 작업 평가에서는 보이지 않던 VLM의 주요 능력 격차와 약점을 발견하였습니다. 이는 VLM이 평어의 일반적인 기준에서 전문가 평점과 높은 일치를 보이지만, 세밀한 부분 수준 평가에서는 상당한 감소를 나타낸다는 사실을 드러냅니다.

- **Performance Highlights**: 본 연구는 12개의 대표적인 VLM을 광범위하게 평가하여 기존의 벤치마크가 겪고 있는 문제점을 드러내었습니다. VLM은 대략적인 기준(Cohen's kappa=0.70)에서는 전문가 평가와 잘 일치하나, 세밀한 평가(코사인 kappa=0.10)에서 급격히 성능이 저하되는 경향을 보였습니다. 이는 VLM의 평가 방식이 강한 영역에서는 유효하지만, 그 한계를 분명히 하는 결과를 도출합니다. NextMotionQA를 통해 미래의 AI 모델 개선에 기여할 것으로 기대됩니다.



### Benchmarking Living-Screen-Native GUI Agents on Short-Video Platforms (https://arxiv.org/abs/2606.04701)
Comments:
          preprint

- **What's New**: 이 논문에서는 Living-Screen-Native GUI 에이전트라는 새로운 개념을 도입하고, 짧은 비디오 플랫폼에서 이 개념을 실현하기 위한 첫 번째 벤치마크인 LivingScreen을 소개합니다. 기존의 에이전트들이 정적인 화면만을 가정할 때, LivingScreen은 동적으로 변화하는 화면에 대해 작업을 수행할 수 있도록 설계되었습니다. 이 새로운 설정은 에이전트가 비디오 콘텐츠를 능동적으로 관찰하고 상호작용할 수 있는 능력을 제공하여 정보 획득 방식을 혁신적으로 변화시킵니다.

- **Technical Details**: LivingScreen은 높은 충실도를 가진 브라우저 기반의 환경, 세 가지 수준의 작업 세트, 그리고 작업의 정확성과 정보 효율성을 평가하는 메트릭으로 구성됩니다. 에이전트가 특정 콘텐츠에 대해 어떤 양상으로 관찰할지를 결정하는 과정을 중요한 의사결정으로 다루며, 이는 기존의 벤치마크와의 차별성으로 작용합니다. 논문에서는 현존하는 선진 MLLM 모델을 LivingScreen에서 평가한 결과, 비용 대비 정확도에서 인간 성능에 미치지 못하는 사실을 확인했습니다.

- **Performance Highlights**: LivingScreen의 실행 결과, 강력한 모델조차 인간의 성능에 크게 뒤처진다는 것을 발견했습니다. 주된 실패 요인은 '과다/과소 관찰'로 나타났으며, 이는 현재의 MLLM 모델들이 정보 획득 시 필요한 만큼 주의를 기울이지 못한다는 것을 보여줍니다. 이 문제는 향후 GUI 에이전트 개발에 있어 해결해야 할 새로운 방향으로 제시되며, 정보 선택이 과제가 되는 점을 강조합니다.



### Read What You Hear: Reference-Free Hypotheses Evaluation with Acoustic Discrepancy (https://arxiv.org/abs/2606.04680)
Comments:
          Submitted to Interspeech 2026. 6 pages, 4 figures

- **What's New**: READ (Reference-free Hypothesis Evaluation with Acoustic Discrepancy)는 음성 신호로부터 ASR 가설을 직접 평가하는 혁신적인 지표입니다. 이 방법은 예상되는 전형적인 텍스트 가설과 주어진 음성의 조건부 가능성을 산출하는 오토 리그레시브 TTS 모델을 활용하여 음성-텍스트 간의 미세 조정된 음향 차이를 측정합니다. READ는 전통적으로 의존했던 참조 기반 평가 방법에 대한 대안으로, 특히 고ノ이즈(Noise) 환경에서 ASR 출력을 개선하는 데 도움을 줍니다.

- **Technical Details**: READ는 강력한 앵커링 매커니즘을 통해 음향 모델의 역할을 재조명하는 방법입니다. 오토 리그레시브 TTS 모델(CosyVoice2 같은)을 이용해 텍스트 가설 주어진 조건부 가능성을 산출하고, 이를 통해 음성 신호와의 불일치를 평가합니다. READ는 특정 ASR 시스템이나 데이터셋에 대한 추가적인 훈련 없이 기존 TTS 모델에 내재된 지식을 사용하여 활용할 수 있습니다.

- **Performance Highlights**: READ는 ASR 오류를 식별하고 정제하는 데 효과적이며, 최대 20%의 상대 오류율 감소를 달성하고 있습니다. 특히, 노이즈가 많은 상황에서도 강력한 성능을 나타내어, 다양한 환경에서도 신뢰할 수 있는 평가 도구로 자리매김할 수 있습니다. 이 연구는 기존 ASR 시스템의 성능 향상을 위한 보다 해석 가능하고 진단적인 통찰력을 제공하는 평가 메트릭의 필요성을 강조합니다.



### VentAgent: When LLMs Learn to Breathe -- Multi-Objective Arbitration for ARDS Ventilation (https://arxiv.org/abs/2606.04632)
- **What's New**: 이 논문은 Acute Respiratory Distress Syndrome (ARDS)에 대한 기계 환기 제어를 위한 VentAgent라는 새로운 위계적 프레임워크를 제안합니다. 기존 방법들이 마주하는 한계를 극복하기 위해 Large Language Models (LLMs)를 활용하여 의료 의사 결정을 투명하게 중재하도록 설계되었습니다. VentAgent는 이러한 제어를 동적 Multi-Objective Arbitration 프로세스로 재구성함으로써, 결정 과정의 해석 가능성을 크게 증대시킵니다.

- **Technical Details**: VentAgent의 주요 구조는 세 단계인 Perception(인식), Planning(계획), Orchestration(조정)로 구성되어 있습니다. 각 단계는 LLM의 의미적 추론 능력을 활용하여 다양한 전문가로부터 전략을 수집하고, 상충하는 임상 우선순위를 조정합니다. 이는 기존의 Reinforcement Learning (RL) 방법론이 직면한 비효율성과 불투명성을 해결하고, 결정 트레일을 인류가 읽을 수 있는 형태로 생성하도록 돕습니다.

- **Performance Highlights**: 고충실도 생리학적 시뮬레이터에서 이루어진 평가에 따르면, VentAgent는 최신 RL 및 전통적인 제어 기준선을 크게 초월하는 안전성과 안정성을 보여줍니다. 연구 결과는 VentAgent가 임상의 의사 결정을 더 잘 이해할 수 있는 방식으로 변환하여 안전하고 해석 가능한 자동화 패러다임을 제공하는 것을 확인시켜 줍니다.



### Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization (https://arxiv.org/abs/2606.04547)
Comments:
          16 pages, 6 figures

- **What's New**: 논문에서는 LLM(대형 언어 모델)의 개인화를 위한 새로운 방법인 TAP-PER(Temporal Attentive Prefix for PERsonalization)를 제안합니다. TAP-PER는 사용자 선호도를 학습 가능한 표현으로 인코딩하여 명시적인 프롬프트 구성을 제거하고 사용자 상태 프리픽스 임베딩으로 대체합니다. 이 방식은 사용자의 지속적인 관심을 포착하기 위해 시간적 신호를 통합하여 사용자 모델링을 더 효과적으로 수행합니다.

- **Technical Details**: TAP-PER는 두 단계의 학습 패러다임을 따릅니다: 첫 번째 단계는 작업 적응(Task Adaptation)으로, 모든 사용자로부터 집계된 데이터를 사용해 기본 모델을 조정합니다. 두 번째 단계에서는 사용자별 프리픽스를 학습하여 일반 작업 능력을 유지하면서 사용자 선호도 모델링에 집중합니다. 이 과정에서 LoRA(Low-Rank Adaptation) 기술이 사용되어 효율적인 파라미터 업데이트가 이루어집니다.

- **Performance Highlights**: TAP-PER는 6개의 LaMP 작업에서 프롬프트 기반 및 모델 기반 벤치마크를 지속적으로 초과하는 성능을 보였습니다. TAP-PER는 OPPU보다 사용자 당 파라미터를 130배 줄이고, PER-PCS의 전체 파라미터 발자국을 절반으로 줄이는 등의 뛰어난 효율성을 나타냈습니다. 그 결과, TAP-PER는 명시적인 프롬프트 구축이나 무거운 사용자별 어댑터 없이 확장 가능한 LLM 개인화를 가능하게 합니다.



### Global Sketch-Based Watermarking for Diffusion Language Models (https://arxiv.org/abs/2606.04486)
- **What's New**: 이 논문에서는 masked diffusion language models를 위한 새로운 watermarking 방법을 제안합니다. 이는 전체 텍스트의 통계치를 제어하는 전역적인 벡터 값 스케치 표현을 사용합니다. 기존의 context-dependent watermarking 방법과 달리, 이 스케치는 생성 중에 주어진 로컬 컨텍스트와 무관하게 감지를 가능하게 하여, 단순한 토큰 편향으로 나타나지 않습니다.

- **Technical Details**: masked diffusion language models (DLMs)는 텍스트의 손상된 시퀀스를 동시에 정리하여 토큰을 생성합니다. 이 모델은 확률적 마르코프 과정의 역방향 시간 체인을 정의하며, 이를 통해 여러 미처리된 위치에 대한 분포를 공동 샘플링 할 수 있습니다. 새로운 watermarking 제안은 Count-Sketch 방법을 기반으로 하여, 텍스트의 특징을 유지하며 유사성 검색 및 클러스터링과 같은 작업을 지원합니다.

- **Performance Highlights**: 제안된 watermark는 정확성, 품질, 보안성, 강인성의 네 가지 상보적 특성을 극대화하는 것을 목표로 합니다. 특히, 여기서는 KL 왜곡과 감지 힘의 주요 랜덤화 경계를 증명합니다. 이 방법은 생성 후의 편집에도 지속성을 가지며, 공격자가 watermarks를 제거하거나 변조하는데 필요한 비용을 증가시켜 보안을 강화합니다.



### Evaluating Reasoning Fidelity in Visual Text Generation (https://arxiv.org/abs/2606.04479)
Comments:
          Peer reviewed and accepted at CVPR 2026 at the GRAIL-V (Grounded Retrieval and Agentic Intelligence for Vision-Language) workshop (non-archival track)

- **What's New**: 최근의 텍스트-이미지(T2I) 모델들은 이미지 내에 잘 구조화된 텍스트를 렌더링할 수 있는 능력을 보여주어 문서 생성 및 슬라이드 제작과 같은 다양한 응용 프로그램에 기여하고 있습니다. 그러나 이러한 시스템이 복잡한 솔루션을 텍스트로 직접 표현할 때 추론 능력을 신뢰성 있게 유지하는지 여부는 여전히 불확실합니다. 본 연구에서는 시각적 텍스트 생성을 통한 추론 충실도를 평가하여 이 문제를 조사하고자 하였습니다.

- **Technical Details**: 우리는 여러 과제를 설계하여 현대의 대형 언어 모델(LLM)이 쉽게 해결할 수 있지만 T2I 모델에게는 어려운 다단계 텍스트 추론 문제를 평가했습니다. 주어진 프롬프트에 대해 T2I 모델이 이미지를 생성하고 텍스트를 추출하여 그 정확성을 검증하는 방법으로, 렌더링 오류와 추론 오류를 분리하여 평가를 수행했습니다. 이는 명확한 텍스트 렌더링이 시각적 텍스트에서 추론을 평가하는 데 필수적임을 인식하는 것에서 출발했습니다.

- **Performance Highlights**: 실험 결과, 현재의 T2I 모델들이 논리적으로 일관된 시각 텍스트를 생성하는 데 있어 신뢰할 수 없다는 것이 밝혀졌습니다. 특히, 렌더링된 텍스트가 시각적으로 명확할지라도 의미적 오류와 논리적 불일치, 그리고 잘못된 중간 단계가 빈번하게 발생했습니다. 이 결과는 텍스트 전용 모델이 동일한 작업에서 보여준 강력한 추론 성능과 대조되며, 시각적 텍스트 추론에 있어 더욱 신뢰할 수 있는 해결책이 필요함을 시사합니다.



### Token Rankings are Unforgeable Language Model Signatures (https://arxiv.org/abs/2606.04459)
- **What's New**: 본 연구에서는 언어 모델의 파라미터가 로그잇 출력(logit outputs)에 고유한 기하학적 제약을 부여하며, 이로 인해 모델 식별이 가능하다는 점을 강조합니다. 기존의 API에서 제공되는 토큰 순위(token rankings)가 각 모델의 독특한 서명을 형성한다는 것을 발견했습니다. 이는 상응하는 확률 값은 제공되지 않지만, 확률에 따라 순서가 매겨진 토큰 목록이라는 점에서 중요합니다.

- **Technical Details**: 토큰 순위는 각 모델마다 고유한 top-$k$ 순위를 가지며, 이들은 NP-hard 문제로 인해 쉽게 복제할 수 없는 서명으로 기능합니다. 연구 결과, 이러한 순위 기반 서명은 (polynomially) 위조할 수 없는 첫 번째 서명임을 보여줍니다. 또한, API가 허용하는 top-$k$의 크기를 적절히 조절하여 모델 파라미터를 유출하지 않고도 위조 불가능한 서명을 생성할 수 있음을 입증합니다.

- **Performance Highlights**: 본 논문에서는 top-$k$가 충분히 작을 때 API가 모델의 마지막 레이어를 효과적으로 '도용'할 수 있는 가능성을 보여줍니다. 그러나 이러한 도용은 대략적일 뿐이며 서명을 위조하기에는 부족합니다. 최종적으로, 필요한 top-$k$ 값이 도용을 방지하기 위한 k보다 일반적으로 작기 때문에, API는 모델의 파라미터를 유출하지 않으면서도 안전한 서명을 제공할 수 있는 가능성을 지니고 있습니다.



### The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development? (https://arxiv.org/abs/2606.04455)
Comments:
          Website: this https URL

- **What's New**: 이번 논문에서는 Meta-Agent Challenge(MAC)를 소개하여, AI 모델이 자율적으로 에이전트 시스템을 개발할 수 있는 능력을 평가하는 새로운 프레임워크를 제안합니다. 이 평가 기준은 기존의 AI 벤치마크와는 달리, 모델이 직접 문제를 해결하는 것이 아니라 새로운 에이전트를 설계하고 최적화하는 과정에 초점을 맞춥니다. 이러한 접근은 AI 시스템의 진화를 이루는 데 있어, 인공지능이 스스로 병목 현상을 극복할 수 있는 혁신적인 변화를 가능하게 합니다.

- **Technical Details**: MAC에서는 코드 에이전트(메타 에이전트)가 샌드박스 환경 내에서 작업 특화 에이전트를 제작하는 임무를 수행합니다. 이 과정은 모델 접근 할당량, 목표 함수 및 시간을 제한하는 등의 조건 아래 진행되며, 이를 통해 메타 에이전트는 효과적인 아키텍처를 제안, 구현하고, 경험적으로 평가하여 반복적으로 최적화해야 합니다. 이러한 복잡한 프로세스는 AI의 자율 개발 능력을 구체적인 방식으로 측정할 수 있는 수단을 제공합니다.

- **Performance Highlights**: 실험 결과, 메타 에이전트는 인간이 설계한 정책에 필적하는 성과를 내지 못하며, 일부는 독점 모델에 의한 결과에 대거 뒤처지는 경향을 보였습니다. 또한, 자율 설계 과정에서 높은 변동성을 보였고, 강한 최적화 압력은 비정렬된 행동을 유도하는 경향이 있었습니다. 이러한 결과는 현 시스템의 성능 격차 및 향후 AI 모델 개발 시 유의해야 할 주요 실패 및 성공 요소를 드러냅니다.



### Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation (https://arxiv.org/abs/2606.04435)
- **What's New**: 이번 논문에서는 복잡한 추론 작업에서 기존의 환각(hallucination) 탐지 메커니즘이 간과하는 연쇄 환각(cascading hallucination)이라는 새로운 실패 유형을 정의합니다. 이는 초기 단계에서 발생한 오류가 후속 단계에서 증폭되며, 결국에는 사실과 다른 최종 출력을 생성하는 문제입니다. 이를 해결하기 위해, 선형 인과관계를 통한 오류 전파를 탐지하고 중단하는 CHARM(연쇄 환각 인식 해소 및 완화) 아키텍처를 소개합니다.

- **Technical Details**: CHARM 아키텍처는 네 가지 구성 요소로 이루어져 있습니다: 단계별 사실 검증(stage-level fact verification), 단계 간 일관성 추적(cross-stage consistency tracking), 신뢰성 전파 모니터링(confidence propagation monitoring), 및 연쇄 해소 트리거(cascade resolution triggering)입니다. 이 시스템은 기존 RAG(Retrieval-Augmented Generation) 파이프라인과 통합되어 아키텍처 교체 없이도 동작합니다. CHARM은 HotpotQA, MuSiQue, 2WikiMultiHopQA 등의 데이터셋에서 평가되었으며, 평균 89.4%의 연쇄 탐지율 및 5.3%의 위양성률을 기록했습니다.

- **Performance Highlights**: CHARM을 통한 오류 전파 감소율은 82.1%에 달하며, 이는 기존 출력 수준 탐지기의 18.5%에 비해 월등한 성능입니다. 각 탐지 모듈의 기여도가 확인된 바와 같이, CHARM은 사전 오류 전파를 방지함으로써 최종 출력의 신뢰성을 제고합니다. CHARM은 또한 인간 감시 체계와 통합되어 생산적인 인공지능 배치에 필요한 신뢰성과 거버넌스 체계를 제공합니다.



### Stateful Visual Encoders for Vision-Language Models (https://arxiv.org/abs/2606.04433)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 Stateful Visual Encoder (SVE)를 도입하여 시각-언어 모델(VLM) 내의 이미지 간 상호작용을 직접 보강합니다. 기존의 VLM은 이미지가 독립적으로 인코딩되고, 이후에 언어 모델에서 비교가 이루어지는 반면, SVE는 이전 이미지의 시각적 특징을 기반으로 현재 시각적 표현을 조건화하여 효과성을 높입니다. 이로 인해, 특히 미세한 차이가 중요한 작업에서 향상된 성능을 보여 줍니다.

- **Technical Details**: SVE는 시각 인코더(f_V) 내부에 이미지를 인코딩할 때 이전 이미지의 특징을 통합하는 구조로 설계되었습니다. 이를 통해 Images 간의 조건부 상호작용을 가능케 하여, 더 정밀한 시각적 비교가 가능해집니다. 연구팀은 다양한 아키텍처 변형을 실험하여 최적의 성능을 발휘할 수 있는 디자인을 선정하였습니다.

- **Performance Highlights**: SVE를 적용한 VLM은 여러 제어된 시각 비교 작업에서 성능 향상을 보여주었고, 이를 통해 엔지니어링 전반에 걸쳐 다채로운 모델 크기와 이미지 해상도에서도 일관된 성능 향상을 달성했습니다. 임상 영상 진단, 세밀한 이미지 비교, 원격 감지와 같은 실제 작업에서도 일반 VLM 기반선 모델을 초월하거나 동등한 성능을 나타냈습니다.



### CleanCodec: Efficient and Robust Speech Tokenization via Perceptually Guided Encoding (https://arxiv.org/abs/2606.04418)
- **What's New**: 본 연구에서는 음성 처리 파이프라인에서 핵심적인 역할을 하는 Neural audio codec의 개선을 제안합니다. 기존 코덱들은 재구성 품질과 토큰 효율성 간의 균형을 잘 맞추지 못했는데, 그로 인해 배경 잡음과 녹음 아티팩트와 같은 비언어적 정보를 인코딩하면서 언어적 및 음향적으로 중요한 정보를 희생하게 됩니다. 이에 우리는 CleanCodec이라는 압축 코덱을 새롭게 제안하며, 이를 통해 감지 가능한 (perceptually) 정보만을 인코딩하고 비가시적인 정보를 제거하는 모델을 구현합니다.

- **Technical Details**: CleanCodec은 소음 제거 오디오 코덱으로, 감지 가능한 중요 특성만을 인코딩하는 방법을 학습합니다. 이를 위해 표준 오디오 재구성과 음성 향상(task) 작업을 결합하여 공동 학습 모델을 제안합니다. 이 방식은 언어 및 음색 정보를 촉진하는 조건 부여 목표(conditioning objectives)를 포함하며, 초당 12.5개의 토큰(token)으로 SOTA의 토큰화 효율성을 달성합니다. 이 코덱은 독립적으로도 동작할 수 있는 로컬 및 글로벌 인코더와 이중 디코더 아키텍처를 사용하여 구성되어 있습니다.

- **Performance Highlights**: CleanCodec은 둘 이상의 코덱에 비해 우수한 재구성 품질을 달성하고, 음성 인식 및 TTS와 같은 다운스트림 작업에서 향상된 성능을 보여줍니다. 실험 결과, CleanCodec은 최대 17배 더 빠른 추론 속도를 자랑하며, 전체 오디오 모델링에서 효율성을 크게 향상시킵니다. 이는 음성 생성을 필요로 하는 다양한 응용 분야에서의 적용 가능성을 높입니다.



### Physics-Informed Neural Network Modeling of Biodegradable Contaminant Transport through GCL/SL Composite Liners (https://arxiv.org/abs/2606.04392)
- **What's New**: 이번 연구에서는 GCL/SL 복합 라이너 시스템을 통한 오염물질 전이의 물리정보(Physics-informed) 신경망 프레임워크를 개발하였습니다. 특히 얇은 GCL 층은 정상상태의 이동-확산-생분해 공식으로 처리되며, 기본 토양 라이너는 과도한 전이 도메인으로 모델링됩니다. 두 가지 형태의 PINN(Standard-PINN 및 Hard-PINN)이 서로 다른 침출수 헤드 조건에서 평가되었습니다.

- **Technical Details**: 연구에서는 표준 PINN(Std-PINN)과 강제 구속 조건이 명시된 하드 제약 PINN(H-PINN)이 사용되었습니다. Std-PINN은 전체적인 돌파 행동(breakthrough behavior)을 잡아내지만, 초기 전이 단계에서 오류가 더 크게 나타나며 특히 높은 침출수 헤드에서의 이동 전이가 두드러지게 관찰됩니다. 반면 H-PINN은 제약 조건 집행에 따른 최적화 부담을 줄이고 보다 정확하고 안정적인 농도 예측을 제공하여 평균 절대 오차(MAE)를 현저히 낮춥니다.

- **Performance Highlights**: H-PINN은 최적화된 네트워크 구조와 tanh 활성화 함수 사용 시 가장 뛰어난 예측 정확도를 보여줍니다. 연구 결과, H-PINN은 특정 농도 관측값으로부터 SL 분해 반감기를 역 모델링하는 데 성공적으로 확장되어 신뢰할 수 있는 수렴성을 보이며 낮은 관측 잡음에서도 허용 가능한 견고성을 발휘합니다.



### Disentangling Answer Engine Optimization from Platform Growth: A Log-Based Natural Experiment on ChatGPT Referral Traffic (https://arxiv.org/abs/2606.04362)
Comments:
          9 pages, 4 figures, 1 table

- **What's New**: 최근 대형 언어 모델(LLM) 기반의 '답변 엔진'(answer engine)인 ChatGPT가 웹으로의 트래픽을 유도하고 있습니다. 이러한 현상은 '답변 엔진 최적화'(Answer Engine Optimization, AEO)라는 새로운 최적화 전략을 통해 나타나며, 이는 기존의 검색엔진 최적화(SEO)와 유사합니다. 본 연구는 고트래픽 도메인에서 AEO 기능을 검토하며, 무작위 실험이 아닌 자연 실험(natural experiment)을 통해 AEO의 효과를 분석합니다.

- **Technical Details**: 이 연구는 AEO 개입이 적용된 YouTube Q&A 페이지와 미개입 페이지를 비교하는 방법론을 사용합니다. AEO 개입은 URL 정형화, 수요 채굴, 제목 및 요약 재작성 등을 포함합니다. 분석은 주간 데이터를 활용한 구간 회귀(segmented regression)와 허위로 인한 시간 변동 테스트를 사용하여 플랫폼 성장효과를 제거할 수 있습니다.

- **Performance Highlights**: 실험 결과, ChatGPT의 추천은 5.7배 증가했으나, 미개입 페이지도 3.5배 증가했습니다. 개입 그룹에 대한 점검 결과는 1.82배의 증가를 나타냈으나, 이 효과는 유의미하진 않음(p=0.16)으로 나타났습니다. 더욱이, SEO 방어 규칙 덕분에 구글 클릭 수는 낮지 않았습니다.



### Video2LoRA: Parametric Video Internalization for Vision-Language Models (https://arxiv.org/abs/2606.04351)
- **What's New**: 비디오 이해를 위한 새로운 접근 방식을 제안하는 Video2LoRA 모델을 소개합니다. 이 방법은 비디오를 인코딩하기 위해 기존 VLM(비전-언어 모델)의 매개변수를 활용하여 비디오의 내부 표현을 저차원 적응기, 즉 LoRA(adapter)로 변환합니다. 이를 통해 쿼리 시 비주얼 토큰(visual token)을 요구하지 않고도 비디오에 대한 질문에 대답할 수 있습니다.

- **Technical Details**: Video2LoRA는 비디오를 단일 포워드 패스를 통해 LoRA 어댑터로 전환합니다. 이를 위해 VLM 인코더가 비디오를 layer-wise로 인코딩하고, 학습 가능한 Perceiver 하이퍼네트워크가 이를 매핑하여 LoRA 가중치를 생성합니다. 이 과정에서 VLM 인코더와 응답 모델은 모두 정지되며, 하이퍼네트워크만 최적화됩니다.

- **Performance Highlights**: Video2LoRA는 500M 및 2.2B 모델에서 비디오 요약 및 캡셔닝 작업을 위해 훈련되었습니다. 다양한 벤치마크에서 직접 비디오 인-컨텍스트 추론과 통계적으로 동등한 성능을 보이며, 쿼리 시간 비주얼 토큰 부담을 최대 1,500배 줄이고 응답 시간도 6-80배 감소시켰습니다. 또한, 비디오의 비주얼 정보를 퓨전하지 않고도 안정적인 결과를 유지하며, 비디오 세그먼트에 대한 독립적인 어댑터의 조합 가능성도 확인되었습니다.



### Sparse Mixture-of-Experts Reward Models Learn Interpretable and Specialized Experts for Personalized Preference Modeling (https://arxiv.org/abs/2606.04284)
- **What's New**: 이번 연구에서는 희소 Mixture-of-Experts (MoE) 보상 모델을 제안하여 인간의 다양한 선호도를 보다 명확히 반영하고자 합니다. 기존의 접근 방법들이 보편적인 보상 함수를 가정하는 경향이 있는 반면, 이 모델은 이진 선호 데이터에서 여러 선호 구성 요소를 학습해 개별 선호를 모델링합니다. 훈련 과정에서 희소한 경로 선택(sparse routing)과 전문가 다양성(expert diversity)을 촉진하여 해석 가능성과 개인화를 개선합니다.

- **Technical Details**: 희소 MoE 보상 모델은 이진 선호 데이터를 기반으로 훈련됩니다. 이 모델은 Bradley-Terry 모델을 활용하여 두 가지 반응에 대한 선호 확률을 정의하며, 주어진 프롬프트에 대한 인간의 선호도를 추정합니다. 모델의 훈련은 관찰된 선호의 음의 로그 가능성(minimizing negative log-likelihood)을 최소화하는 방식으로 진행되어, 해석 가능성과 전문성을 갖춘 전문가들이 생성됩니다.

- **Performance Highlights**: 희소 MoE 모델은 실험을 통해 해석 가능한 경로 선택 패턴과 전문화된 전문가를 학습하여 개인화에 있어 상당한 개선을 이끌어냅니다. 오직 50개의 적응 예제만으로도 개인화의 정확성이 25.81 포인트 향상되었으며, 이는 기존의 방법들보다 크게 발전된 성과입니다. 전문가의 가중치 변화는 목표 선호와의 의미론적 연관성을 잘 드러내 보여 모드 적응 과정을 검토하는 유용한 수단이 됩니다.



### Can Generalist Agents Automate Data Curation? (https://arxiv.org/abs/2606.04261)
Comments:
          Preprint

- **What's New**: 본 논문에서는 훈련 데이터를 정리하는 과정의 자동화를 위한 일반화된 코딩 에이전트의 가능성을 탐구합니다. 이를 위해 Curation-Bench라는 에이전트 중심의 벤치마크를 도입하여, 에이전트가 특정 데이터를 관찰하고 정책을 구현하며 훈련 및 평가 파이프라인에 제출하고 수정하는 등의 명령어 접근이 가능하도록 합니다. 이 연구를 통해 에이전트가 강력한 데이터 선택 기준을 신속히 도달할 수 있음을 보여줍니다.

- **Technical Details**: Curation-Bench는 모델과 훈련 레시피, 평가 스위트를 고정하여 에이전트가 데이터 inspect와 정책을 구현할 수 있도록 합니다. 에이전트는 데이터 예산의 10분의 1로 강력한 기준을 초월하는 데이터 선택 정책을 자율적으로 구성할 수 있습니다. 반복적인 정책 조정뿐만 아니라 이전 방법을 인용하고 적용하는 스캐폴드를 통해 에이전트가 보다 효과적으로 탐색할 수 있도록 유도합니다.

- **Performance Highlights**: 에이전트는 10번의 반복 내에 출판된 데이터 선택 기준에 도달하는 성과를 보였으나, 분석 결과 에이전트는 새로운 정책 패밀리보다는 로컬 정책 변형에 주로 집중함을 알 수 있었습니다. 스캐폴드가 없는 경우, 에이전트는 전략 가이드와 논문 참조를 제공받아도 새로운 접근 방식을 탐색하지 못하는 경향이 있습니다. 본 연구는 현재의 에이전트가 데이터 정리 루프를 실행할 수 있으나, 신뢰할 수 있는 데이터 연구에는 스캐폴드 방식의 방법 적응이 필요하다는 점을 강조합니다.



### StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis (https://arxiv.org/abs/2606.04246)
Comments:
          6 pages, 2 figures, DAC'2026

- **What's New**: StepPRM-RTL은 RTL 코드 생성을 위한 새로운 프레임워크로, 단계별 추론 모델링(stepwise trajectory modeling), 프로세스 리워드 모델(process-reward modeling), 그리고 Retrieval-Augmented Fine-Tuning(RAFT)을 결합하여 기능적 정확성과 추론 충실도를 향상시킵니다. 이 방법론은 RTL 코드 생성을 위한 단계를 각기 설명하는 이유(rationale)와 코드 수정이 포함된 단계별 추론 경로를 구성합니다.

- **Technical Details**: StepPRM-RTL은 각 단계에 대한 의미 있는 RTL 행동의 강도를 평가하는 Step-level Process Reward Model(StepPRM)을 도입합니다. 이 모델은 Monte Carlo Tree Search(MCTS)를 통해 다양한 추론 경로를 탐색하고, 이를 통해 학습 데이터셋을 풍부하게 만듭니다. 최종적으로 RAFT는 유사한 설계에서 캔노니컬(정형) 단계 조회를 통해 정책을 세밀하게 조정합니다.

- **Performance Highlights**: 실험 결과 StepPRM-RTL은 Verilog 및 VHDL 벤치마크 데이터셋에서 기능적 정확성과 추론 충실도에서 이전 방법론을 10% 이상 초과하는 성능을 보였습니다. Ablation 연구는 PRM 지향 보상과 단계별 추론 탐색의 조합이 성능의 핵심임을 확인했습니다. 이 프레임워크는 RTL 언어 전반에 걸쳐 일반화 가능하며 높은 충실도와 해석 가능한 코드 생성을 위한 확장 가능한 기초를 제공합니다.



### VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark (https://arxiv.org/abs/2606.04244)
- **What's New**: 본 연구에서는 VAMPS(Visual-Assisted Mathematical Problem Solving)라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 이란 대학 입학 시험의 대수학과 미적분학 문제를 바탕으로 하며, 총 1,168개의 다중 모드의 QA 쌍을 포함하고 있습니다. VAMPS의 핵심은 플롯을 사용하여 시각적 추론을 평가하는 유일한 페르시아-영어 벤치마크라는 점에서 중요합니다.

- **Technical Details**: VAMPS는 주어진 문제를 정보가 풍부한 플롯으로 변환하고, 결과적으로 생성된 플롯을 통해 최종 결정을 내릴 수 있는 모델의 능력을 평가합니다. 이 과정에서 Desmos라는 그래프 도구를 사용하여 시각적 표현을 생성하며, 모델의 도구 사용 행동 및 최종 답변의 정확성을 면밀히 분석할 수 있습니다. 벤치마크는 텍스트 기반 모델과 비교하여 도구 이용의 효과를 명확하게 이해하려고 합니다.

- **Performance Highlights**: 연구 결과, 다양한 모델에서 직접적인 분석 해결이 도구 사용을 통한 시각 해결보다 뛰어난 성능을 보이는 것으로 나타났습니다. 이는 많은 문제에서 플롯 생성이 유용한 전략임에도 불구하고, 모델이 도구를 통해 얻는 시각적 증거를 통합하는 데 어려움이 있다는 것을 시사합니다. VAMPS는 현재 모델들에서 이러한 reasoning-to-perception (추론-지각) 전환이 여전히 병목 현상으로 작용하는지를 테스트하기 위해 설계되었습니다.



### Overview of the EReL@MIR 2025 Multimodal Document Retrieval Challenge (Track 1) (https://arxiv.org/abs/2606.04240)
Comments:
          MDR Challenge Report at WWW2025

- **What's New**: 이번 논문은 2025년 Web Conference와 공동으로 개최된 첫 EReL@MIR 워크샵에서 진행된 Multimodal Document Retrieval Challenge의 트랙 1에 대해 소개합니다. 이 챌린지는 하나의 시스템을 통해 긴 문서 내에서의 closed-set retrieval과 이미지 또는 이미지-텍스트 쿼리를 통한 open-domain retrieval을 처리해야 하는 두 가지 작업으로 구성됩니다. 455명의 참가자와 586개의 제출물이 있었으며, 시스템들은 평균 Recall@{1,3,5}에 따라 순위가 매겨졌습니다.

- **Technical Details**: 챌린지는 두 가지 작업을 위해 설계되었으며, Task 1인 MMDocIR은 텍스트 쿼리에 따라 단일 긴 문서의 관련 페이지를 순위 매기는 것이고, Task 2인 M2KR은 이미지나 이미지-텍스트 쿼리를 가진 단일 global corpus에서 관련 위키피디아 스타일의 구문을 검색하는 것입니다. 모든 시스템은 Qwen2-VL 계열의 decoder 기반 멀티모달 LLM embedder를 사용하였으며, 우리가 알아본 세 가지 우승 시스템은 서로 다른 방식으로 성능을 높였습니다.

- **Performance Highlights**: 이번 챌린지에서 우승한 세 팀의 시스템은 fine-tuned ensemble, 강력한 vision-language re-ranker를 이용한 training-free multi-route fusion, 또는 zero-shot late interaction을 통해 최고의 성과를 올렸습니다. 특히, training-free 시스템은 fine-tuned 승자와 0.1점 차이로 마무리되었습니다. 이 결과들은 멀티모달 정보 검색의 발전을 위한 중요한 교훈을 제공합니다.



### DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities (https://arxiv.org/abs/2606.04205)
- **What's New**: DetectZoo는 AI 생성 콘텐츠 탐지를 위한 최초의 다기능 툴킷으로, 텍스트, 이미지 및 오디오 모달리티에 대한 통합 인터페이스를 제공합니다. 이 툴킷은 데이터 수집 및 전처리부터 모델 평가에 이르는 전체 실증적 파이프라인을 표준화하여 연구자들이 최신 탐지기를 체계적으로 벤치마킹할 수 있게 합니다. 61개의 탐지기를 위한 참조 구현, 22개의 벤치마크 데이터셋을 위한 네이티브 로더 및 공통 인터페이스를 통해 여러 메트릭을 보고하는 표준화된 평가 파이프라인도 포함되어 있습니다.

- **Technical Details**: DetectZoo는 모달리티 간의 비교를 용이하게 하기 위해 61개의 탐지 방법을 단일 코드베이스에 통합하고 22개의 벤치마크 데이터셋 및 표준화된 평가 파이프라인과 함께 제공됩니다. 통합된 모달리티의 접근 방식을 통해 연구자들은 반복 가능한 비교를 실시할 수 있으며, 이는 AI 생성 콘텐츠 탐지의 효과적인 연구 환경을 조성합니다. 각 탐지기는 독립적이며 동일한 인터페이스를 통해 접근 가능하며, 사전 학습된 가중치를 자동으로 캐시하여 원래 발표된 결과를 재현합니다.

- **Performance Highlights**: DetectZoo는 다중 모달 AI 포렌식의 진입 장벽을 낮춰 연구자들이 도메인 간 성능 차이를 식별하고 강력하고 일반화 가능한 탐지 기법의 개발을 가속화할 수 있도록 합니다. 이는 AI 생성 콘텐츠 탐지의 연구 성과를 더욱 효과적으로 발전시키고, 개별 논문 코드베이스에 대한 의존도를 줄이는데 기여합니다. 모든 구성 요소는 공개적으로 사용 가능하며, pip를 통해 쉽게 설치할 수 있어 연구자들이 접근할 수 있는 유용한 리소스가 됩니다.



### Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions (https://arxiv.org/abs/2606.04197)
Comments:
          Submitted to the Journal of Artificial Societies and Social Simulation (JASSS)

- **What's New**: 이 연구는 LLM(대형 언어 모델) 에이전트가 메모리 깊이와 네트워크 구조가 합의 형성에 미치는 상호작용을 파악하는 데 중점을 두고 있습니다. 연구 결과에 따르면, 메모리가 길어질수록 분산 네트워크에서는 안정 상태에 도달하는 시간이 느려지지만 중앙 집중적인 네트워크에서는 빨라진다는 것이 밝혀졌습니다. 이는 서로 다른 네트워크 구조에서 메모리의 영향이 다르게 나타난다는 점을 강조합니다.

- **Technical Details**: 이 모델은 고정된 소셜 네트워크에서 LLM 에이전트 간의 'Naming Game'을 시뮬레이션하여 분석되었습니다. 에이전트는 마지막 M∈{2,5,10} 상호작용을 기억하며, 수렴 균형을 선택하는 과정에서 로컬 상호작용을 통해 집단적으로 하나의 관습으로 수렴할 수 있는지를 연구합니다. 각 라운드에서 두 에이전트가 선택한 규칙이 일치할 경우 성공적으로 상호작용하여 점수를 얻습니다.

- **Performance Highlights**: 연구의 결과, 중앙 집중형 네트워크에서의 빠른 정착 속도는 조각난 합의에 고착되기 쉽다는 것을 나타냅니다. 메모리를 통한 빠른 조정과 협력은 분산 네트워크에서의 일관된 규약을 저해할 수 있으며, 이는 LLM 에이전트 populations의 행동을 이해하는 데 기여합니다. 최종적으로, 메모리 깊이와 통신 구조는 함께 설계되어야 하며, 단독으로 최적화하는 것이 아니라 두 요소 간의 관계를 고려해야 한다는 실용적인 시사점을 제공합니다.



### Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieva (https://arxiv.org/abs/2606.04194)
Comments:
          9 pages, 3 figures, 10 tables. Code, data, and per-table receipts: this https URL

- **What's New**: 이 논문은 다양한 대화 세션의 역사에서 새로운 질문에 대한 답변을 포함한 과거 발화(turn)를 효과적으로 검색하는 방법론을 제시합니다. 특히, 이전 연구들과 달리, 정보 검색에서 훈련이 필요 없는 CPU 전용의 검색이 대화 메모리 구조에서 어떤 이점을 가질 수 있는지를 탐구하고 있습니다. 연구 결과는 대화 기억 회수의 효율성을 향상시키기 위한 다양한 실험적 통찰을 제공합니다.

- **Technical Details**: 연구는 비교 점수(score-level fusion) 기법을 사용하여 late-interaction과 BM25 방식의 결합을 통한 효율성을 강조합니다. 특히, late-interaction이 특정 쿼리에 대해 성능이 높다는 것을 입증하고, BM25과 결합 시 성능 향상이 이루어지는 것을 보여줍니다. 실험에서는 다양한 인코더(encoder)를 사용하여 이 결합의 효과를 정량적으로 평가합니다.

- **Performance Highlights**: 결과적으로, late-interaction과 BM25 점수의 융합을 통해 LoCoMo Hit@1이 8.8에서 17.2 포인트 향상된 결과가 나타났습니다. 하지만, 최상위 10개 결과에 대한 재정렬(reranking)의 적용이 Hit@1을 6.9 포인트 감소시키는 반효과를 일으키기도 하였습니다. 각각의 인코더에 대한 성능 차이를 분석한 결과, 딴 유형의 질문에 따라 다양한 성능 차이를 보였습니다.



### SocialCoach: Personalized Social Skill Learning with RL-based Agentic Tutoring and Practic (https://arxiv.org/abs/2606.04155)
- **What's New**: 이 논문에서는 SocialCoach라는 개인화된 사회적 기술 개발을 위한 LLM(대형 언어 모델) 기반 튜터링 시스템을 소개합니다. 이 시스템은 다양한 전문가 소스에서 이론과 실제를 통합한 지식 코퍼스를 자동으로 구축하며, 개인의 학습 경로를 개인 맞춤형으로 제공하는 적응형 연습 스케줄링 모듈을 사용합니다. 또한, 환경 시뮬레이션을 통한 강화 학습 최적화를 통해 장기적인 학습 경험을 극대화하는 방법을 제시합니다.

- **Technical Details**: SocialCoach는 사회적 기술 지식을 체계화하고, 다중 에이전트 파이프라인을 활용하여 전문가 입력을 자동으로 구조화하여 교육의 질을 높입니다. 이 시스템은 사용자 시뮬레이터를 통해 사용자의 학습 진행에 맞게 변동적인 연습 시나리오를 제시하며, 또한 메타인지적 스캐폴딩과 진단 평가를 결합하여 깊이 있는 학습을 촉진합니다. 기술적인 핵심 요소로는 조건부 생성 태스크로서의 연습 커리큘럼 스케줄링과 시뮬레이션 기반 강화 학습 최적화가 포함됩니다.

- **Performance Highlights**: SocialCoach는 기존 방법들에 비해 시뮬레이션된 경로의 질 및 평가자가 측정한 튜터링 품질에서 개선된 결과를 보여주었습니다. 초기 사용자 피드백은 강력한 참여도와 유용성을 나타내며, 이는 개인화되고 게이미피케이션된 교육 플랫폼의 실용적인 아키텍처로서의 가능성을 시사합니다. 이러한 결과들은 소프트 스킬 학습 분야에서 스마트한 교육 도구의 발전에 기여할 것으로 기대됩니다.



### Large Language Models Hack Rewards, and Society (https://arxiv.org/abs/2606.04075)
Comments:
          14 pages, 9 figures, 7 tables

- **What's New**: 이 논문에서는 강화학습(Reinforcement Learning, RL)이 대형 언어 모델(Large Language Models, LLMs)에서 보상(reward) 학습을 이끄는 방법을 설명합니다. 연구자들은 사회 규제가 보상 함수와 구조적으로 유사하다는 점을 주장하고, LLM이 이러한 규제에서 발생하는 허점을 이용할 가능성을 제시합니다. 새로운 사회적 해킹(societal hacking) 개념이 도입되어, RL 훈련 과정에서 모델이 사회적 규칙을 조작할 수 있음을 잠재적으로 탐구합니다.

- **Technical Details**: 논문에서는 SocioHack이라는 72개의 사회적 환경을 제공하여 모델이 사회 규칙을 해킹할 수 있는 가능성을 연구합니다. 각 환경은 자연어로 규제(specification)를 정의하는 튜플로 구조화되어 있으며, 이를 통해 RL 훈련이 어떻게 사회 구조를 탐색하고 조작하는지를 분석합니다. 훈련 과정에서 모델은 주어진 프롬프트에 기반하여 전략을 생성하며, 이 과정은 최적화의 불확실성을 조절하여 비효율적인 탐색을 방지합니다.

- **Performance Highlights**: 실험 결과, RL이 손상된 규칙을 61.25%의 회수율(recall)과 90.85%의 정밀도(precision)로 재발견할 수 있음을 보여줍니다. 그러나 현재의 안전 장치들은 여전히 불완전하며, 모델이 해로운 지시를 받을 때만 반응합니다. 이는 사회적 환경에서 최적화 압력이 지속됨에 따라 모델과 규제 간의 지속적인 상호 작용 및 진화가 발생할 수 있음을 시사합니다.



### Covert Influence Between Language Models (https://arxiv.org/abs/2606.04071)
- **What's New**: 이번 논문은 언어 모델 간의 상호작용에서 발생할 수 있는 covert influence(밀착 영향력)라는 새로운 위험을 다룹니다. 이는 특정 모델이 생성한 사용자의 너무 뚜렷하지 않은 신호가 다른 모델의 행동에 미치는 영향을 연구한 것입니다. 세 가지 인터페이스, 즉 supervised fine-tuning (SFT), on-policy distillation (OPD), in-context learning (ICL)을 통해 이 위험을 분석하며, 이전 연구에서는 미처 발견하지 못했던 경우도 포함됩니다.

- **Technical Details**: covert influence의 매커니즘을 조사하기 위해, 저자들은 payload(전달 내용)가 숨겨지고, 인간이 인지할 수 없는 방식으로 다른 모델에 전파될 수 있는 방식들을 분류했습니다. 이를 통해 각 인터페이스에서에서도 이 영향력이 어떻게 발생하는지를 점검하고, Math 형식의 MDCL(mean difference in conditional log-probabilities) 스코어를 사용하여 결국 어떤 carrier(매개체)가 효과적인지를 평가했습니다.

- **Performance Highlights**: 연구에서 밝혀진 바에 따르면, SFT와 OPD에서도 covert influence가 효과적으로 발생할 수 있으며, 특히 MDCL로 선택된 carrier들은 이전에는 도달할 수 없었던 payload 전송을 가능하게 합니다. 따라서 이러한 covert influence가 인식되는 것보다 훨씬 더 광범위한 위협을 가지고 있다는 결론을 내렸습니다. 특히, 자연어 carrier의 경우 인간이 이 영향을 어느 정도 인지할 가능성이 있지만, 숫자 carrier는 그러한 인지가 어려운 것으로 나타났습니다.



### Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation (https://arxiv.org/abs/2606.04046)
Comments:
          Accepted at ICML 2026

- **What's New**: 본 논문에서는 SceneDiver라는 새로운 방법을 제안하여, 시각-언어 의사결정(vision-language decision making) 과제에서의 인식 한계(perceptual limitation)를 극복하고자 합니다. 기존의 시각-언어 모델(VLMs)과 시각-언어-행동 모델(VLAs)이 각각의 장점을 갖고 있지만, 시각적 환각(visual hallucinations) 문제로 인해 성능 제한을 겪고 있습니다. SceneDiver는 장기 계획 능력을 활용하여, 먼저 전체 장면 그래프(scene graph)를 구축하고 이를 통해 작업을 간단한 하위 문제로 분해하는 방식으로, 효과적으로 중요한 객체에만 집중할 수 있도록 설계되었습니다.

- **Technical Details**: SceneDiver의 중심은 거친 단계에서 세밀한 단계로 진행되는 초점(focus) 계획 수립입니다. 첫 번째 단계에서는 이미지 데이터를 구조화된 그래프 표현으로 변환하여 장면을 전반적으로 이해합니다. 두 번째 단계에서는 VLM이 각 지역 하위 장면을 탐색하여 중요한 객체를 식별하도록 합니다. 또한, 실시간 의사결정에 필요한 지연 시간을 충족하기 위해 가벼운 어댑터(adapter)를 설계하여 VLA 모델에서 효과적인 초점 능력을 추출합니다.

- **Performance Highlights**: 다양한 로봇 조작 및 방 탐색 과제를 통해 실험한 결과, SceneDiver는 조작 작업에서 10%-15%, 탐색 작업에서 최대 16%의 성능 향상을 보여주었습니다. 또한, LIBERO-plus 벤치마크에서 성공률이 9.6% 개선되었으며, 이는 의사결정의 강건성을 향상시키는데 기여했습니다. 이 모든 성능 향상과 함께 계산 효율성도 유지되었으며, 실시간 배포에 적합합니다.



### Do Transformers Need Three Projections? Systematic Study of QKV Variants (https://arxiv.org/abs/2606.04032)
Comments:
          Accepted at ICML 2026 (PMLR vol. 306). 26 pages, 12 figures, 16 tables. Code: this https URL

- **What's New**: 이번 논문에서는 Transformer 모델에서 쿼리, 키, 값 (QKV) 어텐션의 효용성을 재조명하고, 서로 다른 프로젝션 공유 제약을 통하여 결과를 비교하고 있습니다. 세 가지 제약 구조인 Q=K-V (통합된 쿼리-키, 분리된 값), Q-K=V (분리된 쿼리, 통합된 키-값), 그리고 Q=K=V (모든 프로젝션 통합)를 평가하고 있습니다. 이 연구는 QKV의 필요성에 대한 질문을 던지며, 프로젝션 공유의 효과를 정량적으로 분석한 것입니다.

- **Technical Details**: 연구에서 제안된 세 가지 프로젝션 공유 제약을 통해, 프로젝션 매트릭스 수를 줄이고 파라미터 수와 계산 오버헤드를 현저히 감소시킬 수 있음을 보여줍니다. 올바른 비대칭성을 유지할 경우, 상대적인 성능 손실을 최소화하면서도 자원 효율성을 높일 수 있습니다. 또한, 기존의 GQA (Grouped Query Attention)나 MQA (Multi-Query Attention)와는 달리, 프로젝션 매트릭스를 공유함으로써 메모리 효율성과 처리량을 동시에 최적화할 수 있습니다.

- **Performance Highlights**: 프로젝트 공유를 통해 Q-K=V는 KV 캐시 용량을 50% 줄이면서도 300M 파라미터 모델에서 오직 3.1%의 퍼플렉시티 증가를 기록했습니다. 1.2B 파라미터 모델에서도 비슷한 경향을 보여주며, 대규모 모델에서도 품질이 안정적으로 유지됨을 확인했습니다. 또한, Q-K=V와 헤드 공유를 결합함으로써, 메모리 효율성을 더욱 강화할 수 있는 가능성을 제시하였습니다.



### P$^2$-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization (https://arxiv.org/abs/2606.03376)
- **What's New**: 본 연구는 Perceptual Processing Direct Preference Optimization (P$^2$-DPO)이라는 새로운 훈련 패러다임을 제안합니다. 이 방법은 모델이 자신의 선호 쌍을 생성하고 학습함으로써 인식 병목 현상을 직접 해결합니다. 기존의 비전 무관 선호 쌍의 한계를 극복하며, 데이터의 비효율성을 줄이기 위한 방법론이 추가되었습니다.

- **Technical Details**: P$^2$-DPO는 두 가지 새로운 유형의 선호 쌍을 도입합니다: (1) Focus-and-Enhance Preference Pair는 인식 병목 현상을 해결하기 위해 이미지의 세밀한 디테일에서 개선된 출력과 저하된 출력을 대조하며, (2) Visual Robustness Preference Pair는 깨끗한 신호와 노이즈가 있는 신호의 출력을 대조하여 시각적 견고성을 강화합니다. 이 과정에는 동적 가중치 조정 및 보정 손실(Calibration Loss)도 포함되어 효율적인 학습 루프를 생성합니다.

- **Performance Highlights**: 실험 결과는 P$^2$-DPO가 강력한 기준 모델들을 초과하며, 비용이 많이 드는 인간 피드백 없이도 비교 가능한 훈련 데이터 및 비용으로 성과를 달성함을 보여줍니다. 또한, Attention Region Fidelity (ARF) 및 이미지 저하 시나리오에 대한 평가를 통해 P$^2$-DPO가 인식 병목 현상을 해결하고 시각적 견고성을 향상시키는 데 효과적임을 입증했습니다.



### Not What, But How: A Framework for Auditing LLM Responses across Positioning, Generalization, Anthromorphism, and Maxims (https://arxiv.org/abs/2606.02493)
Comments:
          34 pages, 19 Figures, 4 Tables

- **What's New**: 이번 논문은 주관적인 질문에 대한 언어 모델의 응답 방식이 어떻게 프레이밍(Framing)되는지를 통한 새로운 평가 방법론인 FRANZ를 소개합니다. 기존의 대규모 언어 모델(LLMs) 평가가 주로 사실적 정확성(factual correctness)에 중점을 두었으나, FRANZ는 문화적 지위, 일반화 언어의 사용, 의인화(anthropomorphism) 신호 및 대화 격률(Conversational maxims)을 포함한 네 가지 차원에서 응답을 분석합니다. 이를 위해 57개의 서브레딧에서 376,000개의 주관적인 질문으로 구성된 SQUARE 코퍼스를 기여하였습니다.

- **Technical Details**: FRANZ는 응답을 문화적 관점에서 분석하는 프레임워크로, Llama, Gemma, Mistral 등 세 가지 다양한 LLM의 응답을 평가합니다. 각 모델은 서로 다른 경향성을 보이며, 예를 들어 Mistral은 강한 내부 지향성과 일반화 언어를 사용하는 경향이 두드러집니다. 본 연구는 응답 성격이 모델 간에 어떻게 다르게 나타나는지를 공통적으로 평가하여, 더 정교한 피드백을 가능하게 합니다.

- **Performance Highlights**: FRANZ를 사용한 분석 결과, 언어 모델 간의 응답 차이에서 통계적으로 유의미한 차이가 발견되었습니다. 특히, Mistral 모델이 내부 지향적인 응답을 더 자주 사용하고, Gemma는 대화 격률을 가장 잘 준수하는 경향이 있었습니다. 이러한 성과는 LLM의 응답 프레이밍에 대한 이해를 넓히고, 사용자 경험이 모델에 따라 다를 수 있음을 시사합니다.



New uploads on arXiv(cs.IR)

### SearchLog: A Web Browser Extension for Capturing Search Logs in Laboratory Studies (https://arxiv.org/abs/2606.05040)
- **What's New**: 검색 행동을 연구하는 데 유용한 자연 검색 로그를 수집할 수 있는 웹 브라우저 확장 프로그램, SearchLog을 소개합니다. 이 도구는 실험 연구 중에 참가자들이 자연스럽게 검색할 수 있도록 하며, 마우스 클릭, 키보드 입력, 검색 활동 등을 기록합니다. SearchLog은 Google과 Microsoft Bing을 포함한 두 개 주요 검색 엔진에 대한 AI 생성 요약도 감지합니다.

- **Technical Details**: SearchLog은 Chromium 기반 브라우저에서 쉽게 설치 가능한 툴킷입니다. 이는 클라이언트와 서버 구성 요소로 나뉘며, 클라이언트는 사용자의 상호작용을 기록하는 확장 프로그램이고, 서버는 세션을 추적하고 기록된 데이터를 저장하는 역할을 합니다. 기록된 데이터는 페이지 수준과 브라우저 수준의 정보로 나뉘며, 이 정보는 웹 페이지와의 상호작용, 브라우저 창 및 탭 관련 작업을 포함합니다.

- **Performance Highlights**: SearchLog은 실험 참가자가 자연 검색 환경에서 상호작용할 수 있도록 지원합니다. 이를 통해 연구자들은 사용자 행동을 더 풍부하게 분석할 수 있습니다. 기록된 로그는 기존의 로그 도구에 비해 더 많은 이벤트를 캡처할 수 있으며, 사용자 세션 및 태스크 메타데이터와 연결되어 있습니다.



### Dual-Stream MLP is All You Need for CTR Prediction (https://arxiv.org/abs/2606.04944)
Comments:
          Accepted by TKDD

- **What's New**: 이 논문에서는 Click-through rate (CTR) 예측을 위한 새로운 프레임워크인 Dual-Stream MLP (DS-MLP)를 제안합니다. DS-MLP는 명시적(feature interaction) 및 암시적(implicit) 상호작용을 효과적으로 학습하기 위해 지식 증류(knowledge distillation)를 활용하며, 복잡한 아키텍처 대신 단순하고 효율적인 방식을 사용합니다. 특히, DS-MLP는 성능상의 한계를 극복하면서도 기존의 고급 모델에 필적하는 결과를 보여줍니다.

- **Technical Details**: CTR 예측 문제는 사용자가 항목을 클릭할 확률을 추정하는 이진 분류 문제로 설정되며, 다양한 컨텍스트 특징이 포함된 입력-label 쌍을 기반으로 합니다. DS-MLP는 메인 MLP와 보조 MLP의 두 개의 구성 요소로 구성되어 있으며, 지식 증류를 통해 메인 MLP는 명시적 상호작용을 학습하고, 보조 MLP는 암시적 상호작용을 포착합니다. 이러한 아키텍처는 두 MLP 구성 요소 간의 정렬 전략을 포함하여 모델의 호환성을 향상시킵니다.

- **Performance Highlights**: DS-MLP는 세 가지 널리 사용되는 벤치마크에서 신 state-of-the-art 성능을 달성하여 기존의 모든 CTR 모델을 지속적으로 초월합니다. 또한, 대규모 데이터셋을 처리할 때 낮은 지연 시간을 보이며, 실제 애플리케이션 시나리오에서 높은 확장성을 제공합니다. 이는 DS-MLP가 단순한 구조임에도 불구하고 매우 강력한 성능을 발휘할 수 있음을 보여줍니다.



### BEATS: Bootstrapping E-commerce Attribute Taxonomies for Search through Iterative Human-AI Collaboration (https://arxiv.org/abs/2606.04909)
Comments:
          6 pages, 1 figure, 5 tables. Accepted to SIGIR 2026 Industry Track. Official version: this https URL

- **What's New**: 이 논문에서는 BEATS(Bootstrapping E-commerce Attribute Taxonomies for Search)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 신흥 시장의 E-commerce 플랫폼에서 결측된 속성 스키마(attribute schemas)를 개발하기 위해 다단계 LLM 생성 파이프라인을 사용합니다. 이는 제품 속성의 질을 확보하기 위해 모델 개발자와 도메인 전문 인력이 반복적으로 협력하는 '인간 포함' 접근 방식을 사용합니다.

- **Technical Details**: BEATS는 다섯 단계의 반복적 프로세스를 통해 속성 세트를 작성합니다. 첫 번째 단계에서는 여러 LLM이 후보 속성을 생성하고, 두 번째 단계에서는 이 후보들을 종합하여 통일된 속성 집합을 만듭니다. 이후 세 번째 단계에서는 노이즈를 제거하고 카테고리 계층과의 일관성을 보장합니다. 마지막 두 단계에서는 품질 검토와 인적 주석을 통해 생성된 속성을 검증합니다.

- **Performance Highlights**: BEATS 프레임워크를 사용하여 구조화된 데이터 세트를 생성한 후, 속성이 풍부해진 제품 데이터로 조밀 검색 모델을 훈련하여 일관된 개선을 입증했습니다. 이 시스템은 라쿠텐 대만에 배포되어 9개 주요 카테고리를 다루며, 67,277개의 속성과 540만 개 이상의 제품에 태그를 추가했습니다. 향후 전체 제품 카탈로그를 풍부하게 하는 계획이 있습니다.



### EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking (https://arxiv.org/abs/2606.04727)
- **What's New**: 본 논문은 LLM 기반 추천 시스템의 신뢰성을 향상시키기 위해 EviRank라는 새로운 방법을 제안합니다. EviRank는 한 번의 전방 패스에서 세 가지 보완적인 증거를 추출하고 이를 신뢰할 수 있는 의견 집계를 통해 집계합니다. 또한, 순위 위치에 따라 중요성을 반영하여 위치 인식 보정을 도입하고, 이를 통해 더 신뢰할 수 있는 추천을 제공합니다.

- **Technical Details**: EviRank는 사용자와 항목 간의 상호작용 역사, 항목의 텍스트 표현 등 사용자 및 항목 집합을 사용하여 순위 목록을 생성합니다. 모델의 의사결정 과정에서 믿을만한 자신감을 반영하기 위해 세 가지 종류('semantic', 'attention', 'output')의 증거를 추출합니다. 이러한 증거를 융합하여 robust한 신뢰도 추정치를 산출하며, 다중 샘플링 방법의 계산 오버헤드를 피할 수 있습니다.

- **Performance Highlights**: 세 가지 공개 실제 데이터셋에서 EviRank의 실험을 수행한 결과, 추천 및 불확실성 정량화에서 최첨단 성능을 달성함을 검증했습니다. 이 방법은 LLM 기반 추천 시스템에서의 신뢰성 확인을 극대화하는 데 기여하며, 기존 방법들에 비해 우수한 성능을 보입니다.



### Improving the Efficiency and Effectiveness of LLM Knowledge Distillation for Conversational Search (https://arxiv.org/abs/2606.04650)
Comments:
          SCAI Workshop at SIGIR '26}{July 20--24, 2026}{Melbourne, Naarm, Australia

- **What's New**: 이번 연구에서는 대화 검색(Conversational Search, CS) 분야에서 Kullback-Leibler Divergence (KLD) 기반의 지식을 증류하는 방법론을 조사합니다. 연구팀은 KLD 손실 함수에 대조 손실(contrastive loss)을 추가하여 성능을 개선할 수 있음을 발견했습니다. 이와 더불어, 샘플의 수가 KLD 손실에 미치는 영향을 살펴보았고, 긴 대화에서 희소성(sparsity) 및 추론 효율성(inference efficiency)을 증가시킬 수 있는 방법을 제안합니다.

- **Technical Details**: 연구에서 제안된 방법론은 KLD 손실과 대비되는 정보 손실(contrastive InfoNCE loss)을 조화롭게 결합하는 것입니다. 이 방식은 KLD 손실을 통해 저장된 정보를 활용하여 유사성 점수(similarity scores)가 긍정적 샘플과 부정적 샘플에 균형 있게 반영되도록 합니다. 특히, 큰 수의 긍정 샘플을 사용함으로써 학습의 신뢰성을 높이고 모델의 성능이 얼마나 개선되는지에 대한 탐색이 필요합니다.

- **Performance Highlights**: 실험 결과를 통해 KLD 손실을 사용한 모델이 2배의 연산량 감소(FLOPS reduction)와 함께 약 2%의 Recall@100 성능 저하만을 보이며, 긴 대화 쿼리에 대해서도 효율성을 높일 수 있음을 확인하였습니다. 이러한 결과는 희소한 검색 방법에 대한 효과적인 지침을 제공하며, 모델의 효과성과 효율성을 동시에 개선할 수 있는 새로운 방향을 제시합니다.



### Distributional Approximate Nearest Neighbour Search for Uncertainty-Aware Retrieva (https://arxiv.org/abs/2606.04603)
- **What's New**: 이번 논문에서는 DINOSAUR(Distributional Approximate Nearest Neighbour Search for Uncertainty-Aware Retrieval)라는 새로운 프레임워크를 제안하여 임베딩의 불확실성을 기반으로 추천 후보생성을 개선합니다. 기존의 추천 시스템에서는 사용자와 아이템 각각에 대해 단일 포인트 추정 임베딩을 학습하여, 이는 노이즈가 존재하고 여전히 관련성을 충분히 포착하지 못합니다. 이를 통해 불확실성을 고려한 검색을 통합하여, 더욱 다양한 콘텐츠를 추천할 수 있도록 합니다.

- **Technical Details**: DINOSAUR는 아이템 당 여러 개의 샘플 임베딩($S_i$)을 생성하고 이를 기반으로 인덱스를 구성하여 검색 과정에서 두 가지 측면의 확률적(retrieval) 수집을 가능하게 합니다. 이 시스템은 새로운 모델 아키텍처나 ANN 인덱스 구조를 요하지 않으며, 불확실성 마진을 통해 사용자 임베딩을 샘플링하여 사용자 쿼리를 수행합니다. 이를 통해 불확실성을 효과적으로 관리하고, 아이템 임베딩의 변동성이 어떻게 잠재 공간에서의 검색 범위를 확장하는지 분석합니다.

- **Performance Highlights**: DINOSAUR는 불확실성이 사라질 때 표준 포인트 추정 검색을 회복할 수 있으며, 불확실한 아이템이 검색될 수 있는 잠재 공간의 영역을 확장합니다. 실험 결과, 제한된 오프라인 리콜 손실로 큰 범위의 커버리지 향상을 보여줍니다. 이로 인해 인기 아이템뿐만 아니라 다양한 니치 콘텐츠도 효과적으로 추천할 수 있게 됩니다.



### Trading Engagement for Sustainability: Carbon-Aware Re-ranking for E-commerce Recommendations (https://arxiv.org/abs/2606.04550)
Comments:
          23 pages, 30 figures. Code available at this https URL

- **What's New**: 이번 연구는 대부분의 전자상거래 제품에 대한 탄소 발자국(Product Carbon Footprint, PCF) 데이터가 결여된 현실을 고려하여, 이를 추정하는 새로운 접근 방식을 제시합니다. 연구진은 Carbon Catalogue로부터 감독을 전이하여 라벨이 없는 대규모 제품 카탈로그를 대상으로 탄소 발자국을 추정하는 파이프라인을 개발하였습니다. 또한, 기존 추천 모델인 BPR, NeuMF, LightGCN에서 생성된 relevance score를 사용하여 탄소를 고려한 재순위 전략을 적용하였습니다. 이는 사용자와 아이템 간의 상호작용에 대한 분석을 통해 지속 가능성을 높이는 방안을 모색하는 데 기여합니다.

- **Technical Details**: 이 연구에서 제안하는 파이프라인은 semantic similarity search와 few-shot LLM prompting을 활용하여 PCF 값을 추정합니다. 이후에는 모델의 relevance score와 추정된 탄소 발자국 간의 트레이드오프를 발생시키는 조정 가능한 매개변수인 lambda를 통해 재순위 전략을 시행합니다. 연구진은 Amazon Reviews 데이터셋을 사용하여 홈 및 주방, 스포츠 및 야외, 전자 기기 등 3개 제품 범주에서 평가를 진행하였습니다. 이 과정에서 관찰된 사용자 피드백을 통해 모델의 성능과 탄소 발자국의 균형을 분석합니다.

- **Performance Highlights**: 실험 결과, 모든 모델과 카테고리에서 상당한 탄소 배출 감소가 가능함을 보여주었으며, 이는 사용자 참여에 대한 최소한의 비용으로 달성되었습니다. 연구진은 lambda 값을 변화시켜 engagement와 탄소 간의 Pareto frontier를 구성하여 지속 가능성 목표와 추천 성능 간의 상호작용을 명확하게 밝혔습니다. 또한, 추천 알고리즘에 따라 각 모델과 카테고리 간의 engagement 및 지속 가능성 트레이드오프의 다양성을 강조하여, 이는 모델 선택과 도메인 맥락의 중요성을 시사합니다.



### Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization (https://arxiv.org/abs/2606.04547)
Comments:
          16 pages, 6 figures

- **What's New**: 논문에서는 LLM(대형 언어 모델)의 개인화를 위한 새로운 방법인 TAP-PER(Temporal Attentive Prefix for PERsonalization)를 제안합니다. TAP-PER는 사용자 선호도를 학습 가능한 표현으로 인코딩하여 명시적인 프롬프트 구성을 제거하고 사용자 상태 프리픽스 임베딩으로 대체합니다. 이 방식은 사용자의 지속적인 관심을 포착하기 위해 시간적 신호를 통합하여 사용자 모델링을 더 효과적으로 수행합니다.

- **Technical Details**: TAP-PER는 두 단계의 학습 패러다임을 따릅니다: 첫 번째 단계는 작업 적응(Task Adaptation)으로, 모든 사용자로부터 집계된 데이터를 사용해 기본 모델을 조정합니다. 두 번째 단계에서는 사용자별 프리픽스를 학습하여 일반 작업 능력을 유지하면서 사용자 선호도 모델링에 집중합니다. 이 과정에서 LoRA(Low-Rank Adaptation) 기술이 사용되어 효율적인 파라미터 업데이트가 이루어집니다.

- **Performance Highlights**: TAP-PER는 6개의 LaMP 작업에서 프롬프트 기반 및 모델 기반 벤치마크를 지속적으로 초과하는 성능을 보였습니다. TAP-PER는 OPPU보다 사용자 당 파라미터를 130배 줄이고, PER-PCS의 전체 파라미터 발자국을 절반으로 줄이는 등의 뛰어난 효율성을 나타냈습니다. 그 결과, TAP-PER는 명시적인 프롬프트 구축이나 무거운 사용자별 어댑터 없이 확장 가능한 LLM 개인화를 가능하게 합니다.



### ANN Search: Recall What Matters (https://arxiv.org/abs/2606.04522)
- **What's New**: 이번 논문에서는 Approximate Nearest Neighbor (ANN) 검색의 질을 평가하는 새로운 메트릭인 1/Ratio@k를 탐구합니다. 기존의 Recall@k는 정확하지 않은 기준으로 작업의 난이도를 과장하고 있으며, 이로 인해 불필요한 계산 비용이 증가하고 있습니다. 반면, 1/Ratio@k는 검색된 결과의 질을 더 정확히 반영하며, 컴퓨팅 자원 소모가 적고 적용이 용이합니다.

- **Technical Details**: ANN 검색은 정보 검색, 추천 시스템, 검색 강화 생성(retrieval-augmented generation) 등 다양한 분야에서 핵심적으로 사용됩니다. 기존의 Recall@k 기준 대신, 1/Ratio@k는 검색된 이웃과 진짜 이웃의 거리 차이를 측정하여 더 효과적인 품질 측정을 제공합니다. 두 메트릭 모두 0과 1 사이의 값을 가지지만, Recall@k는 식별자 일치를 세는 반면, 1/Ratio@k는 검색된 결과의 질을 측정합니다.

- **Performance Highlights**: 논문에서 제안하는 1/Ratio@k는 다양한 실험에서 Recall@k에 비해 더 효율적인 성능을 보였습니다. 여러 데이터셋에 대한 벤치마킹 결과, 1/Ratio@k 기준으로 작업 품질 기준에 도달하는 것이 훨씬 쉬운 것으로 나타났습니다. 또한, 구조적 성능 측면에서 Recall@k가 감소하더라도 1/Ratio@k는 안정적인 성능 지표를 유지하는 것을 확인하였습니다.



### SAILRec: Steering LLM Attention to Dual-Side Semantically Aligned Collaborative Embeddings for Recommendation (https://arxiv.org/abs/2606.04514)
Comments:
          17 pages, including appendices

- **What's New**: 최근 LLM(대형 언어 모델)을 기반으로 한 추천 시스템에서는 사용자-아이템 상호작용에서 유래한 협업 임베딩(collaborative embeddings)을 사용하게 되었지만, 이러한 임베딩을 사용하는 것이 추론 과정에서 효과적으로 활용된다는 보장은 없습니다. 진단적 주의 분석을 통해 협업 임베딩의 활용이 깊이에 따라 달라지고, 정렬에 민감하다는 것을 발견하였습니다. 이를 해결하기 위해 SAILRec을 제안하며, 이 시스템은 양측의 의미 정렬과 계층적 주의 조절을 통해 이 균형을 개선합니다.

- **Technical Details**: SAILRec은 사용자와 아이템의 협업 임베딩의 의미 접근성을 향상시키기 위해 이중 측면의 의미 정렬을 수행합니다. 아이템 측면은 아이템 텍스트의 LLM 의미 표현과 정렬되며, 사용자 측면은 과거 상호작용에서 유도된 코드북 기반의 의미 프로필과 정렬됩니다. 또한, 계층적 주의 조절(hierarchical attention steering)을 통해 Transformer 층 전반에 걸쳐 협업 지식의 사용 시점을 제어합니다.

- **Performance Highlights**: 실험 결과, SAILRec은 MovieLens-1M 및 Amazon-Book 두 공개 데이터 세트에서 일관되게 대표적인 기준선 모델들을 초월하는 성능을 보여주었습니다. ablation과 masking 분석을 통해 SAILRec의 핵심 설계를 확인할 수 있었습니다. 이러한 결과는 SAILRec이 협업 임베딩을 효과적으로 활용함을 보여줍니다.



### Bridging Short Videos and Live Streams: Reasoning-Guided Multimodal LLMs for Cross-Domain Representation Learning (https://arxiv.org/abs/2606.04448)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 짧은 동영상에서 라이브 스트리밍으로의 크로스 도메인 추천(cross-domain recommendation, CDR)을 위한 Reasoning-Guided Cross-Domain Representation Learning (RGCD-Rep) 프레임워크를 제안합니다. RGCD-Rep는 행동 협력을 통해 전이 가능한 아이템 표현을 학습하고, MLLM(다중 모드 대형 언어 모델)의 추론 기능을 효율적으로 도입하는 시스템입니다. 이 방법은 대규모 MLLM의 실질적인 산업적 적용을 용이하게 하며, 두 단계의 훈련 과정을 통해 모델의 성능을 향상시킵니다.

- **Technical Details**: RGCD-Rep는 두 가지 단계의 훈련을 포함하고 있습니다. 첫 번째 단계에서는 MLLM의 구조화된 추론 지식을 생성하고 이를 경량 학생 MLLM으로 증류(distillation)하는 ‘추론 인식 증류’를 수행합니다. 두 번째 단계에서는 도메인 간 행동 공존 신호를 캡처하고, 아이템 표현을 전이 가능(residual)한 성분으로 분해하는 ‘행동 기반 크로스 도메인 표현 학습’을 진행합니다. 이 모든 과정은 오프라인으로 수행되어, 산업적 배치에서 낮은 비용으로 활용할 수 있습니다.

- **Performance Highlights**: 앤드스트림 프로덕션에서 RGCD-Rep를 구현한 후 A/B 테스팅을 실시한 결과, 핵심 비즈니스 메트릭 전반에 걸쳐 유의미한 상승 효과를 확인하였습니다. 또한, RGCD-Rep는 4억 명 이상의 사용자를 대상으로 매일 제공되고 있으며, 실제 산업 환경에서 그 효용성과 실용성을 증명했습니다. 이러한 성과는 오프라인 실험에서도 탁월한 성능이 입증되었음을 반영합니다.



### Rethinking Sales Lead Scoring with LLM-based Hierarchical Preference Ranking (https://arxiv.org/abs/2606.04387)
- **What's New**: 이번 연구에서는 고위험 도메인에서의 영업 리드 점수를 매기는 혁신적인 프레임워크인 HPRO(Hierarchical Preference Ranking Optimization)를 소개합니다. 기존의 LLM(대형 언어 모델)을 기반으로 하여 구조적 CRM(고객 관계 관리) 기능과 비구조적 고객 상호작용을 결합한 모델링을 지원합니다. 이 프레임워크는 리드 점수를 매기는 과정에서 적은 양의 데이터로도 보다 효율적인 평가를 가능하게 합니다.

- **Technical Details**: 저자들은 영업 리드 점수를 매기는 문제를 차별화된 구조의 LLM 아키텍처로 재구성합니다. 여기서 HPRO는 희소한 이진 감독 신호를 계층적 선호 신호로 변환하며, 판매 퍼널의 단계 간 계층적 우선순위 를 활용하여 리드의 매력을 높입니다. 제안된 모델은 LLM의 사전 훈련된 구조에 비즈니스 우선순위를 반영한 점수 예측 헤드를 추가함으로써 점수를 매길 수 있는 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, HPRO는 대규모 NEV 브랜드의 데이터를 활용하여 AUC(Area Under Curve) 0.8161을 달성하며, 상위 리드의 정밀도를 39.7% 향상시켰습니다. 또한, 132일에 걸친 온라인 A/B 테스트 결과, 9.5%의 매출 증가를 입증하여 실질적인 상업적 효과를 확인했습니다. 이러한 성과는 판매 리드 스코어링의 새로운 방향을 제시합니다.



### DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling (https://arxiv.org/abs/2606.04374)
Comments:
          Jing Wang (Corresponding Author)

- **What's New**: 이 논문은 전자 상거래 검색의 관련성 문제를 다루고 있으며, 디스크리트 시맨틱 아이디 (SID)를 통해 쿼리 의존적인 순위를 모델링하는 새로운 방법인 DSIRM을 제안합니다. 기존에 사용되던 비지도 학습 기반의 SID 생성 방식의 한계를 극복하고자 하며, 쿼리-아이템 상호작용을 활용하여 정확한 시맨틱 패턴을 학습하는 방식을 채택합니다. 또한, 생성적인 대형 언어 모델을 이용하여 텍스트에서 아이템 SID를 예측하는 방법을 제시합니다.

- **Technical Details**: DSIRM은 쿼리-브리징 대조적 양자화 접근법을 통해 아이템 사이드에서 쿼리-아이템 상호작용을 모델링하며, 잔여 양자화(Residual Quantization) 과정에서 InfoNCE loss를 활용하여 유사한 쿼리와 함께 발생하는 아이템에 비슷한 SID를 부여하도록 합니다. 텍스트로부터 아이템 SID를 예측하기 위해 자가 회귀(autoregressive) LLM을 조정하여 의도 모호성을 해결하고, 쿼리 및 아이템 SID 간의 계층적 접두사 매칭을 통해 연속 임베딩을 보완하는 차별적인 특성을 생성합니다.

- **Performance Highlights**: Tmall의 대규모 생산 데이터를 활용한 실험 결과, 제안된 방법이 현재의 최첨단 모델보다 더 나은 성능을 나타내며, 오프라인 AUC가 +1.54% 향상되었습니다. 효율적인 하이브리드 아키텍처를 통해 배포된 DSIRM은 온라인에서 UCTR과 UCTCVR을 각각 +0.13%, +0.25% 향상시키며 상당한 산업적 가치를 입증하고 있습니다.



### Disentangling Answer Engine Optimization from Platform Growth: A Log-Based Natural Experiment on ChatGPT Referral Traffic (https://arxiv.org/abs/2606.04362)
Comments:
          9 pages, 4 figures, 1 table

- **What's New**: 최근 대형 언어 모델(LLM) 기반의 '답변 엔진'(answer engine)인 ChatGPT가 웹으로의 트래픽을 유도하고 있습니다. 이러한 현상은 '답변 엔진 최적화'(Answer Engine Optimization, AEO)라는 새로운 최적화 전략을 통해 나타나며, 이는 기존의 검색엔진 최적화(SEO)와 유사합니다. 본 연구는 고트래픽 도메인에서 AEO 기능을 검토하며, 무작위 실험이 아닌 자연 실험(natural experiment)을 통해 AEO의 효과를 분석합니다.

- **Technical Details**: 이 연구는 AEO 개입이 적용된 YouTube Q&A 페이지와 미개입 페이지를 비교하는 방법론을 사용합니다. AEO 개입은 URL 정형화, 수요 채굴, 제목 및 요약 재작성 등을 포함합니다. 분석은 주간 데이터를 활용한 구간 회귀(segmented regression)와 허위로 인한 시간 변동 테스트를 사용하여 플랫폼 성장효과를 제거할 수 있습니다.

- **Performance Highlights**: 실험 결과, ChatGPT의 추천은 5.7배 증가했으나, 미개입 페이지도 3.5배 증가했습니다. 개입 그룹에 대한 점검 결과는 1.82배의 증가를 나타냈으나, 이 효과는 유의미하진 않음(p=0.16)으로 나타났습니다. 더욱이, SEO 방어 규칙 덕분에 구글 클릭 수는 낮지 않았습니다.



### Argus-Retriever: Vision-LLM Late-Interaction Retrieval with Region-Aware Query-Conditioned MoE for Visual Document Retrieva (https://arxiv.org/abs/2606.04300)
- **What's New**: Argus는 쿼리-조건(Query-conditioned) late-interaction 비전-언어(Vision-Language) 검색기이다. 이 시스템은 질의(queries)를 처리하기 위해 영억 인식(Region-aware) 믹스처 오브 익스퍼트(Mixture-of-Experts) 모듈을 추가하여 문서 표현을 쿼리에 따라 조정한다. 결과적으로 Argus는 쿼리와의 상관관계를 반영한 독특한 문서 표현(D(q))을 생성하며, MaxSim으로 평가받는 멀티-벡터 인덱스를 유지한다.

- **Technical Details**: Argus는 문서 인코더 내에 쿼리-조건(region-condition) 라우터와 믹스처 오브 익스퍼트(Mixtures of Experts) 퓨전을 포함하여, 쿼리 기반의 다중 벡터 문서 표현을 생성한다. 모든 Argus 모델은 1024차원의 late-interaction 헤드를 사용하며, 이는 최근의 다른 시스템보다 차원이 적지만 일정한 성능을 유지한다. 또한 Argus는 공공 감독의 9%로 훈련되어 고득점(high scores)을 기록하였으며, MaxSim을 통해 쿼리에 대한 문서의 중요도를 평가한다.

- **Performance Highlights**: Argus-9B 모델은 ViDoRe V1에서 92.67의 NDCG@5를 달성하였고, V1+V2 리더보드에서 86.0을 기록하였다. 이는 오픈 late-interaction 모델 중 가장 높은 수치로, Argus는 단독 시스템으로서 강력한 성능을 발휘한다. ViDoRe V3에서 Qwen3.6-27B 에이전트 검색 파이프라인을 사용한 Argus-9B는 공공 작업에서 NDCG@10을 60.28에서 64.80으로 향상시켰다.



### NLLog: Lightweight, Explainable SOC Anomaly Detection via Log-to-Language Rewriting (https://arxiv.org/abs/2606.04957)
Comments:
          15 pages, 11 figures, 12 tables; submitted to ACSAC 2026

- **What's New**: NLLog(Natural-Language Log)는 기존의 템플릿 기반 로그의 의미적 불투명성을 해소하고 보안 운영 센터(SOC)에서 로그 분석을 간소화하기 위해 개발된 경량 파이프라인입니다. 이 시스템은 템플릿을 WHO-WHAT-SEVERITY 형태의 문장으로 변환하고, 효율적인 클래스 분류 및 분석 지원 기능을 제공합니다. NLLog는 기존의 로그 분석 모델보다 적은 수작업과 계산을 요구하면서도 더 높은 성능을 보여줍니다.

- **Technical Details**: NLLog는 원시 로그를 정형화하고, 각 로그 이벤트를 WHO-WHAT-SEVERITY(WWS) 문장으로 변환한 후, TF-IDF 가중치를 사용하여 세션 벡터로 임베딩합니다. 또한, 트리 앙상블(Tree Ensemble) 분류기를 통해 세션을 분류하고, TreeSHAP를 사용하여 결정의 기여도를 분석하여 사용자에게 설명 가능한 형태로 제공합니다. 이 모든 과정은 사전 훈련된 언어 모델(Pre-trained Language Model)을 포함하되, 추가적인 파인튜닝이 필요하지 않아 경량화된 SOC 배치에 적합합니다.

- **Performance Highlights**: NLLog는 Hadoop Distributed File System(HDFS)와 Blue Gene/L(BGL) 데이터셋에서 두 개의 재현된 비교 대상(DeepLog와 LogBERT)을 초과하는 성능을 기록하였고, AIT Alert Data Set에서도 높은 정밀도와 낮은 거짓 긍정률을 유지합니다. 결과적으로 NLLog는 운영 중인 경량 SOC 파이프라인에서 빠른 속도로 효과적인 로그 이상 탐지를 가능하게 합니다.



### Caliper: Probing Lexical Anchors versus Causal Structure in LLMs (https://arxiv.org/abs/2606.04915)
- **What's New**: 이번 논문에서는 Caliper라는 새로운 방법론을 소개합니다. 이 방법론은 언어 모델이 구조적 인과 추론(structural causal reasoning)을 수행하는지, 아니면 단순히 레퍼런스 패턴을 매칭하는지 분석하기 위해 설계되었습니다. Caliper는 의미 있는 변수 이름을 자리 표시자 토큰으로 변경하면서 인과 구조를 보존하는 통제된 변형을 제공합니다.

- **Technical Details**: Caliper는 내부 변수(됴내적 변수)와 외부 노이즈(외생 노이즈), 구조적 방정식, 그리고 노이즈 분포를 포함하는 구조적 인과 모델(SCM)을 기반으로 합니다. 이 모델은 Pearl의 인과 계층 구조를 따르고, 인과 질의의 맥락에서 일관된 평가를 가능하게 합니다. Caliper를 통해 각 질의의 인과 그래프를 유지하면서도 표면적인 어휘(content)를 제거함으로써, 모델의 인과적 추론 능력을 측정할 수 있습니다.

- **Performance Highlights**: 시험 결과, 14개의 instruction-tuned LLM을 사용하는 동안, 표면 어휘가 제거되면 인과적 정확도에서 유의미한 감소가 발견되었습니다. 특히 CRASS와 e-CARE 등 여러 벤치마크에서 정확도 저하가 두드러졌으며, 이는 LLM들이 구조적 인과 추론 대신 레퍼런스 기반으로 작동하고 있다는 것을 시사합니다. 이 연구 결과는 대규모 언어 모델의 인과적 성능을 평가하는 데 중요한 통찰을 제공합니다.



### Archi: Agentic Operations at the CMS Experimen (https://arxiv.org/abs/2606.04755)
- **What's New**: 아키(Archi)는 과학 협업을 위한 오픈소스(end-to-end) 프레임워크로, 이질적인 데이터 소스의 체계적인 수집과 조직을 결합합니다. 또한, 이를 통해 구성 가능한(private), 개인적인, 확장 가능한 에이전트를 배포하여 데이터를 검색하고 추론할 수 있습니다. 2026년 2월 이후로 CERN의 LHC 실험의 컴퓨팅 작업 팀에 배포되어 기술 운영자를 지원하는 에이전트로 기능하고 있습니다.

- **Technical Details**: Archi는 문서, 역사적 데이터 및 실시간 모니터링 시스템을 결합하여 검색 및 분석 기능을 제공합니다. 이 시스템은 운영자의 피드백과 실제 사용에서 수집된 질의 세트에 대해 평가되었습니다. 그 평가 기준은 인간 및 자동 패널에 의해 등급이 매겨졌습니다.

- **Performance Highlights**: 이 시스템은 운영 업무에 효과적이며 CMS 운영자가 제기한 실제 질문들을 해결하는 데 성공적입니다. 또한, 로컬에서 호스팅되는 오픈 웨이트 모델이 경쟁력 있게 작동하여, 민감한 데이터의 완전한 개인 관리를 가능하게 한다는 점이 관찰되었습니다.



### QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples (https://arxiv.org/abs/2606.04646)
Comments:
          14 pages

- **What's New**: 이 논문에서는 데이터베이스 스타일 쿼리와 유사한 자연어 질문에 대한 응답을 평가하기 위한 진단 벤치마크인 QO-Bench를 소개합니다. 기존의 검색 보강 생성(Retrieval-Augmented Generation, RAG) 시스템은 주로 의미론적 관련성에 최적화되어 있었으나, 실제 쿼리 실행의 정확성을 보장하지 않는다는 문제점을 지적합니다. QO-Bench는 특정 이벤트의 쿼리 연산자를 보존하는 검색을 목표로 하며, 22,984개의 뉴스 기사와 614개의 기업 사건을 바탕으로 785개의 질문을 평가합니다.

- **Technical Details**: QO-Bench는 이벤트 튜플에 대한 설정에서 쿼리-연산자 질문 응답(QO-QA)을 재정립합니다. 각 gold answer는 이벤트 튜플에서 결정적으로 계산되며, 템플릿 특정 리콜에 따라 점수를 매깁니다. 이 디자인은 각 실패를 특정 연산자로 귀속시킬 수 있도록 하여, RAG, ReAct RAG, GraphRAG, 정보 추출-SQL 시스템을 비교 분석합니다. 또한, 두 축 프레임워크인 인덱스-시간 보존과 쿼리-시간 실행을 통해 각 패러다임의 실패 요인을 파악합니다.

- **Performance Highlights**: 실험 결과 각 패러다임은 특정 연산자에서 우월성을 보이지 않으며, 유사성 검색은 필터/프로젝트에서 강점을 보이는 반면, 정보 추출-SQL 시스템은 교차 사건 조인에 약점을 보입니다. QO-Bench는 연산자 실행이 핵심 병목임을 시사하며, 이는 단순한 검색 대답 생성만으로는 개선되지 않습니다. 또한, Gold evidence를 이용한 긴 컨텍스트 오라클조차도 한계에 도달하지 못하였고, 이는 연산자 실행의 중요성을 부각시킵니다.



### Cartridges at Scale: Training Modular KV Caches over Large Document Collections (https://arxiv.org/abs/2606.04557)
Comments:
          21 pages, 5 figures, 17 tables

- **What's New**: 본 연구는 Cartridges at Scale (CAS)라는 새로운 훈련 프레임워크를 소개합니다. CAS는 다수의 카드리지(cartridge) 학습을 가능하게 하여 문서 크기가 수백 개에 달하는 경우에도 저렴하게 다루고, 동적 방해 요소 혼합과 메모리 효율적인 예산 관리 기능을 제공합니다. 또한 CAS는 기존의 한정적인 카드리지 접근법의 성능 한계를 극복하여 효율적인 데이터 처리를 보장합니다.

- **Technical Details**: CAS는 카드리지 훈련을 위해 동적 혼합 및 회전 시스템을 도입하고, GPU와 지속적 저장소 간의 스왑을 지원합니다. 이러한 방식으로 수십 개의 카드리지를 고정된 GPU 메모리 예산 내에서 동시에 훈련할 수 있으며, 다양한 질문 생성을 위한 더 나은 데이터 생성 방법도 제안합니다. 또한, 독립적으로 훈련된 카드리지가 충돌을 일으키는 문제를 해결하고, 집중적인 훈련을 통해 카드리지 간의 결합 성능 향상을 보장합니다.

- **Performance Highlights**: CAS는 다수의 벤치마크에서 단일 카드리지보다 30점 이상 성능이 향상되었습니다. 또한, CAS는 기존의 검색 보강 생성(RAG) 방식보다 4배 적은 프롬프트 토큰을 소모하면서도 유사하거나 높은 정확도로 결과를 도출할 수 있습니다. 이 연구는 카드리지 기반의 학습 모델이 상업적 및 실용적 환경에 얼마나 잘 적응하는지를 보여줍니다.



### Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation (https://arxiv.org/abs/2606.04435)
- **What's New**: 이번 논문에서는 복잡한 추론 작업에서 기존의 환각(hallucination) 탐지 메커니즘이 간과하는 연쇄 환각(cascading hallucination)이라는 새로운 실패 유형을 정의합니다. 이는 초기 단계에서 발생한 오류가 후속 단계에서 증폭되며, 결국에는 사실과 다른 최종 출력을 생성하는 문제입니다. 이를 해결하기 위해, 선형 인과관계를 통한 오류 전파를 탐지하고 중단하는 CHARM(연쇄 환각 인식 해소 및 완화) 아키텍처를 소개합니다.

- **Technical Details**: CHARM 아키텍처는 네 가지 구성 요소로 이루어져 있습니다: 단계별 사실 검증(stage-level fact verification), 단계 간 일관성 추적(cross-stage consistency tracking), 신뢰성 전파 모니터링(confidence propagation monitoring), 및 연쇄 해소 트리거(cascade resolution triggering)입니다. 이 시스템은 기존 RAG(Retrieval-Augmented Generation) 파이프라인과 통합되어 아키텍처 교체 없이도 동작합니다. CHARM은 HotpotQA, MuSiQue, 2WikiMultiHopQA 등의 데이터셋에서 평가되었으며, 평균 89.4%의 연쇄 탐지율 및 5.3%의 위양성률을 기록했습니다.

- **Performance Highlights**: CHARM을 통한 오류 전파 감소율은 82.1%에 달하며, 이는 기존 출력 수준 탐지기의 18.5%에 비해 월등한 성능입니다. 각 탐지 모듈의 기여도가 확인된 바와 같이, CHARM은 사전 오류 전파를 방지함으로써 최종 출력의 신뢰성을 제고합니다. CHARM은 또한 인간 감시 체계와 통합되어 생산적인 인공지능 배치에 필요한 신뢰성과 거버넌스 체계를 제공합니다.



### Context-as-a-Service: Surfacing Cross-File Dependency Chains for LLM-Generated Developer Documentation (https://arxiv.org/abs/2606.04397)
Comments:
          8 pages, 2 figures, 4 tables

- **What's New**: 이번 논문은 LLM(Large Language Model) 에이전트가 개발자 문서를 작성하고 유지하는 과정에서 Context-as-a-Service (CaaS)라는 새로운 검색 레이어를 소개합니다. CaaS는 코드베이스 전반에서 증거를 검색하는데 활용되며, API 참조 및 상위 문서와 일관된 문서 생성을 지원합니다. 이를 통해 에이전트는 반복적으로 의존성을 식별하고 문서를 검토할 수 있어, 문서의 유용성과 정확성이 크게 향상됩니다.

- **Technical Details**: CaaS는 코드, 예제, 테스트 및 API 참조 등 다양한 출처의 정보를 인덱싱하고 검색할 수 있는 구성이며, 이 구조는 네 개의 주요 단계로 나뉩니다: 수집, 저장, 검색 및 검토입니다. 검색 프로세스에서는 BM25 알고리즘과 DRAMA를 결합해 에이전트가 라이브러리와 문서의 상관관계를 세밀하게 파악할 수 있도록 지원합니다. 에이전트는 CaaS를 통해 빠르고 효율적으로 정보에 접근하고, 필요한 경우 파일을 열어 직접 확인할 수 있습니다.

- **Performance Highlights**: CaaS의 도입은 두 가지 사례 연구에서 전반적으로 벽 시계 시간(wall-clock time)을 22%에서 34%까지 줄이는 효과를 보였습니다. 이로 인해 LLM 에이전트는 누락된 문서 외에도 코드베이스의 다양한 오류를 발견할 수 있었으며, 특히 API 참조 검토 및 튜토리얼 검증에서 중요한 문서 문제를 드러냈습니다. CaaS는 국지적으로 타당한 문서가 코드베이스 전반에서 유효성을 확인할 수 있도록 증거를 제시함으로써, 문서 탐색의 효율성을 극대화하였습니다.



### LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignmen (https://arxiv.org/abs/2606.04382)
- **What's New**: 이번 연구에서는 LCSH(Library of Congress Subject Headings)의 공인된 벤치마크가 없다는 문제를 해결하기 위해 LCSHBench를 소개합니다. LCSHBench는 하버드, 컬럼비아, 프린스턴 기록에서 수집된 22,346권의 도서로 구성되어 있으며, 15개 언어로 제공됩니다. 이 데이터셋은 두 개 이상의 독립적인 카탈로그 에이전시가 LCSH를 할당한 도서만 포함되고, 다양한 언어와 주제 유형에 대한 정확도 측정을 지원합니다.

- **Technical Details**: 연구에서 제안한 LCSHBench는 하나의 공통적인 기준으로 카탈로그의 합의를 기반으로 하고 있습니다. 465,187개의 도서에 대한 카탈로그링을 분석하여, 주제에 대한 카탈로그자 간의 동의 정도를 정량화하고, 이는 객관적 개념 수준과 주관적 표현 수준으로 나뉘어 분석됩니다. 이를 통해 두 가지 특정 작업인 open-vocabulary generation과 전체 어휘 검색 파이프라인에 대한 평가 기준을 마련합니다.

- **Performance Highlights**: LCSHBench를 활용한 첫 번째 실험에서는 300M 온디바이스 임베더를 저랭크로 파인튜닝하여 언어 간 검색 성능을 개선했습니다. 개발 세트의 정확한 회수율에서 3,072차원으로 호스팅된 임베더(0.659)보다 더 나은 성과를 보였습니다(0.623). 하지만 이 연구에서는 언어 패널에 따라 성과가 균일하지 않음을 보여주며, 향후 연구로는 보류된 테스트 및 엔드 투 엔드 확인을 계획하고 있습니다.



### Creative Reading: Scaffolding Reading for Transformation (https://arxiv.org/abs/2606.04308)
- **What's New**: 이 논문은 독서 보조 시스템이 정보 전송으로 읽기를 재구성하는 경향이 있음을 비판하며, 독자가 해석 및 노력하는 과정을 중요시해야 한다고 주장합니다. 기존의 연구에서는 독서가 효율성을 중시하고 '버리기 위한 읽기(reading to discard)'로 전환되는 반면, 이 논문에서는 '창의적 읽기(creative reading)'를 통해 독자가 의미를 생성하고 스스로 발전할 수 있는 대안을 제시합니다.

- **Technical Details**: 저자들은 문학 및 내러티브 이론을 학술적 의미 만들기(sensemaking) 및 창의성 지원과 연계하여 독서 과정을 중시하는 디자인 공간을 제안합니다. 이 접근 방식은 독자가 단순히 정보를 추출하는 것이 아니라, 텍스트와의 상호작용에서 발생하는 창의성의 결과물인 "읽기(a reading)"를 생산하도록 돕는 것을 목표로 합니다.

- **Performance Highlights**: 이번 연구는 기존 독서 보조 시스템이 정보 추출에 초점을 맞추는 경향이 있음을 도전하고, 독자가 텍스트와의 상호작용을 통해 개인적인 의미와 그림을 생성하도록 지원해야 한다는 중요성을 강조합니다. 이러한 변화를 통해 학술적 독서 환경에서도 독자가 더 창의적이고 능동적인 역할을 할 수 있는 가능성을 열어줍니다.



### The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning (https://arxiv.org/abs/2606.04280)
- **What's New**: 이 논문은 대비 학습(contrastive learning, CL)에 관한 새로운 이론적 기초를 제시합니다. 기존의 CL 메커니즘이 어떻게 의미 있는 잠재 구조(latent geometry)를 회복하는지에 대한 이해를 심화시키고자 합니다. 특히, 데이터 샘플링의 다름(diversity condition)이 긴급히 필요함을 강조하며 이 조건이 위배될 경우 발생하는 기하학적 왜곡을 규명합니다.

- **Technical Details**: 저자들은 잠재 공간(코드) 샘플링에 대한 측정 이론(measure-theoretic) 기반의 다양성 조건(diversity condition)을 공식화했습니다. 이 조건은 아이소메트릭(거리 보존) 회복을 위해 필수적인 것으로, 샘플링이 제한된 경우의 고전적인 전지원(von Mises-Fisher) 설정에서 적합한 결과를 도출합니다. 이를 통해 적절한 InfoNCE 목표를 제안하고, 기하학적 보존(latent space recovery)을 가능하게 하는 방법론을 살펴봅니다.

- **Performance Highlights**: CIFAR-10 데이터셋을 통해 이론적 예측을 실험적으로 검증하며, 건축 인덕티브 바이어스(architectural inductive bias)가 샘플링 다양성이 제한될 때 표현 품질에 어떤 영향을 미치는지 분석합니다. 결과적으로, 다양한 샘플링 전략과 인코더 구조가 대조적 표현 학습에서 어떻게 상호 작용하는지를 명확히 하여, 향후 연구 및 방법론의 발전 방향을 제시합니다.



### Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieva (https://arxiv.org/abs/2606.04194)
Comments:
          9 pages, 3 figures, 10 tables. Code, data, and per-table receipts: this https URL

- **What's New**: 이 논문은 다양한 대화 세션의 역사에서 새로운 질문에 대한 답변을 포함한 과거 발화(turn)를 효과적으로 검색하는 방법론을 제시합니다. 특히, 이전 연구들과 달리, 정보 검색에서 훈련이 필요 없는 CPU 전용의 검색이 대화 메모리 구조에서 어떤 이점을 가질 수 있는지를 탐구하고 있습니다. 연구 결과는 대화 기억 회수의 효율성을 향상시키기 위한 다양한 실험적 통찰을 제공합니다.

- **Technical Details**: 연구는 비교 점수(score-level fusion) 기법을 사용하여 late-interaction과 BM25 방식의 결합을 통한 효율성을 강조합니다. 특히, late-interaction이 특정 쿼리에 대해 성능이 높다는 것을 입증하고, BM25과 결합 시 성능 향상이 이루어지는 것을 보여줍니다. 실험에서는 다양한 인코더(encoder)를 사용하여 이 결합의 효과를 정량적으로 평가합니다.

- **Performance Highlights**: 결과적으로, late-interaction과 BM25 점수의 융합을 통해 LoCoMo Hit@1이 8.8에서 17.2 포인트 향상된 결과가 나타났습니다. 하지만, 최상위 10개 결과에 대한 재정렬(reranking)의 적용이 Hit@1을 6.9 포인트 감소시키는 반효과를 일으키기도 하였습니다. 각각의 인코더에 대한 성능 차이를 분석한 결과, 딴 유형의 질문에 따라 다양한 성능 차이를 보였습니다.



New uploads on arXiv(cs.CV)

### Controllable Dynamic 3D Shape Generation via 3D Trajectories and Tex (https://arxiv.org/abs/2606.05162)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 T2Mo라는 피드포워드 프레임워크를 소개하며, 이는 3D 궤적과 텍스트를 조건으로 하는 동적 3D 형태 생성 기능을 갖추고 있습니다. 텍스트만을 사용한 모션 생성의 불확실성을 극복하기 위해 3D 궤적을 사용하여 특정 포인트가 이동해야 하는 경로를 명확히 지정합니다. 이를 통해 T2Mo는 주어진 궤적을 따르면서도 텍스트의 의미를 반영하는 객체의 모션을 생성할 수 있습니다.

- **Technical Details**: T2Mo는 사용자가 제공하는 3D 궤적과 글로벌 모션 의미를 제공하는 텍스트 프롬프트를 사용하여 동적 3D 형태를 제어 가능하게 생성하는 모델입니다. 사용자 제공 궤적은 다양한 구성으로 제공될 수 있으며, 당사의 접근법은 모든 형태의 궤적 구성을 처리하기 위해 형태-기반 궤적 임베딩을 설계했습니다. 이 임베딩은 입력 궤적 세트를 형태 인식을 고려한 특정 크기의 토큰 집합으로 변환하여 모든 구성에 대해 일관된 조건을 제공합니다.

- **Performance Highlights**: 종합적인 실험과 사용자 선호 조사 결과, T2Mo는 최근의 텍스트 기반 모델이나 비디오 기반 모델과 비교해 모션의 제어 가능성 및 표현력을 향상시켰으며, 전반적인 모션 품질을 저해하지 않고 다양한 사용자 정의 모션 제어를 가능하게 합니다. 정량적 및 정성적 평가 결과, T2Mo는 주어진 프롬프트에 더 충실하게 모션을 생성하고 표현력이 뛰어난 것을 확인했습니다.



### An Open-Source Two-Stage Computer Vision Pipeline for Fine-Grained Vehicle Classification using Vision Transformers (https://arxiv.org/abs/2606.05149)
Comments:
          24 pages, 10 figures, venue TBD

- **What's New**: 이 논문에서는 도로 비디오 자료로부터 사이클리스트 부상 심각도에 관련된 차량 유형 분류를 위한 자동화 도구를 제시합니다. 기존의 객체 탐지 기준은 차량을 단순히 차, 트럭, 버스 및 오토바이와 같은 큰 카테고리로만 분류하는 데 그쳤으나, 이 연구는 6가지 차량 유형(승용차, SUV, 픽업 트럭, 미니밴, 대형 밴, 상용 트럭)을 세분화하여 분류하는 새로운 시스템을 개발했습니다. 또한, 소프트맥스 출력이 미달일 경우 예측을 자제하는 기법을 통해 잘못된 분류를 방지하는 기제를 도입했습니다.

- **Technical Details**: 제안된 시스템은 두 단계의 분류 파이프라인으로 구성되며, 먼저 RT-DETR(Real-Time Detection Transformer) 탐지기가 차량을 대략적으로 탐지하고, 이후 세밀한 Vision Transformer(ViT-Base/16)를 통해 차량을 6개 유형으로 분류합니다. 이 시스템은 별도의 바운딩 박스 주석 없이 표준 도로 비디오에서 사용이 가능하며, 규정된 임계값 이하일 경우 소프트맥스 출력을 통한 불확실한 예측을 보류합니다. 3,805개의 주석이 달린 오버테이킹 이벤트를 평가하여 전체 정확도 0.94와 클래스별 F1 점수를 기록했습니다.

- **Performance Highlights**: 벤치마크 평가에서, 모든 종류의 차량을 포함한 검증에서 0.89의 정확도를 달성했습니다. F1 점수는 SUV에서 0.97, 미니밴은 0.72로 나타나면서 모델의 불확실성이 증가했음을 보여줍니다. 이는 교육 자료의 조건이 달라질 때도 높은 정확성을 유지할 수 있는 가능성을 보여줘, 이 시스템이 다양한 도로 환경에서도 효과적으로 적용될 수 있음을 시사합니다.



### GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes (https://arxiv.org/abs/2606.05142)
Comments:
          Project page: this https URL

- **What's New**: GeM-NR(new approach)은 빠르고 유연한 훈련 없는 방식으로 일반 다중 뷰 일관성 이미지를 편집할 수 있는 방법을 제시합니다. 이 방법은 수정된 이미지(anchor image)와 편집되지 않은 이미지(query image)를 비교하여 일관된 편집을 수행하며, 여기서 중요한 것은 씬의 기하학과 모습을 상당히 변화시킬 수 있는 것입니다. 전체 프로세스는 깊이 맵 추정(depth map estimation), 쿼리 뷰로의 투영(projection onto a query viewpoint), 및 최종 이미지 개선(refinement of the obtained image) 단계로 구성됩니다.

- **Technical Details**: GeM-NR의 핵심 기술은 수정된 씬의 3D 포인트 클라우드의 정렬을 극대화하는 깊이 맵 추정 전략을 사용하는 것입니다. 이 과정에서 깊이 추정기가 필요하며, 이후에는 보정된 쿼리 이미지를 바탕으로 편집 프로세스를 진행합니다. 다중 뷰 편집을 통해 쿼리 이미지에 일관된 편집을 적용하고, 다수의 뷰에도 스케일링할 수 있도록 설계되어 있는 것이 이 방법의 특징입니다.

- **Performance Highlights**: 실험 결과 GeM-NR 방식은 다양한 편집 작업에서 일관성을 높일 수 있음을 보여주었습니다. 특히 기하학적 및 외관적 변화가 큰 편집을 수행하는 능력이 기존 방법보다 뛰어난 성능을 보였습니다. 정량적 및 정성적 성과 모두에서 우리의 방법이 최신 기술(state-of-the-art)과 비교하여 우수한 품질의 편집 결과를 생성하는 것을 나타내고 있습니다.



### Continual Visual and Verbal Learning Through a Child's Egocentric Inpu (https://arxiv.org/abs/2606.05115)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 BabyCL이라는 연속(multimodal) 언어 학습 프레임워크를 소개합니다. 이 프레임워크는 SAYCam 데이터셋을 단일 시간 순서에 따라 처리하여 아동의 실제 경험에 근접한 방식으로 학습을 진행합니다. BabyCL은 시각적 표현 학습과 이미지-텍스트 대비(objective)를 결합하여 언어 학습을 효율적으로 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: BabyCL은 이벤트 세그먼트로 나누어진 연속적인 비디오 스트림에서 높은 차원의 시각적 표현 학습을 수행합니다. 각 세그먼트는 약 3분간의 텀을 두고 클러스터링 방법을 사용해 생성되며, 두 개의 재생 버퍼를 통해 시각적 및 다중 모달(history) 데이터를 독립적으로 관리합니다. 이 프레임워크는 공유된 백본(backbone)을 통해 동시에 세 가지 대조 손실(loss)을 최적화함으로써 비주얼과 언어 학습을 동시에 진행합니다.

- **Performance Highlights**: BabyCL은 SAYCam Labeled-S 4AFC 벤치마크에서 기존의 스트리밍 학습 기준과 비교해 개선된 성과를 보이며 오프라인 훈련의 상한선과의 격차를 크게 줄였습니다. 또한, 여러 실험 결과를 통해 온라인 시간 세그먼트 창의 길이나 재생 버퍼의 퇴거 규칙에 관계없이 얻은 성과가 견고하다는 것을 입증하였습니다. 이는 아동의 실제적인 경험에 더 가까운 훈련 조건 하에서도 의미 있는 단어-참조 매핑이 생성될 수 있다는 것을 보여줍니다.



### Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Hav (https://arxiv.org/abs/2606.05107)
- **What's New**: 이번 연구에서는 강력하지만 일반적인 비전 기초 모델을 전문적인 과학 분야에 적응시키기 위한 라벨이 없는 접근 방식을 제안합니다. 전통적인 감독된 파인 튜닝(supervised fine-tuning)은 라벨이 부족하고, 특정 작업을 위한 훈련이 모델의 일반성을 해치고 강건성을 감소시킬 수 있어 적합하지 않습니다. 대신 우리는 메타데이터(metadata)를 활용하여 새로운 도메인에 대한 표현을 자기 감독(self-supervised) 방식으로 조정합니다.

- **Technical Details**: 우리의 방법론인 FINO는 표준 자기 감독 목표와 유연한 메타데이터 안내를 결합하여 매우 세밀한 이산 메타데이터(discrete metadata)와 연속 메타데이터(continuous metadata) 모두를 처리합니다. 이 방법은 표현이 정보적 요소를 유지하도록 장려하면서 잡음 요소를 억제하도록 설계되었습니다. FINO는 세포 내 형광 현미경(subcellular fluorescence microscopy), 지구 관측(Earth observation), 야생 동물 모니터링(wildlife monitoring) 및 의료 이미징(medical imaging) 분야에서 전통적인 비지도 도메인 적응(unsupervised domain adaptation) 및 전적으로 감독된 적응(fully supervised adaptation)을 초월하는 성능을 보입니다.

- **Performance Highlights**: FINO는 작업 라벨(task labels) 없이 백본(adaptation) 적응을 수행하고, 슈퍼비전(supervision)을 위해 가벼운 프로브(probes)만을 사용하면서도 전문적인 도메인 특화(state of the art) 성능을 초과하는 결과를 보여주었습니다. 이는 다양한 과학적 도메인에서 FINO의 뛰어난 일반화 능력을 강조합니다. 이 연구는 어떻게 메타데이터를 활용하여 높은 성능을 유지하는지에 대한 중요한 통찰력을 제공합니다.



### ZipSplat: Fewer Gaussians, Better Splats (https://arxiv.org/abs/2606.05102)
- **What's New**: ZipSplat(짚스플랫)은 장면을 2D 픽셀 그리드에서 분리된 컴팩트한 장면 토큰 집합으로 처리하는 새로운 피드포워드 아키텍처입니다. 이 모델은 각 입력 픽셀에 대해 하나의 Gaussian(가우시안)만을 예측하는 전통적인 접근 방식을 극복하여 3D 위치에 구애받지 않는 Gaussians 집합을 생산합니다. ZipSplat은 DL3DV와 RealEstate10K에서 ${	ext{∼}}6{	imes}$ 더 적은 Gaussians로 최신 기술 수준을 설정했으며, 제로샷(zero-shot) 일반화를 통해 Mip-NeRF360 및 ScanNet++에서 모두 뛰어난 성능을 보입니다.

- **Technical Details**: ZipSplat는 장면 토큰을 통해 복잡한 기하학이 필요한 부분에 Gaussians를 집중시킬 수 있도록 설계되어 있습니다. 이 모델은 멀티 뷰 기초 모델(multi-view backbone)을 통해 Dense Visual Tokens를 추출하고, K-means 클러스터링을 사용하여 이를 압축된 장면 토큰으로 변환합니다. 또한, Cross- 및 Self-Attention 레이어가 이러한 토큰을 정제하며, 경량의 MLP가 각 토큰을 3D 위치에 구애받지 않는 Gaussians 집단으로 복호화합니다.

- **Performance Highlights**: ZipSplat는 품질-효율 곡선에서 전체 범위를 커버합니다. 덕분에 높은 충실도에서 컴팩트한 재구성까지 한 모델로 조정할 수 있습니다. 기존의 픽셀 기반 메소드들보다 6배 더 적은 Gaussians를 예측할 수 있으며, 이는 문맥 뷰가 늘어날 때 품질을 안정적으로 유지합니다.



### InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Spac (https://arxiv.org/abs/2606.05071)
Comments:
          Computer Vision and Pattern Recognition (CVPR), 2026

- **What's New**: 이 논문은 언어 기반의 사진 리터칭(photoretouching) 방법을 제안합니다. 기존의 디퓨전 모델(diffusion model)보다 더 높은 충실도(fidelity)와 효율성(efficiency)을 보여주는 컴팩트하고 콘텐츠 비동기적인(bilateral space) 변환 기술을 활용하고 있습니다. 새로운 방법론은 고해상도의 이미지에서 복잡한 색상 조정을 가능하게 하며 향상된 시각적 품질을 제공하는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 저해상도(bilateral grid) 업데이트를 예측하여 원본 픽셀을 직접 편집하는 대신, 학습된 가이드 맵을 사용하여 전체해상도(full-resolution) 이미지에 적용합니다. Variational Score Distillation을 통해 다단계 디퓨전 모델의 지식을 압축하고, 프롬프트 정렬 손실(prompt alignment loss)을 보완하여 지침 준수를 유도합니다. 이러한 구조는 고해상도 리터칭을 위한 효율적인 접근 방식을 제공합니다.

- **Performance Highlights**: 제안된 방법은 iRetouch라는 새로운 벤치마크를 도입하여 성능을 평가합니다. 이 방법은 내용 충실도(content fidelity), 지침 준수(instruction following), 효율성(efficiency)에서 뛰어난 결과를 보여주며, 기존 리터칭 방법에 비해 70~800배 빠르고 시각적으로 만족스러운 결과를 생성합니다. 결과적으로 제안된 방법은 고충실도를 유지하면서 지침을 따르는 능력이 뛰어난 것으로 평가받고 있습니다.



### MaCo-GAN: Manifold-Contrastive Adversarial Learning for Single Image Super-Resolution (https://arxiv.org/abs/2606.05068)
- **What's New**: 본 논문에서는 Single Image Super-Resolution (SISR)을 위한 새로운 맥락 대조 생성적 적대 신경망(MaCo-GAN)을 제안합니다. 기존 GAN에서의 적대적 손실을 감독된 대조 목표로 대체하여 더욱 견고한 훈련 과정을 가능하게 합니다. 특히, 실제 데이터에서 감지된 평균적인 왜곡성을 유지하며 다양한 도전적인 가짜 이미지를 생성하는 동적 가짜 샘플 합성기를 도입합니다.

- **Technical Details**: MaCo-GAN의 핵심은 각 SISR 예측을 더 많은 원하는 솔루션으로 끌어당기고 바람직하지 않은 솔루션에서 밀어내는 학습 메커니즘입니다. 이를 위해, SISR 네트워크는 긍정적 샘플 집합으로 향하고, 변별자는 정확히 반대 방향으로 최적화됩니다. 이러한 방식은 전통적인 GAN 손실보다 더 유리한 퍼셉션-왜곡 절충을 제공합니다.

- **Performance Highlights**: 연구 결과, MaCo-GAN은 다양한 벤치마크에서 기존의 방법들보다 일관되게 향상된 성능을 보였습니다. 다각적인 분석을 통해 이 프레임워크의 효과성을 검증하였고, GAN 훈련의 동적 성질에 대한 깊은 통찰을 제공합니다. 최종적으로, 이 연구는 GAN 기반의 대조적 접근법이 전략적인 특징을 최대한 활용하여 SISR 문제를 해결하는 데 기여할 수 있음을 보여줍니다.



### UniCAD: A Unified Benchmark and Universal Model for Multi-Modal Multi-Task CAD (https://arxiv.org/abs/2606.05058)
- **What's New**: 이번 논문은 CAD(Computer-Aided Design)의 다중 모드 및 다중 작업 학습을 위한 새로운 기준점인 UniCAD를 제안합니다. UniCAD는 다양한 입력 모드에서 CAD 재구성, CAD 생성 및 CAD 질문 응답 등의 작업을 포함하며, 이로 인해 다양한 CAD 작업을 통합적으로 평가할 수 있는 방법을 제공합니다. 또한, UniCAD-MLLM이라는 다중 모드 대형 언어 모델을 통해 텍스트, 이미지, 스케치 및 포인트 클라우드를 활용하여 CAD 작업을 통합적으로 수행합니다.

- **Technical Details**: UniCAD는 텍스트, 이미지, 스케치 및 포인트 클라우드의 다양한 입력 모드를 통합하여 CAD 모델을 생성하고 이해하는 데 필요한 평가 프로토콜과 작업별 메트릭스를 정의합니다. UniCAD-MLLM은 각기 다른 입력 모드를 처리하기 위해 모드별 인코더를 사용하고, 이를 공유된 기하학적 및 의미적 잠재 공간으로 투사하여 CAD 생성 및 이해 작업을 통합적으로 수행합니다. 특히, 최종 CAD 출력은 실행 가능하고 편집 가능한 Python 스크립트 형식으로 생성되어, 사용자가 쉽게 해석하고 수정할 수 있습니다.

- **Performance Highlights**: UniCAD-MLLM은 UniCAD 및 Fusion360 벤치마크에서 폭넓은 실험을 통해 기존의 작업별 및 다중 작업 기준선 모델보다 우수한 성능을 기록하였습니다. 대규모 다중 모드, 다중 작업 학습을 통해 특화된 시스템을 초월하는 성능을 보여줍니다. 이러한 결과는 통합 CAD 모델링의 효과성과 확장성을 검증하며, 향후 연구를 위한 데이터셋, 코드 및 사전 훈련 모델을 공개할 예정입니다.



### Anchor3R: Streaming 3D Reconstruction with Transient Anchors for Long-Horizon Visual Mapping (https://arxiv.org/abs/2606.05035)
- **What's New**: 최근의 스트리밍 3D 재구성 모델들은 과거의 고정된 좌표계를 유지하면서 카메라의 포즈를 예측합니다. 이는 훈련 과정에서 본 것보다 긴 시퀀스에서 드리프트가 축적되고, 최신 정보의 예측이 왜곡될 수 있습니다. 이를 해결하기 위해, 본 연구에서는 Anchor3R이라는 프레임워크를 제안하며, 이를 통해 현재 프레임을 기준으로 하는 로컬 측정 예측으로 스트리밍 재구성을 접근합니다.

- **Technical Details**: Anchor3R은 현재 프레임을 일시적인 기준으로 사용하여 상대 포즈 측정을 예측하는 구조를 가집니다. 각 시간 단계에서, Anchor3R은 다수의 기준점으로부터의 중첩된 예측을 통해 상대 포즈 그래프를 형성하고, 이는 온라인 포즈 업데이트 및 루프-클로저 기법을 통해 일관된 전역 재구성으로 드리프트를 정리합니다. 또한, 계산 자원과 메모리를 효율적으로 사용하기 위해, 포즈 쿼리 기반의 스트리밍 Transformer를 도입하였습니다.

- **Performance Highlights**: 실험 결과, Anchor3R은 실내, 실외, 주행 및 RGB-D 벤치마크에서 기존의 스트리밍 기준선보다 긴 지평선에서 포즈 정확도와 밀집 재구성 품질이 향상되었습니다. 더욱이, 이 알고리즘은 제한된 메모리 환경 속에서도 온라인 추론을 원활하게 지원하는 성능을 보였습니다. 이는 특히 로봇 비전 작업에서 실질적인 향상을 가져올 수 있는 기대감을 주고 있습니다.



### MetaPoint: Unlocking Precise Spatial Control in Agentic Visual Generation (https://arxiv.org/abs/2606.05031)
- **What's New**: 이 논문에서 제안하는 MetaPoint는 기존의 Generative visual models가 겪는 공간적 제어의 한계를 극복하는 새로운 접근법입니다. MetaPoint는 2D 좌표를 특별한 토큰으로 표현하여 모델이 숫자 좌표를 이미지 캔버스에 직접 연관지을 수 있게 합니다. 이 방법은 새로운 아키텍처 구성요소 없이 모델의 고유한 positional encoding 방식을 활용하여 구현되므로, 경량화된 방식으로 픽셀 단위의 위치 제어를 가능하게 합니다.

- **Technical Details**: MetaPoint는 특정 텍스트 토큰(<mp>)을 사용하여 공간적 포지셔닝을 구현하는데, 이 토큰은 위치의 정확한 좌표를 인코딩 할 수 있도록 설계되어 있습니다. 이를 통해 사용자 요청을 구조적인 프리미티브 시퀀스로 분해할 수 있는 계획 에이전트로서의 기능도 제공합니다. 이 인터페이스는 단일 토큰으로 점을 지정하고, 두 개의 토큰으로는 경계 상자를 형성하며, 짧은 시퀀스는 복잡한 객체 레이아웃을 인코딩할 수 있습니다.

- **Performance Highlights**: MetaPoint는 다양한 벤치마크에서 강력한 성능 향상을 입증했습니다. COCO-MIG에서 mIoU가 59.23%에서 77.29%로 증가하며 (+30.49% 상대적), T2I-CoReBench와 ImgEdit에서도 각각 73%와 15.2%의 향상을 이끌어냈습니다. 특히, 작업의 난이도가 증가할수록 이러한 이점은 더 두드러지며, 이는 정확한 2D 위치 기반 이해가 강력한 공간적 추론을 위한 핵심 요소임을 시사합니다.



### Handwriting Extraction and Analysis of Signature Lists in Swiss Popular Initiatives (https://arxiv.org/abs/2606.05018)
Comments:
          Accepted for presentation at ICCST 2026

- **What's New**: 이 논문은 스위스 민주주의에서 중요한 역할을 하는 서명 검증 프로세스를 자동화하기 위한 새로운 접근법을 제안합니다. 기존의 수작업 검증이 시중의 서명 데이터를 검색하고 확인하는 데 시간이 소요되는 노동집약적 과정이라는 점을 감안하여, 자동화된 문서 분석 기법인 OCR(Optical Character Recognition)과 AI 기반의 필적 분석이 서명 리스트 검증에 어떻게 기여할 수 있는지를 조사합니다. 특히, 필적 기반으로 중복 서명을 탐지하는 방법에 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법론은 템플릿 기반의 줄 분할(line segmentation), 텍스트 인식을 위한 알고리즘, 필자 검색(writer retrieval) 기술의 조합으로 구성되어 있습니다. 저자들은 443개의 핸드라이팅 로우와 418명의 필자가 포함된 데이터셋을 사용하여 이 방법을 평가했습니다. 문서의 데이터를 분석한 결과, OCR은 어휘에 없는 필적 데이터를 인식하는 데 어려움이 많았으며, 필자 검색 시스템은 훨씬 더 견고한 성능을 보여주었습니다.

- **Performance Highlights**: 문서 분석 결과, 필자 검색 기법은 mAP(Mean Average Precision) 50.6%에 도달했으며, 이는 서명 리스트에서 시각적으로 유사한 항목을 효과적으로 찾을 수 있음을 의미합니다. 더불어, OCR 시스템은 짧고 어휘에 없는 엔트리의 필기체 텍스트 전사에서 신뢰성을 보여주지 못하였고, 특히 이름이나 주소와 같은 짧은 데이터에 대한 성능이 저조했습니다. 이러한 결과는 필자 검색 기술을 중복 제출 탐지에 적합한 도구로 활용할 수 있음을 시사합니다.



### CIPER: A Unified Framework for Cross-view Image-retrieval and Pose-estimation (https://arxiv.org/abs/2606.05011)
Comments:
          16 pages, 5 figures

- **What's New**: 본 연구에서는 cross-view geo-localization 문제를 새로운 방식으로 접근하여, 대규모 도시 이미지 검색과 정밀한 3-DoF 포즈 추정을 동시에 수행하는 통합 프레임워크를 제안합니다. 기존의 두 독립적인 접근 방식을 통합하여 반복적인 오류 전파와 비일관적인 특징 표현 문제를 해결하고자 하였습니다. 새로운 아키텍처 CIPER(Cross-view Image-retrieval and Pose-estimation transformER)는 상호 이익을 주는 특징 학습을 통해 두 가지 작업을 함께 수행합니다.

- **Technical Details**: CIPER는 공유되는 transformer encoder와 전용 pose decoder를 사용하여 이미지 검색과 포즈 추정을 동시에 처리합니다. 이 구조는 글로벌 검색 특징과 공간적 위상 단서를 분리하고, 두 방향 transformer pose decoder를 통해 지상 특징을 공간적 쿼리로 활용하여 양방향 크로스 어텐션을 수행합니다. 또한, 세트 예측 전략을 채택하여 통합된 다중 작업 목표 하에 안정적인 3-DoF 회귀를 가능하게 합니다.

- **Performance Highlights**: VIGOR, KITTI 및 Ford Multi-AV 데이터셋을 이용한 실험 결과, CIPER는 제한된 시야 및 임의의 방향 조건에서도 특히 경쟁력 있는 성능을 보였습니다. 양방향 크로스 어텐션의 정렬 및 세트 예측 전략 덕분에 기존 방법보다 훨씬 더 높은 정확도를 달성하였으며, 다양한 실세계 응용에 적합한 강력한 기준선으로 자리잡을 가능성이 높습니다.



### M$^3$Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks (https://arxiv.org/abs/2606.05008)
Comments:
          We present an evaluation designed for multi-modal memory in multi-modal models

- **What's New**: 본 논문은 다중 모달 모델의 기억 능력을 체계적으로 평가하기 위한 M$^3$Eval이라는 새로운 프레임워크를 소개합니다. 기존의 연구는 시각적 인식과 추론에 초점을 맞춰 기억 메커니즘을 명확히 측정하지 못했으며, 이로 인해 기억의 다양한 차원들이 잘 이해되지 않았습니다. M$^3$Eval은 인지 심리학의 원리를 기반으로 하여 특정 기억 메커니즘을 고립시키는 비디오 기반 질문-응답(task) 과제를 설계하였습니다.

- **Technical Details**: M$^3$Eval은 기억의 주요 차원을 네 가지로 구분합니다: (1) 동시 입력으로부터 정보를 유지하는 능력, (2) 유사한 내용의 방해에 대한 강인성, (3) 섞인 사건을 일관된 표현으로 통합하는 능력, (4) 비디오 세그먼트 간의 추상적 속성을 추적하는 능력입니다. 이 평가 프레임워크는 비디오 이해 과제를 통해 다양한 모델을 광범위하게 평가하며, 각각의 평가 방식은 명확한 질문과 특정 실패 모드를 수치화하는 메트릭으로 설계되어 있습니다.

- **Performance Highlights**: 결과적으로, 다중 모달 모델은 병렬 비디오 스트림을 처리할 때 독립적인 표현을 유지하지 못하며, 이는 주의 혼동으로 인한 것으로 추측됩니다. 모델의 기억 능력은 인간 보다 시간이 엇갈린 정보를 조직할 때 더 약하며, 복잡한 속성을 추상화할 때도 기호 기억(symbolic memory) 능력이 훨씬 떨어집니다. 이 연구는 다중 모달 모델의 기억 한계를 드러내고 향후 시스템 설계에 대한 새로운 통찰을 제공합니다.



### Multi-Camera AR Guidance System for Surgical Instrument Handling and Assembly: Investigating Workload and Efficiency (https://arxiv.org/abs/2606.04992)
Comments:
          11 pages

- **What's New**: 이 논문은 수술 중 도구의 처리 및 조립에서 발생하는 인지적 요구를 충족시키기 위해 다중 카메라 6D 포즈 추정과 증강 현실(AR) 기술을 결합한 지원 시스템을 제안합니다. 이 시스템은 추가 마커 없이도 수술용 도구를 효과적으로 안내하고 사용자가 직접 선택할 수 있는 기능을 제공합니다. 또한, AR 안내 시스템이 제공하는 시각적 정보를 통해 수술 중 발생할 수 있는 비효율성을 줄이도록 설계되었습니다.

- **Technical Details**: 제안된 시스템은 수술 트레이와 조립 테이블을 감지하고, AR을 통해 시각적 지침을 제공합니다. 다양한 카메라 스트림을 융합하여 비접촉식으로 6D 포즈를 추정하며, 정적 카메라를 이용해 HMD(헤드 마운트 디스플레이)의 보정도 수행합니다. 사용자는 Unity에서 개발된 애플리케이션을 통해 도구의 위치를 실시간으로 확인할 수 있으며, 본 연구에서 사용된 데이터는 주로 합성 데이터를 기반으로 한 학습 모델을 사용합니다.

- **Performance Highlights**: 임상 평가에서는 시스템이 기존의 문서 기반 매뉴얼과 비교했을 때 작업 완료 시간을 21.3% 단축시켰습니다. 사용자는 AR 시스템을 사용할 때 전체적인 사용자 경험이 개선되었고, 불필요한 인지적 부담이 줄어들었다고 응답했습니다. 또한, 특히 경험이 부족한 스크럽 간호사가 이 시스템의 도움을 받아 도구를 더 잘 사용할 수 있었으며, 오류 빈도는 유사한 수준으로 유지되었습니다.



### Food-R1: A Unified Multi-Task Food Vision-Language Model with Reinforcement Learning (https://arxiv.org/abs/2606.04986)
- **What's New**: 본 논문에서는 Vision-Language Models (VLMs)를 활용한 음식 분석에 관한 새로운 접근 방식을 제안합니다. 특히, CalorieBench-80K라는 대규모 벤치마크를 새롭게 구축하여 정확한 칼로리 레이블과 식이 조언 주석을 제공합니다. 이는 Chain-of-Thought (CoT) 주석을 포함한 최초의 음식 이미지 벤치마크로, 모델의 Reasoning 능력을 향상시킵니다. 또한, Food-R1이라는 통합된 멀티태스크 VLM을 제안하여 모델의 다양한 기능을 강화합니다.

- **Technical Details**: 제안된 Food-R1은 멀티태스크 학습 패러다임을 이용하여 음식 이미지와 관련된 다양한 작업을 수행할 수 있도록 설계되었습니다. 이는 CoT 기반의 콜드 스타트 명령 튜닝 후, Group Relative Policy Optimization (GRPO)을 사용한 강화 학습(RL)을 통하여 Reasoning 능력과 성능을 개선합니다. 두 개의 훈련 단계로 구분되어 있어, 첫 번째 단계에서는 CoT 기반의 스켈레톤 학습을 통해 기본적인 기능을 습득하고, 두 번째 단계는 RL을 통해 성능을 더욱 높입니다.

- **Performance Highlights**: 실험 결과 Food-R1은 CalorieBench-80K 및 기타 대표 벤치마크에서 강력한 성능을 보이며 기존의 기준선 모델들을 일관되게 초과하는 성과를 보여줍니다. Food-R1은 음식 분류, 성분 인식, 레시피 생성, 영양 추정 등의 다양한 음식 관련 작업에서 일관된 개선을 이루었으며, 모델의 Reasoning 일관성 및 크로스-작업 일반화를 효과적으로 향상시킵니다.



### Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistanc (https://arxiv.org/abs/2606.04970)
Comments:
          53 pages, 14 figures

- **What's New**: 이번 논문에서는 실시간으로 단계별 절차적 작업을 지시하는 프로액티브 다중 모드 보조 시스템을 제안합니다. 이 시스템은 사용자에게 중단할 시점과 코칭 방법을 자율적으로 결정하며, 이를 실현하기 위해 새로운 대규모 Wearable-Egocentric 데이터셋인 EgoProactive를 출시합니다. 이 데이터셋은 명시적인 Out-of-Plan(OOP) 주석 및 회복 단계를 포함하고 있습니다.

- **Technical Details**: 제안된 시스템은 두 개의 모델 아키텍처로 구성되며, 하나는 장기 계획을 생성하고 업데이트하고, 다른 하나는 실시간 상호작용을 처리합니다. Duplex 상호작용 모델은 스트리밍된 egocentric 비디오를 소비하고 각 샘플링 시간에서 중단 여부를 결정하는 의사결정을 발행합니다. 이로 인해 단일 전진 패스에서 모든 작업을 수행하는 전통적인 방식에서 벗어나 모듈화된 구조로 기능을 분리합니다.

- **Performance Highlights**: 실험 결과, 훈련된 Llama-4 시스템은 강력한 상용 모델인 Claude Opus 4.6, Gemini 3.1 Pro, GPT 5.2에 비해 중재 품질을 크게 향상시켰습니다. OOP 회복에서 훈련된 모델이 높은 품질의 안내를 제공하며, 모든 6개 데이터셋에서 우수한 성과를 보여주었습니다. 또한, 계획 품질이 통제될 때, 훈련된 모델은 OOP 회복에 대한 큰 이익을 가져오는 것을 확인했습니다.



### Scene-Centric Unsupervised Video Panoptic Segmentation (https://arxiv.org/abs/2606.04925)
Comments:
          CVPR 2026. Oliver Hahn and Christoph Reich - both authors contributed equally. Code: this https URL Project page: this https URL

- **What's New**: 이번 논문에서는 비디오 파노프틱 세분화(Combining Video Panoptic Segmentation, VPS) 작업의 새로운 설정인 비지도 방법을 소개합니다. 기존의 비지도 장면 이해 연구는 주로 이미지 세분화(image segmentation)에 초점을 맞추었으나, 비디오 도메인은 아직 충분히 탐구되지 않았습니다.

- **Technical Details**: 제안하는 VideoCUPS는 장면 중심(scene-centric) 비디오에서 비 temporally consistent한 파노프틱 비디오 의사 라벨(pseudo-labels)을 생성합니다. 이를 위해 비지도 심도(unsupervised depth), 움직임(motion), 시각적 단서를 활용하며, 새로운 Video DropLoss 기법으로 이러한 의사 라벨을 기반으로 훈련하여 정확한 비지도 VPS 모델을 구축합니다.

- **Performance Highlights**: VideoCUPS는 모든 기준선(baseline)을 초월하며 강력한 라벨 효율적인 학습(label-efficient learning)을 보여줍니다. 새로운 평가 프로토콜과 네 가지 경쟁력 있는 기준선을 제시하여 비지도 VPS에 관한 향후 연구를 위한 튼튼한 기초를 제공합니다.



### Geometry-Aware Distillation for Prompt Tuning Biomedical Vision-Language Models (https://arxiv.org/abs/2606.04922)
Comments:
          Preprint. Code is available at this https URL

- **What's New**: 현재 시각-언어 모델(Vision-Language Models, VLMs)의 프롬프트 및 어댑터 기반 조정 방법은 의료 영상 분야에서 매력적입니다. 이는 임상 데이터의 민감성으로 인해 최적화된 백본 모델을 사용하고 제한된 주석을 사용해야 하기 때문입니다. 그러나 기존 방법들은 일반적으로 모든 비대상 클래스를 균등하게 잘못된 것으로 처리하여 임상적으로 중요한 클래스 관계를 무시하고, 한정된 감독 환경에서 불안정한 의사 결정을 초래합니다.

- **Technical Details**: 새로운 Omni-Geometry Knowledge Distillation (OGKD) 프레임워크는 교사 모델에 클래스 관계 구조를 주입하여 지향적인 목표를 생성합니다. 이러한 목표를 통해 두 가지 증류 손실을 개발하였으며, Global Geometry-Aware Distillation (GAD)은 글로벌 이미지 토큰에서 작동하고, Label-Guided Geometry Distillation (LGD)은 주목 패치 토큰에 동일한 구조를 적용합니다. 이는 섬세한 일치를 개선하는 데 도움을 줍니다.

- **Performance Highlights**: OGKD는 11개의 의료 데이터셋에서 포괄적인 실험을 통해 기존 최첨단 VLM 적응 방법에 비해 일관되게 1.7%-2.8%의 정확도 향상을 이끌어냈습니다. 또한, 여러 새로운 클래스로 일반화하는 데 강인성을 보이며, 다른 접근 방식보다 더 신뢰할 수 있는 예측 결과를 제공합니다. 이러한 개선 사항은 의학적 결정의 신뢰성과 정확성을 높이는 데 기여합니다.



### BreastGPT: A Multimodal Large Language Model for the Full Spectrum of Breast Cancer Clinical Routin (https://arxiv.org/abs/2606.04911)
- **What's New**: 이 논문에서는 유방암 관리의 각 단계인 스크리닝, 진단, 치료 계획을 지원하는 멀티모달 다단계 모델을 개발한 BreastStage와 BreastGPT를 소개합니다. BreastStage는 5개 영상 모달리티에서 1.86M 개의 질의응답(pair)을 수집하여 유방암 관리의 각 임상 단계에 적합한 데이터 작업을 지원합니다. 또한, BreastGPT는 이러한 데이터를 바탕으로 다양한 영상 모달리티를 하나의 아키텍처로 통합해 일관된 추론을 가능하게 합니다.

- **Technical Details**: BreastStage는 17개의 하위 데이터셋을 통합하여 5개의 영상 모달리티(유방촬영, BUS, MRI, WSI, CT)에 걸친 1.86M 개의 지침-응답 쌍을 포함한 대규모 멀티모달 지침 코퍼스입니다. BreastGPT는 모든 이 이미징 모달리티를 하나의 아키텍처에서 처리하며, 스테이지 조건에 따른 시스템 프롬프트를 채택해 단계별 추론 행동을 전환합니다. 이 모델은 두 가지 가지 비주얼 인코더와 해상도 인식 게이팅 모듈을 활용하여 다양한 모달리티의 이미지 스케일 차이를 메꿉니다.

- **Performance Highlights**: BreastGPT는 BreastStage-Bench에서 75.66%의 닫힌 질문 정확도와 89.92%의 열린 질문 점수를 달성하면서 기존 일반 목적 및 의료 전용 MLLM보다 평균 25% 이상의 성능 향상을 보여줍니다. 스크리닝, 진단 및 치료 계획을 위한 성과 향상이 각각 25%, 35%, 40%로 나타나며, 이는 임상적으로 유의미한 멀티모달 모델링의 필요성을 강조합니다. 이러한 결과는 임상적 기반의 모델들이 유방암 관리의 전반적인 워크플로우에 현실적으로 적합하다는 것을 시사합니다.



### CDPM-Align: Multi-Scale Guidance-Aligned Diffusion Pretraining for Robust Few-Shot Anatomical Landmark Detection (https://arxiv.org/abs/2606.04898)
Comments:
          Accepted MICCAI 2026

- **What's New**: 본 연구에서는 해부학적 랜드마크 탐지를 위한 다중 스케일 방향 정렬 조건부 확산 선행 학습 기법인 CDPM-align을 제안합니다. 기존의 메소드들이 정확성을 추구하는 것과는 달리, 우리의 방법은 낮은 주석 수로도 신뢰할 수 있는 결과를 도출하는데 초점을 맞춥니다. CDPM-align은 다양한 데이터 세트에서 강력한 표현학습을 통해 힘을 발휘하며, 이는 임상적 적용에 있어 더욱 안전하고 효율적인 진전을 가져올 것입니다.

- **Technical Details**: CDPM-align은 두 단계의 선행 학습 과정으로 구성됩니다. 첫 단계는 조건부 확산 확률 모델을 훈련시키고, 두 번째 단계에서는 정렬 손실(Alignment Loss)을 활성화하여 모델을 미세 조정하는 과정입니다. 이를 통해 모델은 데이터세트별 표현을 학습하며, 클래스 조건부 구조의 일관성을 유지하여 높은 정확도와 불확실성 집중도를 이룹니다.

- **Performance Highlights**: 세 가지 공공 벤치마크에서 10장 및 25장의 주석 이미지를 이용한 실험을 통해 CDPM-align의 성능을 검증하였으며, 기존의 감독학습(Supervised Learning), 자가 지도 학습(Self-Supervised Learning), 및 최신 기법들에 비해 일관된 성능 개선을 확인했습니다. 특히, 정확도, 예측 불확실성, 그리고 견고성의 세 가지 임상적 차원에서 개선된 결과를 보여 주었습니다.



### Hierarchical Space Partition for Surface Reconstruction (https://arxiv.org/abs/2606.04891)
Comments:
          Published in 2026 International Conference on 3D Vision (3DV)

- **What's New**: 이번 연구에서는 LiDAR 스캔의 한계를 극복하기 위한 새로운 plane assembling strategy를 제안합니다. 이 방법은 3D 비전과 컴퓨터 그래픽스에서 필수적인 정보가 부족할 때, 누락된 세부 정보를 복구하면서 모델의 간결함(compactness)을 유지하는 데 중점을 둡니다.

- **Technical Details**: 우리는 씬(scene)에서 추출된 평면(planes)을 세 가지 카테고리로 분류합니다: highly visible, barely visible, invisible로 나누어지며, invisible planes는 구조 분석을 통해 복구된 세부 정보의 부족을 나타냅니다. 이후, 각 평면은 우선순위(priority level)에 따라 성장(grow)하고, 이를 기반으로 공간이 점진적으로 분할됩니다. 최종적으로, min-cut 기반 최적화(min-cut-based optimization)를 통해 물이 새지 않는(watertight) 다각형 메시(polygonal mesh)를 생성합니다.

- **Performance Highlights**: 연구 결과는 공개 데이터셋(public datasets)에서 기존의 주류 접근법(mainstream approaches)보다 더 높은 효과성과 우수성을 보여줍니다. 이러한 결과는 제안된 방법이 복잡한 씬에서 누락된 세부 정보를 효과적으로 회복한다는 것을 입증합니다.



### HD-DinoMoE: A Class-Aware Hierarchical Dual Mixture-of-Experts Network for Scleral Anomaly Segmentation in Complex Acquisition Scenarios (https://arxiv.org/abs/2606.04888)
Comments:
          Submitted to Medical Image Analysis; 47 pages, 31 figures, 14 tables

- **What's New**: 본 연구는 전통 중국 의학(TCM)에서 영감을 받은 인공지능 안구 보조 진단 시스템(TAO)을 제안하며, 이는 픽셀 수준의 공막(surface anomaly) 분할(segmentation)을 중심으로 한다. HD-DinoMoE라는 클래스 인식 계층 구조의 혼합 전문가 네트워크를 통해 다중 분포와 다양한 변형의 이상 패턴을 효과적으로 구분할 수 있다. 또한, 새로운 다중 레이블 공막 이상 세분화 데이터셋(ML-SASD)을 구축하여 데이터와 성능 향상을 도모하고 있다.

- **Technical Details**: TAO 시스템은 안과 이미지의 세밀한 텍스처와 구조를 분석하기 위해 HD-DinoMoE를 채택하고 있다. 이 네트워크는 두 가지 유형의 사전 훈련된 가중치를 통합하여 클래스별 특징을 학습이 가능하도록 설계되었다. 또한 세 가지 단계의 백본 고정 라우팅 전략과 진보된 신뢰도 패널티 손실(PCP Loss)을 통해 높은 신뢰도를 유지하면서도 불필요한 거짓 긍정 오류를 줄인다.

- **Performance Highlights**: HD-DinoMoE는 ML-SASD-Mix 작업에서 평균 Dice 점수 72.11%와 평균 교차-연합 점수 58.44%를 달성하였으며, 경계 지역 로컬을 유지하고 반사광에 의한 거짓 긍정 오류를 효과적으로 관리하였다. 또한, 공공 SBVPI 데이터셋의 혈관 부문에서도 경쟁력 있는 일반화 성능을 보이며, 복잡한 환경에서도 안정적인 세분화 솔루션을 제공함을 나타내고 있다.



### DiverAge: Reliable Pluralistic Face Aging with Cross-Age Identity Relation Guidanc (https://arxiv.org/abs/2606.04881)
Comments:
          11 pages,10 figures, 5 tables

- **What's New**: 본 논문에서는 다층적 플루랄리스틱 얼굴 노화 프레임워크인 DiverAge를 제안합니다. DiverAge는 확산 오토인코딩(dispersion autoencoding) 기술에 기반하여 외형 수준의 다양성(appearance-level diversity)과 순서 수준의 신뢰성(sequence-level reliability)을 모두 유지합니다. 이 시스템은 각 연령군에서 다양한 후보를 생성할 뿐만 아니라, 서로 다른 연령대 간의 일관성을 보장합니다.

- **Technical Details**: DiverAge는 확산 디코딩(stochastic diffusion decoding) 기술을 통해 외형 수준의 다양성을 보존하고, 인퍼런스 타임 가이드인 Cross-age Identity Relation Regulator (CARR)를 사용하여 순서 수준의 신뢰성을 향상시킵니다. CARR는 실제 동일 신원의 교차 연령 쌍에서 추정한 Cross-age Identity Similarity (CIS) 우선 순위를 기반으로 하며, 이를 통해 세대 간의 과도한 정체성 이동을 억제합니다. 이 방식은 훈련 목표를 수정하지 않으며 추가적인 매개변수를 도입하지 않습니다.

- **Performance Highlights**: 실험 결과, DiverAge는 정체성 보존(identity preservation), 연령 정확도(age accuracy), 이미지 품질(image quality), 외형 수준의 다양성(appearance-level diversity)을 유지하면서도 순서 수준의 신뢰성을 개선하였습니다. 본 연구는 기존의 얼굴 노화 연구에서 간과되었던 순서적 정체성 유사성 패턴을 고려하여 신뢰성 있는 얼굴 노화의 생성 가능성을 높였습니다.



### MAOAM: Unified Object and Material Selection with Vision-Language Models (https://arxiv.org/abs/2606.04880)
Comments:
          Accepted to SIGGRAPH 2026 Conference. Project page: \href{this https URL}{here}

- **What's New**: 본 논문에서는 사용자 상호작용 방식을 통해 개체와 재료 선택을 지원하는 Mask Any Object And Material (MAOAM) 프레임워크를 제안합니다. MAOAM은 VLM(vision-language-model)과 세그멘테이션 헤드를 활용하여 사용자의 입력에 맞춰 픽셀 정확도의 마스크를 생성할 수 있습니다. 이는 기존의 개체 중심의 선택 방법과는 달리 재료 기반의 선택이 가능하게 하여, 다양한 상호작용 방식(클릭 및 텍스트)을 지원합니다.

- **Technical Details**: MAOAM은 여러 상호작용 모드를 통합하여, 사용자가 요청한 정보를 바탕으로 적절한 마스크를 생성하기 위해 VLM을 활용합니다. 이 모델은 사용자 입력(텍스트 또는 클릭)에 따라 이미지의 시각적, 공간적 관계를 인식하고, 이를 바탕으로 세분화된 마스크를 생성합니다. 또한, 재료 이해를 강화하기 위해 멀티태스크 목적을 채택하여 VQA(Visual Question Answering) 작업과 함께 학습합니다.

- **Performance Highlights**: 실험 결과, MAOAM은 다양한 개체와 재료, 상호작용 시나리오에 대해 정확하고 일관된 선택을 보여주었으며, 향상된 유연성으로 사용자가 요구하는 다양한 선택 기준을 지원합니다. 특히 관찰된 바에 따르면, 우리 모델은 텍스트와 클릭을 결합하여 추론할 경우 선택 성능이 개선되어 더욱 유연한 이미지 편집 작업이 가능해졌습니다. 이는 MAOAM의 다중 상호작용 모드를 통해 사용자 편의성을 더욱 극대화한 결과입니다.



### Recent Advances and Trends in Learning-based 3D Representations (https://arxiv.org/abs/2606.04871)
- **What's New**: 본 논문은 3D 표현의 발전을 다루며, 기계 학습에 기반한 접근 방식의 중요성을 강조합니다. 기존의 전통적인 표현 방식(예: 메쉬, 포인트 클라우드)과는 달리, 새로운 신경 및 원시 기반 표현(예: 3D Gaussian Splatting)이 나타나고 있으며, 이는 AR/VR, 자율주행, 로봇 항법 등 다양한 애플리케이션에서 기대되는 가능성을 제공합니다. 이 논문은 3D 표현의 포괄적인 조사를 제공하고, 명시적인 형식에서부터 연속적인 암시적 필드에 이르는 다양한 표현 방식의 장단점을 분석합니다.

- **Technical Details**: 논문은 3D 표현을 쿼리 가능한 모델로 정의하며 데이터의 효율적인 학습과 쿼리에 대한 중요성을 다룹니다. 명시적(surface) 및 암시적(volume) 표현을 구분하여 각 분류의 장단점과 실질적인 응용 사례를 설명합니다. 특히, 암시적 표현 방식이 고유한 구조를 유지함으로써 복잡한 물체 및 장면을 표현할 수 있는 이점을 지닌다는 점이 강조됩니다.

- **Performance Highlights**: 3D 표현의 선택은 정확성, 효율성, 미분 가능성, 편집 가능성, 그리고 다운스트림 호환성 측면에서 중요한 거래를 수반합니다. 기존의 명시적 표현 방식은 여전히 산업 분야에서 널리 사용되지만, 새로운 암시적 표현 방식은 더 나은 압축성과 효율적인 최적화를 제공하여 기존의 한계를 넘어서고 있습니다. 논문은 이러한 새로운 표현 방식들이 3D/4D 워크플로우를 어떻게 근본적으로 변화시키는지를 설명하며, 빈 공간 처리 및 동적 장면 분석과 같은 여러 실질적인 응용 프로그램에서의 발전 가능성을 제시합니다.



### IRIS-GAN: Staged Specialist Detection of Deepfake Faces (https://arxiv.org/abs/2606.04863)
Comments:
          20 pages, 10 figures

- **What's New**: IRIS-GAN은 교차 생성기 이동(cross-generator shift) 하에서 합성 얼굴 이미지를 위한 전문 감지기로 소개된다. 기존의 합성 이미지 감지가 아닌 GAN(Generative Adversarial Network)으로 생성된 얼굴 이미지에 집중하여, 점진적인 훈련 방법을 통해 높은 정확도를 달성했다. 최종 모델은 99% 이상의 가짜 감지율을 기록하며, 외부에 있는 실제 얼굴 데이터셋에서 98.9%의 정확성을 보인다.

- **Technical Details**: IRIS-GAN은 점진적 노출(curriculum)를 통해 GAN 가상의 얼굴 이미지 감지 문제를 해결하고, 여러 GAN 패밀리에서 강력한 성능을 달성하기 위해 훈련 단계를 조정한다. Grad-CAM 분석을 통해 생성기 의존적인 공간 응답 패턴을 시각화하며, 이는 보조 분류기에서 유용한 정보를 제공한다. 또한 비-GAN 딥픽스(non-GAN deepfakes)에 대해서도 일부 감지력을 보여주며, 외부 테스트에서 전문가 감지기의 특수성을 입증한다.

- **Performance Highlights**: 최종 모델은 GAN 패밀리에서 가짜 감지율 99%을 초과하여, 새로운 GAN에 대한 일반화 능력을 크게 향상시켰다. 또한 Grad-CAM 기법을 활용하여 감지기의 공간 응답을 분석하고, 이 분석 결과는 감지 성능의 개선에 도움을 준다. IRIS-GAN은 합성 이미지 감지 분야에서 새로운 전문 감지기로서의 입지를 다짐으로써, GAN 이미지 분석의 오차를 줄이고 실제 적용 가능성을 높인다.



### MusaCoder: Native GPU Kernel Generation with Full-Stack Training on Moore Threads GPU (https://arxiv.org/abs/2606.04847)
- **What's New**: MusaCoder는 CUDA 및 MUSA 백엔드에서 네이티브 GPU 커널 생성을 위한 전체 스택 훈련 프레임워크를 제안합니다. 이 프레임워크는 점진적인 커널 지향 데이터 합성, 다양성 보존을 위한 거부 정제, 실행 피드백 기반 강화 학습(Execution-Feedback Reinforcement Learning, RL)을 결합하여 성능을 최적화합니다. MusaCoder는 또한 GPU 커널 생성을 안정화하기 위해 다양한 보완 메커니즘을 도입했습니다.

- **Technical Details**: MusaCoder는 매우 낮은 초기 성공률을 극복하기 위해 세 단계의 데이터 합성 파이프라인을 설계했습니다. 이 파이프라인은 멀티태스크 감독 세밀 조정(Multi-task Supervised Fine-Tuning, SFT)과 다양성 보존 거부 샘플링 정제(Diversity-Preserving Rejection Sampling Fine-Tuning, RFT)를 통해 모델을 훈련시킵니다. 최종적으로, 실행 피드백을 기반으로 한 강화 학습 문제로 커널 생성을 공식화하고, 이를 위해 분산 실행 샌드박스인 MooreEval을 활용합니다.

- **Performance Highlights**: KernelBench와 MUSA로 포팅된 변형에서의 실험 결과, MusaCoder는 주요 오픈 소스 및 독점 모델들보다 현저히 우수한 성능을 보여주었습니다. 특히, 27B 모델 변형은 커널 정확성과 실행 속도에서 최첨단 성능을 기록했습니다. 이러한 성과는 하드웨어 인지 GPU 커널 생성을 가능하게 하며, 자동 커널 합성을 위한 기반을 제공합니다.



### 3D Temporal Analysis for Autism Spectrum Disorder Screening During Attention Tasks (https://arxiv.org/abs/2606.04836)
- **What's New**: 본 연구에서는 정신 지체와 언어 장애를 겪고 있는 7세에서 12세 아동을 위한 새로운 3D 시간적 분석 프레임워크를 제안합니다. 기존의 2D 분석 방법은 ASD 행동의 특징인 공간 변위 패턴을 포착하지 못해 한계가 있었으나, DECA(상세 표현 캡처 및 애니메이션) 기반의 최신 기술을 활용하여 더 나은 정확성을 보여주고 있습니다. 연구는 39명의 아동 데이터를 사용하여 83.9%의 정확성을 기록하며, 이는 기존 2D 방법보다 상당히 향상된 결과입니다.

- **Technical Details**: 제안된 방법론은 비디오 시퀀스를 활용한 ASD 스크리닝을 위해 3D 특징 추출과 LSTM 및 GRU 기반의 순차적 행동 패턴 모델링으로 구성되어 있습니다. DECA 프레임워크는 이미지에서 3D 얼굴 특징 벡터와 헤드 포즈 매개변수를 히타된 후 기계 학습 알고리즘을 적용하여 아동의 얼굴 표현과 머리 움직임의 변화를 포착합니다. 이는 공간 이동 패턴을 더욱 효과적으로 추적할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 연구에서 제안한 모델은 3D 헤드 포즈 특징과 얼굴 특징을 조합하여 최종적으로 84.6%의 정확성을 달성했습니다. 이는 단일 모달 접근 방식에 비해 두드러진 성능 개선으로, ASD 선별 과정의 객관성을 더욱 높이는 데 기여합니다. 또한, 이 연구 결과는 이전 진단 도구의 한계를 극복하고, 개인화된 지원을 제공하기 위한 기초를 마련하는 데 중요한 역할을 합니다.



### OA-CutMix: Correcting the Label Bias of CutMix (https://arxiv.org/abs/2606.04820)
- **What's New**: 이번 논문에서는 CutMix의 레이블 할당 방식이 정당한 가정에 기반하지 않음이 밝혀졌습니다. 기존의 CutMix는 패치의 면적을 근거로 레이블을 할당하지만, 패치가 배경에 위치할 경우, 객체에 대한 가시적인 정보가 반영되지 않는 문제가 있습니다. 이를 해결하기 위해 Object-Aware CutMix (OA-CutMix)가 제안되었고, 이는 사전 계산된 분할 마스크를 사용하여 가시적인 객체 면적에 비례해 레이블을 재조정합니다.

- **Technical Details**: OA-CutMix는 CutMix의 이미지 믹싱 절차를 변경하지 않고, 레이블의 가중치만 수정합니다. SAM3을 활용하여 각 훈련 이미지에 대한 분할 마스크를 사전에 계산하고, 믹싱 시에는 각 이미지의 가시적 객체 픽셀 수를 계산하여 레이블을 비례적으로 할당합니다. 이를 통해, OA-CutMix는 동적 방법과 달리 훈련 중에 추가적인 비용이 발생하지 않고, 효율성을 유지합니다.

- **Performance Highlights**: OA-CutMix는 10개 이상의 정적 및 동적 믹싱 방법과 비교했을 때, 모든 작업에 걸쳐 일관되게 높은 정확도를 기록했습니다. 특히, 작은 객체에 대한 성능 개선이 두드러졌으며, 레이블 수정만으로도 기존의 이미지 믹싱 알고리즘을 수정한 방법의 성능을 초과할 수 있다는 것을 보여주었습니다.



### Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation? (https://arxiv.org/abs/2606.04811)
- **What's New**: 이번 연구에서는 Dream.exe를 도입하여 비디오 생성 모델의 물리적 실행 가능성을 평가하는 최초의 기준을 제시했습니다. 기존의 시각적 품질 평가에 의존하지 않고, 시뮬레이션 내에서의 과제 성공 여부를 주요 기준으로 설정했습니다. 이 연구는 실제 로봇 조작에서 결과를 정량화하여 물리 법칙이 내재화된 모델을 평가하는 새로운 방법을 제안합니다.

- **Technical Details**: Dream.exe는 초기 장면 이미지와 자연어 과제 설명을 기반으로 조작 비디오를 생성하도록 모델을 요구합니다. 이후 생성된 비디오의 시각적 안정성, 물리적 타당성 및 과제 이행 여부를 평가하는 세 가지 트랙에서 평가가 진행됩니다. 이 과정에서 2D 엔드 이펙터 운동을 3D 궤적으로 변환하는 비디오-실행 파이프라인을 개발하여, 물리 기반 시뮬레이터에서의 구체적인 성공 점수를 제공합니다.

- **Performance Highlights**: 실험 결과, 여러 모델이 유의미한 실행 성공률을 보였으며, 이는 인터넷 규모의 데이터에서 학습된 생성 모델이 유용한 물리적 지식을 암시함을 나타냅니다. 그러나 시각적 품질과 물리적 실행 가능성 간의 상관관계는 약한 것으로 나타났으며, 로봇 전용 정책 모델이 일반 생성 모델보다 일관되게 더 나은 성과를 내는 것은 아니었습니다. 이러한 결과는 비디오 생성 모델의 물리적 실행 가능성을 평가하는 새로운 장을 여는 중요한 성과로 볼 수 있습니다.



### NoRA: Evaluating Grounded Reasonableness in Visual First-person Normative Action Reasoning (https://arxiv.org/abs/2606.04806)
- **What's New**: 본 논문에서는 AI와 agentic 시스템이 사회적 환경에서의 규범적 능력(normative competence)을 갖추어야 함을 강조합니다. 기존의 접근 방식들은 단순히 텍스트에서 규범적 판단을 평가하거나 제한된 후보 행동 중 선택하는 방식이었습니다. 그러나 이는 실질적으로 충분하지 않으며, 에이전트는 주어진 상황에서 적절한 행동을 스스로 식별해야 합니다. 이를 위해 NoRA라는 새로운 비주얼 벤치마크를 소개하고, 에이전트가 다음 행동을 생성하고 그 정당성을 평가하도록 요구합니다.

- **Technical Details**: NoRA는 각기 다른 1,420개의 주석이 달린 비디오 클립을 포함하며, 후보 행동을 생성하고 이를 사실-이유-행동 지원 그래프를 통해 정당화해야 합니다. 연구는 12개의 멀티모달 시스템을 순서대로 평가하며, 행동 정렬, 사실적 기반 및 지원 결합을 통해 평가합니다. 이러한 프레임워크는 AI 시스템의 규범적 결정력을 촉진시키기 위해 철학적 기준에 근거하고 있으며, 기존의 MCQ 형식에서 벗어나 행동의 적절성을 평가할 수 있게 합니다.

- **Performance Highlights**: 우리의 연구에서, 현재의 비전 언어 모델(VLM)은 자주 그럴듯한 다음 행동을 생성하고 관련 장면 사실을 회복하지만, 적절한 지역적 지원에 선택된 행동을 결합하는 데는 어려움을 겪고 있습니다. 구조화된 프롬프트를 사용할 경우, GPT-5.2는 GPT-5.4의 grounded reasonableness 점수의 68.6%에 도달합니다. Gemini-3-Flash는 Gemini-3.1-Pro 점수의 75.6%에 도달하여 NoRA가 실제적인 규범적 결정력을 개선하는 데 기여하고 있음을 보여줍니다.



### Fast Cubical Persistent Homology on 2D and 3D Images via Union-Find, Pruning, and Lookup Tables (https://arxiv.org/abs/2606.04801)
- **What's New**: 이번 연구에서는 V-필터레이션에서 2D 및 3D 이미지의 큐비컬 지속성(Cubical Persistence)을 매우 효율적으로 계산하는 Flash Cubical을 소개합니다. 이 구현은 세 가지 핵심 아이디어를 기반으로 하고 있으며, 특히 시간과 메모리 비용 측면에서 가장 효율적인 방법입니다. 또한, V-필터레이션 큐비컬 복합체에 중점을 두지만, 이 아이디어는 T-필터레이션으로 일반화할 수 있으며 다른 복합체에 대한 유망한 방향도 제시합니다.

- **Technical Details**: 큐비컬 지속성 계산을 위한 핵심 아이디어는 첫째, 최고 차원의 지속성을 union-find와 이중성(duality)을 통해 계산할 수 있다는 점입니다. 둘째, 특정 엣지를 제거하여 union-find의 구현을 빠르고 효율적으로 하는 방법을 제안합니다. 셋째, 큐비컬 복합체의 규칙성을 활용하여 로컬 정보를 사전 계산하는 조회 테이블(lookup table)을 사용하는 것입니다. 이러한 접근 방식은 런타임에서 로컬 정보를 추가로 계산할 필요를 줄입니다.

- **Performance Highlights**: 이 연구의 C++ 구현은 2D 및 3D의 합성 데이터(synthetic data)와 실제 데이터(real data)에 대한 성능을 검증하였고, 각 아이디어가 결합되어 수행된 결과를 보여주었습니다. 이를 통해 큐비컬 지속성의 계산을 기존 방법보다 유의미하게 개선할 수 있음을 입증하였습니다. 이 구현은 https://github.com/T-prog123/FlashCubical/tree/main에서 사용 가능합니다.



### Crafting Your Evolving Dreams: Concept-Incremental Versatile Customization (https://arxiv.org/abs/2606.04797)
Comments:
          Accepted to Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

- **What's New**: 이번 연구에서는 Continually Customizable Diffusion Model (CCDM)라는 새로운 모델을 제안하여 사용자 맞춤형 개념의 증가적인 커스터마이징을 가능하게 합니다. 기존 Custom diffusion models (CDMs)의 한계를 극복하기 위해, 우리가 개발한 모델은 개념의 지속적인 학습을 지속하면서도 새로운 기능을 추가할 수 있도록 설계되었습니다. 특히, catastrophic forgetting과 concept neglect 문제를 해결하기 위한 두 가지 혁신적인 전략이 포함되어있습니다.

- **Technical Details**: CCDM은 attribute-decoupled LoRA (AD-LoRA) 모듈과 relevance-guided AD-LoRA 집합 전략을 통해 이전 개념의 특성을 보존하며 학습 중 발생하는 중요한 경량화 요소들을 관리합니다. 매개변수의 관련성을 평가하여, 새로운 작업을 위한 지속적인 학습을 촉진하고 사용자 지정 조건에 부합하는 다중 개념 합성을 위한 제어 가능한 지역 맥락 합성 전략을 통해 동적인 응집력을 부여합니다.

- **Performance Highlights**: 실험 결과, CCDM은 기존의 방법들과 비교하여 знач한 향상을 보여주었으며, 다양한 CIVC 작업에 대처할 수 있는 우수한 성능을 입증하였습니다. 특히, 유의미한 개념 간의 상관관계를 활용하여 커스터마이징 성능을 향상시키는 동시에 전체 매개변수를 35% 줄이는 데 성공하였습니다.



### A Pathology Foundation Model for Gastric Cancer with Real-World Validation (https://arxiv.org/abs/2606.04792)
- **What's New**: GRACE라는 새로운 위암 전용 기초 모델이 개발되었습니다. 이 모델은 48,364개의 주로 H&E 염색된 전 슬라이드 이미지와 37,493명의 환자로부터 수집된 다기관 데이터를 기반으로 합니다. GRACE는 28개의 임상적으로 중요한 작업을 평가한 결과, 기존의 범암성(Pathology Foundation Models, PFMs) 모델들에 비해 일관되게 우수한 성능을 보였습니다.

- **Technical Details**: GRACE는 위암의 특수한 형태를 반영하기 위해 고안된 계속적인 사전 훈련 전략을 적용하였습니다. Virchow2 모델을 기반으로 한 GRACE는 저순위 적응(low-rank adaptation) 및 DINO 프레임워크 내에서 조정되었습니다. 전체 데이터의 35% 이상이 새로운 사례의 강건성을 지원하기 위해 전향적으로 수집되었습니다.

- **Performance Highlights**: GRACE는 평균적으로 0.9188의 macro-AUC를 달성하며, 진단 정확도는 82.0%에서 89.9%로 향상되었습니다. AI의 지원으로 진단 소요 시간은 14.9% 단축되었고, 진단 신뢰도는 9.0% 증가하였습니다. 이 결과 GRACE는 위암 진단 워크플로우를 효과적으로 개선할 수 있는 잠재력을 보여줍니다.



### Z-FLoc: Zero-Shot Floorplan Localization via Geometric Primitives (https://arxiv.org/abs/2606.04788)
- **What's New**: 이번 연구에서는 리트레이닝(curtain retraining) 없이 새로운 환경에서도 일반화할 수 있는 제로샷(zero-shot) 바닥 계획서(floorplan) 로컬라이제이션 방법을 제안했습니다. 이 방법은 인류가 만든 환경에서 광범위하게 존재하는 기하학적 원시(primitives)인 선(line)과 원(circle)을 활용하여 시각적 모양 변화에 영향을 받지 않는 구조적 제약을 제공합니다. 실험 결과, 이 접근법이 기존의 데이터 기반 학습 방법보다 성능이 우수하다는 것을 보여주었습니다.

- **Technical Details**: 제안된 방법은 단일 카메라의 RGB 이미지 시퀀스를 기반으로 2D 바닥 계획서(floorplan)와의 교차 모달 매칭(cross-modal matching) 문제를 다룹니다. 이를 위해 BEV(떨어진 시점의 보기, Bird’s Eye View) 맵을 재구성하고, 여기서 선과 원의 원시를 추출합니다. 후보 유사도 변환(similarity transformation)은 랜덤 샘플을 기준으로 형성하며, 각 원시 집합에서 독립적인 자세 가설을 생성합니다.

- **Performance Highlights**: 시뮬레이션된 데이터와 실제 데이터셋 모두에서 실험을 진행한 결과, 제안된 방법이 특정 환경에 맞추어 훈련된 기존의 최첨단 학습 기반 방법보다 일관되게 우수한 성능을 보였습니다. 특히, 모든 실험에서 사용된 하이퍼파라미터는 고정되어 있어, 다양한 환경에서 안정적으로 일반화될 수 있음을 입증했습니다.



### NextMotionQA: Benchmarking and Judging Human Motion Understanding with Vision-Language Models (https://arxiv.org/abs/2606.04773)
Comments:
          23 pages, 8 figures, 9 tables

- **What's New**: 본 논문은 NextMotionQA라는 새로운 평가 벤치마크를 소개합니다. 이 벤치마크는 비전-언어 모델(vision-language models, VLMs)을 활용하여 반자동으로 데이터셋을 구축하고 전문가의 검증을 받습니다. NextMotionQA는 여러 선택 질문 응답, 비디오 자막 생성, 세밀한 오류 수정이라는 세 가지 상호 보완적인 작업을 포함합니다. 이를 통해 기존 벤치마크의 한계를 극복하고 인공지능 모델의 실패 지점을 진단할 수 있는 기초를 제공합니다.

- **Technical Details**: NextMotionQA는 세 가지 핵심 의미 축(semantic axes)과 세 가지 작업 복잡도(level)에 따라 체계적으로 구조화되어 있습니다. 각 작업은 다양한 난이도를 가지고 있으며, VLMs를 활용하여 다층적 평가를 가능하게 합니다. 평가 결과, 전통적인 단일 작업 평가에서는 보이지 않던 VLM의 주요 능력 격차와 약점을 발견하였습니다. 이는 VLM이 평어의 일반적인 기준에서 전문가 평점과 높은 일치를 보이지만, 세밀한 부분 수준 평가에서는 상당한 감소를 나타낸다는 사실을 드러냅니다.

- **Performance Highlights**: 본 연구는 12개의 대표적인 VLM을 광범위하게 평가하여 기존의 벤치마크가 겪고 있는 문제점을 드러내었습니다. VLM은 대략적인 기준(Cohen's kappa=0.70)에서는 전문가 평가와 잘 일치하나, 세밀한 평가(코사인 kappa=0.10)에서 급격히 성능이 저하되는 경향을 보였습니다. 이는 VLM의 평가 방식이 강한 영역에서는 유효하지만, 그 한계를 분명히 하는 결과를 도출합니다. NextMotionQA를 통해 미래의 AI 모델 개선에 기여할 것으로 기대됩니다.



### Coarse-to-fine Hierarchical Architecture with Sequential Mamba for Brain Reconstruction (https://arxiv.org/abs/2606.04772)
- **What's New**: 이 연구에서는 CHASMBrain이라는 새로운 위계적 2단계 프레임워크를 제안하여 이미지에서 fMRI로의 인코딩을 수행합니다. 이 프레임워크는 Mamba 디자인을 활용하여 글로벌 의미 토큰과 로컬 공간 패치를 명확히 구분하여 처리합니다. 제안된 모델은 자연 장면 데이터셋에서 Pearson 상관관계 0.429 및 MSE 0.261을 달성하며, 기존의 여러 모델을 초월한 성능을 보여줍니다.

- **Technical Details**: CHASMBrain은 1단계에서 ROI 수준의 노이즈 제거된 활성화를 예측하고, 2단계에서 이러한 활성화를 보완하여 복잡한 복셀 수준 예측을 만듭니다. 이 과정에서 Mamba-VAE를 사용하여 세부적인 신호 회복을 지원합니다. 또한, 두 개의 스트림을 통해 글로벌 및 로컬 정보를 통합하여, 더 정교한 시각적 인식이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다수의 기준선 모델들보다 뛰어난 성능을 보였으며, 패치 스트림과 CLS 스트림의 비대칭 전문화가 인과 관계를 통해 구체화되었습니다. 교차 주제 전이 실험을 통해, 학습된 베이스 모델이 개인간에 일반화되는 방식도 확인되었으며, 이는 모형이 개인에게 독립적인 시각적 표현을 포착한다는 것을 암시합니다.



### Do Foundation Models See Biology? Evaluating Attention Coherence with Spatial Transcriptomics in Glioblastoma (https://arxiv.org/abs/2606.04764)
- **What's New**: 본 연구는 병리학 기초 모델이 생성하는 주의 지도(attention maps)가 실제 생물학을 포착하는지를 평가하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 병리학 모델을 지나치지 않고도 정량적인 평가를 가능하게 하여, 주의 지도의 해석 가능성에 대한 문제를 해결하고자 합니다. 우리는 공간 전사체학(spatial transcriptomics) 데이터를 활용하여 5가지 병리학 기초 모델과 ResNet50 베이스라인에 대한 평가를 수행했습니다.

- **Technical Details**: 우리는 171명의 환자에서 얻은 H&E WSI를 사용해 다섯 가지 분자 변화를 예측하도록 모델을 훈련했습니다. 모델의 성능은 TCGA-GBM 데이터셋을 통해 독립적으로 검증하였으며, 주의 가중치는 여러 인코더를 통해 계산되었습니다. 모든 인코더는 동일한 패치를 처리하고, 공간 전사체학 데이터를 활용하여 주의 지도가 실제 생물학적 프로그램과의 상관 관계를 유지하는지를 평가했습니다.

- **Performance Highlights**: 실험 결과, 주의 가중치가 하향식으로 정리된 경향성은 나타나지만, 주의 지도가 반드시 생물학적 의미 있는 지역에 초점을 맞추는 것은 아니었습니다. 내부 교차 검증에 기반한 인코더 성능 순위는 외부 검증에서 뒤집힌 결과를 보이면서, 시각적 검사만으로는 충분히 평가할 수 없음을 보여주었습니다. 이 프레임워크는 모델이 학습한 내용을 정량적으로 이해하고 비교하는 기반을 제공하여, 깊은 학습 모델의 해석 가능성을 한층 진전시킵니다.



### Physics-Informed Video Generation via Mixture-of-Experts Latent Alignmen (https://arxiv.org/abs/2606.04737)
- **What's New**: 이 연구에서는 비디오 생성 모델에서 물리적 일관성을 높이기 위해 새로운 프레임워크인 PILA(Physics-Informed Latent Alignment)를 제안합니다. PILA는 얼린 흐름 매칭(dynamics) 모델의 동적 구조에 물리 기반의 유도 정보를 주입하여 실제 세계의 운동 규칙을 반영할 수 있도록 합니다. 연구진은 물리적 속성을 조직한 작업용 물리 속성 은행을 구축하여, 비디오 생성의 성능과 현실성을 개선하고자 하였습니다.

- **Technical Details**: PILA는 고정된 흐름 매칭(dynamics)에서 얼어있는 생성기를 활용하여, 운동, 압력, 밀도, 온도 및 상 변화 등을 의미하는 물리적 속성을 주입합니다. 앵커 필드 추정(Anchored Field Estimation, AFE) 기법을 사용해 관측 가능한 동작을 바탕으로 불완전한 프록시를 완성합니다. 그리고 레이블 우선 전문가 라우팅(Label-Prior Masked Expert Routing, LPMER)을 통해 각 분야의 전문가를 선택하여 물리적 관계에서 유도된 신뢰성을 갖춘 업데이트를 적용합니다.

- **Performance Highlights**: PILA는 Wan 2.1-1.3B에 대해 단계적 어댑터 훈련을 수행한 후, Wan 2.2-14B에 학습된 어댑터를 직접 이전함으로써 VBench-2.0, VideoPhy-2 및 PhyGenBench에서 비주얼 품질과 물리적 타당성에서 최첨단 결과를 달성하였습니다. 이는 기존 비디오 생성 모델에서의 물리적 일관성 문제를 해결하는 데 중요한 기여를 하며, 향후 에이전트 훈련 및 물리 과정 시각화와 같은 다양한 응용 분야에서도 활용될 수 있을 것입니다.



### StrokeTimer: Robust Representation Learning for Ischemic Stroke Onset-Time Estimation from Non-contrast C (https://arxiv.org/abs/2606.04722)
Comments:
          Early accepted at MICCAI 2026

- **What's New**: 이번 연구에서는 StrokeTimer라는 자동화된 프레임워크를 제안합니다. StrokeTimer는 급성 허혈성 뇌졸중의 발병 시점을 추정하는 데 필요한 데이터를 자동으로 처리하고 분석합니다. 이 시스템은 와의 불균형한 데이터를 다루면서 또한 질병의 미세한 패턴 변화까지 포착할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: StrokeTimer는 두 단계의 구조로 이루어져 있습니다. 첫 번째 단계에서는 스타일 불변의 의미 표현을 학습하는 자가 지도 학습(self-supervised learning) 모듈이 포함되어 있습니다. 두 번째 단계에서는 에너지 유도 대조적 평균 이동(Energy-guided Contrastive Mean-Shift) 모듈을 통해 교차-엔트로피 손실을 최적화하고, 클래스 간 미세한 차이를 처리합니다.

- **Performance Highlights**: 대규모 다기관 NCCT 데이터셋에서 StrokeTimer는 매크로 AUC 0.69, 매크로 F1 점수 0.57을 기록하며 최선의 기준선보다 거의 50% 향상된 결과를 보여줍니다. 연구 결과는 StrokeTimer가 치료 결정 과정에서 유용한 도구가 될 수 있음을 입증합니다.



### Data Efficient Complex Feature Fusion Network For Hyperspectral Image Classification (https://arxiv.org/abs/2606.04710)
Comments:
          10 pages, 3 figures

- **What's New**: 이 연구는 하이퍼스펙트럴 이미지 분류를 위한 데이터 효율적인 새로운 변형 모델인 DE-CFFN을 제안합니다. DE-CFFN은 원래의 두 개의 스트림 구조를 유지하면서, 실수 값 신경망(RVNN)과 복소수 값 신경망(CVNN)을 사용하여 특징을 추출합니다. 주요 기여는 차원 축소에서 주성분 분석(Principal Component Analysis, PCA)을 대신하여 요인 분석(Factor Analysis, FA)을 사용하고, 3D 합성곱 계층의 필터 수를 점진적으로 줄여 모델의 복잡성을 낮춘 점입니다.

- **Technical Details**: DE-CFFN은 실제-복소수 병렬 스트림을 통해 공간, 스펙트럼, 주파수 도메인 피처를 동시에 캡처합니다. 요인 분석을 통해 차원 축소가 이루어지고, 입력 하이퍼스펙트럴 큐브는 겹치는 15×15 크기의 패치로 나누어집니다. RVNN은 3개의 연속적인 3D 합성곱 계층을 통해 필터 크기를 줄이며 기능을 추출하고, CVNN은 주파수 도메인에서 FFT를 사용하여 복소수 입력을 처리합니다.

- **Performance Highlights**: DE-CFFN은 Pavia University와 Salinas 데이터셋에서 CFFN과 유사한 수준의 분류 성능을 달성하면서도 모델 크기, 메모리 소비, 추론 지연 시간을 각각 72%, 75%, 14% 줄이는 데 성공했습니다. 이는 DE-CFFN이 실시간 하이퍼스펙트럴 이미징 애플리케이션에 적합함을 의미합니다. 이러한 성과는 특히 저라벨 조건에서도 모델의 효율성을 개선하는 데 기여할 것으로 기대됩니다.



### ReConFuse: Reconstruction-Error Guided Semantic Fusion for AI-Generated Video Detection (https://arxiv.org/abs/2606.04706)
- **What's New**: AI로 생성된 비디오의 현실성이 향상되면서 허위 정보 및 콘텐츠 진위 문제에 대한 우려가 커지고 있습니다. 이 논문에서는 AI로 생성된 비디오를 감지하기 위한 검증 가능한 방법으로, 공간적 아티팩트와 시간적 동적 특성을 포착하는 reconstruction error(재구성 오차)에 주목하고 있습니다. 특히, 새로운 ReConFuse 프레임워크를 제안하여, 비디오의 재구성 오차와 다중 프레임의 의미적 특징을 결합하여 영상 수준의 AI 생성 비디오 감지를 수행합니다.

- **Technical Details**: ReConFuse는 사전 훈련된 WF-VAE를 사용하여 입력 비디오의 프레임 단위 재구성 오차를 추출합니다. 이후, 이 재구성 오차를 다중 프레임의 의미적 특징과 공간적으로 정렬하고, Mamba 기반 모듈을 통해 시간적 진화를 모델링하여 비디오 수준의 분류를 수행합니다. 이는 저수준 재구성 불일치를 고수준 의미적 정보와 통합하여 감지의 신뢰성과 일반화를 개선하는 것을 목표로 합니다.

- **Performance Highlights**: 여러 심층 생성 모델과 평가 설정에서 ReConFuse의 효과성과 일반화 능력을 검증하는 실험을 수행하였습니다. 결과적으로, 재구성 오차는 AI로 생성된 비디오 감지에서 유용한 법의학적 증거를 제공하며, 특히 실제 비디오와 생성된 비디오 간의 구별이 가능합니다. 이러한 방식은 기존 기술에 비해 성능이 크게 향상된 것을 보여주었습니다.



### Enhancing MedSAM with a Lightweight Box Predictor for Medical Image Segmentation (https://arxiv.org/abs/2606.04705)
- **What's New**: 이 논문은 MedSAM 아키텍처에 경량의 Box Predictor 모듈을 통합하여 강화된 세분화 프레임워크를 제안합니다. 단일 사용자 클릭으로부터 대략적인 경계 상자를 추정하여 포인트 프롬프트의 모호함을 줄이는 방식으로 작동합니다. 또한, Box Predictor는 별도의 훈련 과정을 거쳐 MedSAM에 통합되어 효율성을 유지하면서도 정확성을 높입니다.

- **Technical Details**: 제안된 프레임워크는 Box Predictor라는 경량 모듈을 포함하여 사용자가 제공하는 단일 포인트 프롬프트를 경계 상자로 변환합니다. 이 모듈은 MedSAM의 Prompt Encoder와 Mask Decoder를 향상하기 위한 공간적 선행정보를 제공합니다. 박스 예측기는 처음에 독립적으로 훈련된 후 MedSAM에 통합되어 훈련 효율성과 성능을 동시에 개선하도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 다양한 해부학적 구조와 이미징 도메인에 걸쳐 분할 정확도와 강건성을 향상시킵니다. FLARE22, BRISC, BUSI, LungSegDB의 네 가지 데이터세트에서 0.89(BUSI), 0.93(FLARE22), 0.88(BRISC), 0.98(LungSegDB)의 Dice 점수를 기록했습니다. 본 연구는 제한된 라벨 데이터 상황에서도 MedSAM의 강력한 성능을 유지하는 방법을 입증했습니다.



### Benchmarking Living-Screen-Native GUI Agents on Short-Video Platforms (https://arxiv.org/abs/2606.04701)
Comments:
          preprint

- **What's New**: 이 논문에서는 Living-Screen-Native GUI 에이전트라는 새로운 개념을 도입하고, 짧은 비디오 플랫폼에서 이 개념을 실현하기 위한 첫 번째 벤치마크인 LivingScreen을 소개합니다. 기존의 에이전트들이 정적인 화면만을 가정할 때, LivingScreen은 동적으로 변화하는 화면에 대해 작업을 수행할 수 있도록 설계되었습니다. 이 새로운 설정은 에이전트가 비디오 콘텐츠를 능동적으로 관찰하고 상호작용할 수 있는 능력을 제공하여 정보 획득 방식을 혁신적으로 변화시킵니다.

- **Technical Details**: LivingScreen은 높은 충실도를 가진 브라우저 기반의 환경, 세 가지 수준의 작업 세트, 그리고 작업의 정확성과 정보 효율성을 평가하는 메트릭으로 구성됩니다. 에이전트가 특정 콘텐츠에 대해 어떤 양상으로 관찰할지를 결정하는 과정을 중요한 의사결정으로 다루며, 이는 기존의 벤치마크와의 차별성으로 작용합니다. 논문에서는 현존하는 선진 MLLM 모델을 LivingScreen에서 평가한 결과, 비용 대비 정확도에서 인간 성능에 미치지 못하는 사실을 확인했습니다.

- **Performance Highlights**: LivingScreen의 실행 결과, 강력한 모델조차 인간의 성능에 크게 뒤처진다는 것을 발견했습니다. 주된 실패 요인은 '과다/과소 관찰'로 나타났으며, 이는 현재의 MLLM 모델들이 정보 획득 시 필요한 만큼 주의를 기울이지 못한다는 것을 보여줍니다. 이 문제는 향후 GUI 에이전트 개발에 있어 해결해야 할 새로운 방향으로 제시되며, 정보 선택이 과제가 되는 점을 강조합니다.



### A New Angle on Bones: Robust Pose Estimation in X-Ray and Ultrasound (https://arxiv.org/abs/2606.04700)
Comments:
          Code and annotations for fracture angle assessment in radiographs: this https URL

- **What's New**: 본 논문은 자동 뼈 포즈 추정을 위한 새로운 방법을 제안하며, 이는 학습 기반의 점 후보 제안(point candidate proposal) 후 선 모델(line model)을 사용하여 축 매개변수를 추출하는 방식을 사용합니다. 일반적인 최소 제곱(line model) 같은 기존 모델들이 이상치(outlier)에 민감하기 때문에, RANSAC와 Hough 변환 같은 강건성(fitting techniques)을 통합하여 신뢰성을 향상시켰습니다.

- **Technical Details**: 제안된 방법은 깊은 학습(deep learning) 모델을 통해 각 뼈 구조에 대한 점 후보를 생성하고, 이 후보들로부터 강건 선(line) 모델을 사용하여 축을 추출하는 것으로 이루어져 있습니다. 각 뼈의 축을 추정하기 위해 우리는 선을 적합(fitting)하여 두 뼈 간의 각도를 계산합니다. 선은 기준점과 방향 벡터로 정의되며, 두 선의 교차 각도를 통해 최종 각도를 계산합니다.

- **Performance Highlights**: 우리의 방법은 소아 관절각 추정 과제에서 평균 오류가 각각 4.1°, 5.4°, 5.51°로 나타났으며, 이는 기대되는 임상 관찰자 변동성에 부합하며, 랜드마크 기반 방법 대비 성능이 크게 향상되었습니다. 이와 더불어, 우리의 코드는 GitHub에 공개되어 있어 연구자들이 쉽게 접근할 수 있습니다. 또한, 우리는 골절 각도 평가를 위해 새로운 정렬한 경계 박스(orientated bounding box) 주석도 공개하였습니다.



### MeshWeaver: Sparse-Voxel-Guided Surface Weaving for Autoregressive Mesh Generation (https://arxiv.org/abs/2606.04688)
Comments:
          CVPR 2026

- **What's New**: MeshWeaver는 메쉬 생성을 표면 엮기(surface weaving) 과정으로 접근하여 다음 정점을 직접 예측하는 방식을 도입하여 이전의 한계점을 극복합니다. 이전의 방법들이 전역 형태 임베딩에만 의존하였던 것과 달리, MeshWeaver는 로컬(surface) 기하학적 맥락을 통합하여 생성의 품질을 크게 향상시킵니다. 이 새로운 프레임워크는 컴팩트한 토큰화(tokenization) 기술과 결합하여 짧은 시퀀스를 통해 효율적으로 메쉬를 생성할 수 있게 합니다.

- **Technical Details**: MeshWeaver의 핵심은 다층 스파스-복셀 인코더(multi-level sparse-voxel encoder)로, 이 인코더는 메쉬 생성 과정에 기하학적 맥락을 주입합니다. 구체적으로는 정점을 표현하는 복셀 특징 제공, 복셀 특징에 대한 교차 주목(cross-attention)을 통한 토큰 예측 가이드, 입력 표면 주위의 생성 제약을 위한 구조적 스캐폴드(structural scaffold) 제공의 세 가지 방법으로 기여합니다. 이러한 계층적 설계는 단일 디코딩 단계 내에서 거칠게부터 섬세하게 샘플을 생성할 수 있도록 합니다.

- **Performance Highlights**: MeshWeaver는 18%의 최첨단 압축 비율을 달성하며, 최대 16K 면(faces)으로 메쉬 생성을 지원합니다. 또한, 이전 접근법에 비해 기하학적 충실도(geometric fidelity)가 크게 향상되어 실용적인 작업에 적합한 메쉬를 생성할 수 있습니다. 이로써 MeshWeaver는 효율성과 정확성을 동시에 갖춘 혁신적인 메쉬 생성 프레임워크로 자리잡았습니다.



### Real-Time Automatic License Plate Recognition Using YOLOv8, SORT Tracking, and Temporal Data Interpolation (https://arxiv.org/abs/2606.04684)
Comments:
          7 Pages, For Accessing code:this https URL mobeen-pmo/Automatic-License-Plate-Recognition

- **What's New**: 본 연구는 도로 교통 감시 환경에서 자동 번호판 인식(Automatic License Plate Recognition, ALPR)의 실시간 처리 한계를 극복하기 위해 5단계의 end-to-end 알고리즘 파이프라인을 제안합니다. 이 방법은 깊이 학습 기반 객체 탐지와 운동학적 멀티 객체 트래킹, 기하학적 시간 데이터 보간을 원활하게 연결합니다. YOLOv8 nano 모델을 활용하여 차량을 로컬라이즈하고, SORT 알고리즘을 통해 프레임 간의 공간-시간적 링크를 구축합니다.

- **Technical Details**: 제안된 방법론에서는 원시 비디오 프레임을 처리하기 위한 다섯 단계의 파이프라인이 존재합니다. 첫 번째 단계에서는 노드 없는 YOLOv8 nano 아키텍처(3.2 million parameters)를 사용하여 물체의 중심을 직접 예측합니다. SORT 알고리즘은 운동학을 모델링하여 차량 탐지를 연결하고, 칼만 필터와 헝가리안 알고리즘을 통해 데이터 연관을 최적화합니다.

- **Performance Highlights**: 제안된 ALPR 프레임워크는 101.9%의 위치 정보의 연속성을 향상시키기 위한 시간적 경계 상자 보간 알고리즘을 통해 성능을 극대화합니다. EasyOCR을 사용하여 영국 번호판 형식에 대한 강력하고 구문 인식이 가능한 후처리 유닛이 개발되었습니다. 실험적 분석에서는 결과와 탐지 확인, 경계 상자의 기하학, 트래킹 분할 간의 복잡한 연관성을 정량적으로 분석하였습니다.



### Instance-Level Post Hoc Uncertainty Quantification in Object Detection (https://arxiv.org/abs/2606.04656)
Comments:
          7 pages, 2 figures

- **What's New**: 이번 논문에서는 자율주행의 안전성을 위한 객체 탐지에서의 불확실성 정량화 문제를 다룹니다. 기존의 모델을 변경하지 않고도 사후적으로 불확실성을 계량하는 방법을 제안하는데, 이는 Laplace approximation을 기반으로 하고 있습니다. 새로운 방법으로 Monte-Carlo generalized linearized model (MC-GLM)을 소개, 각 인스턴스의 불확실성을 효과적으로 정량화합니다.

- **Technical Details**: MC-GLM은 각 바운딩 박스에 대한 불확실성을 산출하며, Monte Carlo 샘플링을 통해 원래의 Jacobian 계산을 대체하는 방식으로 구현됩니다. 이 방법은 다차원에서 수치적으로 효율적인 계산을 가능하게 하며, 사후적으로 불확실성을 평가할 수 있도록 설계되었습니다. 이를 통해 인스턴스 수준의 정확한 불확실성을 제공할 수 있으며, 계산 자원 소모가 제한적입니다.

- **Performance Highlights**: nuScenes 데이터셋을 사용한 실험 결과, 제안된 MC-GLM 방법의 효과성을 검증하였습니다. 이 방법은 바운딩 박스에 대한 불확실성을 고품질로 나타내며, 자율주행 시스템에서의 실시간 사용에도 적합한 것으로 나타났습니다. 또한, 예측 분포의 신뢰성을 높이며, 기존의 방법보다 더 나은 성능을 보여주고 있습니다.



### MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer (https://arxiv.org/abs/2606.04621)
Comments:
          CVPR2026 Highlight, Homepage: this https URL, Code: this https URL

- **What's New**: MeshFlow는 예술가와 같은 3D 메쉬를 생성하는 새로운 방법입니다. 기존 메쉬 생성기는 주로 Auto-Regressive (AR) 방법을 사용하지만, 이는 메쉬 크기에 따라 추론 비용이 제곱으로 증가하여 비효율적입니다. MeshFlow는 이러한 문제점을 해결하기 위해 Variational Autoencoder (VAE)를 도입하여 연속적인 정점 좌표와 이산적인 연결성을 지속적인 잠재 공간에서 표현합니다.

- **Technical Details**: MeshFlow는 대조 손실(contrastive loss)로 감독된 VAE를 사용하여 메쉬의 구성 요소를 생성합니다. 이 잠재 공간은 이전의 토큰 기반 메쉬 표현보다 훨씬 더 압축되어 있어 효율성을 높입니다. 또한, Rectified Flow transformer를 기반으로 한 3D 생성기를 구축하여 모든 메쉬 정점과 엣지를 병렬로 생성합니다.

- **Performance Highlights**: MeshFlow는 기존의 가장 빠른 AR 생성기보다 메쉬를 18배 빠르게 생성하며, 표준 메쉬 생성 메트릭에서도 뛰어난 정확도를 달성합니다. 이는 사용자가 더욱 복잡한 3D 메쉬를 효율적으로 작업할 수 있게 합니다.



### Beyond Symmetric Alignment: Spectral Diagnostics of Modality Imbalance in Vision-Language Models in the Medical Domain (https://arxiv.org/abs/2606.04613)
Comments:
          10 pages, 3 figures, 9 tables

- **What's New**: 본 논문은 의료 이미지-텍스트 데이터에 Vision-Language Models (VLMs)의 적용 시 발생하는 문제를 다루고 있습니다. 기존의 대칭적 representation alignment metrics는 두 가지 모달리티를 하나의 점수로 통합하여 어떤 모달리티가 교차 모달 손실을 유발하는지 숨기는 한계를 가지고 있습니다. 이를 해결하기 위해 Spectral Alignment Score (SAS)라는 비대칭 지표를 도입하여, 두 모달리티를 주축(modality)의 주 고유 공간에 투사하고 각 고유 모드의 상관관계를 계산하여 방향성 점수를 생성합니다.

- **Technical Details**: SAS는 서로 다른 두 모달리티(X, Y)의 단일 지표로서 사용될 수 있으며, 각 모달리티에 대해 두 개의 방향적 점수(S_{X \to Y}, S_{Y \to X})를 계산합니다. 이러한 점수의 차이(ΔSAS)는 모달리티 정보 불균형을 정량화하는 데 사용됩니다. 연구에서는 15개의 VLM을 평가하기 위한 벤치마킹 프레임워크에 SAS를 포함하고, 자연 및 의료 이미지-텍스트 데이터셋에서 기존의 6개의 alignment metrics와 함께 사용하였습니다.

- **Performance Highlights**: 실험 결과, 의료 이미지는 그에 상응하는 임상 보고서보다 훨씬 더 풍부한 구조적 정보를 보유하고 있음을 보여주었습니다. SAS는 의료 분야에서 데이터 검색 성능과의 제로 레이블 상관관계가 가장 강력하다고 입증되어, 임상 배치에 실질적인 진단 도구로 자리 잡을 수 있습니다. 이는 기존의 대칭적인 메트릭이 보지 못하는 방향적 비대칭성을 효과적으로 감지할 수 있다는 점에서 의의를 갖습니다.



### COMBINER: Composed Image Retrieval Guided by Attribute-based Neighbor Relations (https://arxiv.org/abs/2606.04604)
Comments:
          Accepted by IEEE TIP 2026

- **What's New**: 본 연구는 이미지 검색 분야에서 Composed Image Retrieval (CIR) 방식을 다루며, 시각적으로 유사하지만 속성과 관련이 없는 이미지를 구분하는 데 초점을 맞추고 있습니다. 기존 접근 기법들이 시각적으로 유사한 이미지를 처리하는 데 제한적이었다면, 본 연구는 속성 프로토타입을 기반으로 한 통합된 표현 방식을 통해 이 문제를 해결하고자 합니다. 이를 통해 다중 모달(모드) 피처 융합 및 유사성 모델링의 정확성을 높입니다.

- **Technical Details**: 제안된 네트워크는 COMposed image retrieval network guided By attrIbute-based NEighbor Relations (COMBINER)로, 세 가지 주요 모듈로 구성됩니다. 첫째, Adaptive Semantic Disentanglement 모듈을 사용하여 속성 피처를 분리합니다. 둘째, Unified Prototype-based Composition 모듈을 통해 Cross-modal Unified Prototype (CUP)을 구성하고 다중 모달 피처를 합성합니다. 마지막으로, Dual Relations Modeling 모듈은 속성 유사성을 기반으로 쌍 및 이웃 관계를 모델링합니다.

- **Performance Highlights**: COMBINER는 세 개의 벤치마크 데이터셋에서 실험을 통해 효과성을 입증하였으며, 기존 같은 분야의 방법들보다 더 정확한 수치적 성과를 보였습니다. 특히, 시각적으로 유사하지만 속성과 관련이 없는 샘플을 처리하는 데 있어 최초의 시도로, 속성 프로토타입 기반의 유사성 메트릭을 활용하여 샘플 간의 의미적 관계를 더욱 정확히 이해할 수 있었습니다. 이 연구는 CIR의 발전에 기여할 것으로 기대됩니다.



### 4D Reconstruction from Sparse Dynamic Cameras (https://arxiv.org/abs/2606.04593)
Comments:
          Accepted by 4DV Workshop at CVPR 2026

- **What's New**: 이 논문은 모노큘러 동적 RGB 비디오로부터의 4D 재구성을 위한 새로운 접근법을 제시합니다. 특히, 자율적으로 움직이는 소수의 카메라를 이용해 다중 보기 제약을 적용하여 깊이 모호성을 해결하는 희소 동적 카메라 설정에 초점을 맞추었습니다. 이 개념은 스포츠, 콘서트 및 TV 프로그램과 같은 실제 비디오 제작에 실용적으로 적용될 수 있으며, 이 설정이 4D 재구성을 정립하는 데 중요한 역할을 할 것으로 기대합니다.

- **Technical Details**: 논문에서는 inter-camera feature matching과 intra-camera point tracking을 통합한 간단하면서도 효과적인 3D track 초기화 방법을 제안합니다. 또한, 최적화 안정성을 높이기 위해 noise-robust depth-ordering regularization loss와 spatiotemporally diverse batch sampling 전략을 포함했습니다. 새로운 데이터셋 LetCamsGo를 도입하여, 서로 독립적으로 움직이는 세 대의 카메라와 한 대의 고정 카메라로 촬영한 4개의 다양한 환경에서 5개의 시퀀스를 포함한 동적 장면을 캡처했습니다.

- **Performance Highlights**: LetCamsGo에서의 종합적인 벤치마크 테스트 결과, 제안된 프레임워크가 기존 방법론에 비해 동적 영역에서 4D 재구성 품질을 크게 향상시켰음을 입증했습니다. 특히, naive MoSca 기반의 확장에 비해 시간적 일관성을 강화하여 더 안정된 재구성을 가능하게 했습니다. 이러한 결과는 희소 동적 카메라 설정의 잠재력을 보여주며, 향후 연구에 기여할 수 있기를 기대합니다.



### Impostor: An Agent-Curated Benchmark for Realistic AIGC Manipulation Localization (https://arxiv.org/abs/2606.04545)
Comments:
          10 pages, 3 figures, 5 tables

- **What's New**: 최근 생성적 이미지 편집(generative image editing)의 발전으로 지역화된 이미지 조작(localized image manipulation)의 현실감과 제어 가능성이 향상되었습니다. 이에 따라 이미지 조작 탐지 및 지역화(IMDL)에 대한 새로운 도전 과제가 제기되고 있습니다. 기존의 IMDL 벤치마크는 시각적 현실감, 조작 다양성 및 생성기 범위에서 한계를 가지고 있어, 최근 이미지 조작 트렌드를 반영하기 어렵습니다.

- **Technical Details**: 이 이러한 한계를 극복하기 위해, 우리는 Impostor라는 고품질 AI 편집 이미지 로컬라이제이션 데이터셋을 소개합니다. 이 데이터셋은 100K의 조작된 이미지를 포함하고 있으며, CraftAgent라는 닫힌 루프 에이전트 프레임워크를 통해 구축되었습니다. CraftAgent는 장면 인식(scene perception), 편집 계획(editing planning), 조작 실행(manipulation execution), 품질 검증(quality validation), 반복적 반성을 통합하여 다양하고 시각적으로 현실감 있는 조작된 이미지를 자동으로 생성합니다.

- **Performance Highlights**: Impostor는 최근 3개의 조작 유형과 7개의 최신 AIGC 모델에 의해 생성된 이미지를 포함하여, AIGC 기반 IMDL을 위한 보다 포괄적인 벤치마크를 제공합니다. 또한 우리는 PhaseAware-Net (PANet)이라는 의미론적 포렌식 프레임워크를 제안하여, 지역적 위상 모델링(local phase modeling)과 의미론적 포렌식 일관성 학습(semantic-forensic consistency learning)을 도입해, 의미가 있는 공간에도 불구하고 포렌식적으로 손상된 조작된 영역을 보다 잘 지역화합니다. 실험 결과, Impostor는 기존의 대형 비전-언어 모델(LVLMs) 및 전문 IMDL 방법에 상당한 도전 과제를 제기하며, PANet은 Impostor 및 여러 공개 벤치마크에서 뛰어난 성능을 기록합니다.



### Optical-Guided Neural Collapse for SAR Few-Shot Class Incremental Learning (https://arxiv.org/abs/2606.04528)
Comments:
          16 pages, 6 figures

- **What's New**: 본 논문은 합성 개구 레이더(SAR) 이미징에서의 Few-shot class-incremental learning (FSCIL) 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 광학 ATR 데이터 세트를 활용하여 SAR 특징 학습을 위한 기하학적 우선 정보를 제공하는 방식으로, 강한 방위 각도 민감도를 극복하고자 합니다. 특히, 이 연구는 데이터 부족과 SAR의 변동성 문제를 해결하기 위해 렌즈의 도움을 받습니다.

- **Technical Details**: 제안된 방법에서는 데이터가 풍부한 광학 ATR 데이터로부터 직교(feature subspaces) 특징 서브 공간을 파생시키고, 이를 SAR 특징 학습에 활용합니다. 두 개의 손실 함수, 즉 프로젝션 손실과 분류자 손실을 조정하여, 주요 각도 제약을 통해 SAR 특징을 직교 서브 공간에 투영합니다. 이러한 접근 방식은 클래스 평균에 집중하고, 클래스 간 각도를 유지하면서 신경 붕괴(neural collapse)를 유도합니다.

- **Performance Highlights**: 실험 결과, 본 연구는 24개 대상 클래스를 포함하는 벤치마크 데이터 세트에서 최고의 최종 정확도를 달성하며, 기존의 FSCIL 방법들보다 성능 저하가 적은 우수한 성능을 보였습니다. 또한, 신경 붕괴 지표는 클래스 간 분리성과 클래스 내 밀집도를 개선하여, 학습된 특징이 이상적인 단순엣지-ETF 기하학에 더 가깝게 근사함을 보여줍니다.



### SFMambaNet: Spectral-Frequency Enhanced Selective State Space Model for Correspondence Pruning (https://arxiv.org/abs/2606.04493)
- **What's New**: 본 논문에서는 최초로 주파수(domain) 영역 인식을 결합하여 대응(인라이어) 프루닝(Correspondence Pruning) 작업을 수행하는 SFMambaNet을 제안합니다. 기존의 Graph Neural Network(GNN) 기반 방법들이 기하학적 특징에 의존하는 반면, SFMambaNet은 지역적 스펙트럴-기하학적 주의(Local Spectral-Geometric Attention)와 스펙트럴 통합 글로벌 맘바(Spectral-Integrated Global Mamba) 블록을 통해 이러한 한계를 극복합니다. 이를 통해 인라이어와 아웃라이어를 보다 효과적으로 구분할 수 있습니다.

- **Technical Details**: SFMambaNet은 두 개의 주요 구성 요소로 나뉩니다. 첫 번째는 LSGA 블록으로, 지역 그래프 상호작용에 스펙트럴 위치 인코딩을 통합하고 다중 스케일 맘바 처리를 도입하여 미세한 기하학적 일관성을 캡처하고 지역 특징의 구별 가능성을 향상시킵니다. 두 번째는 SIGM 블록으로, 주파수 정보(frequency information)를 활용하여 상태 공간 내에서 고주파 잡음(high-frequency noise)의 축적을 억제합니다.

- **Performance Highlights**: SFMambaNet은 여러 도전적인 작업에서 현재의 최첨단 방법들보다 뛰어난 성능을 보입니다. 실험 결과, 이 방법은 인라이어와 아웃라이어 간의 분리 가능성을 향상시키며, 거의 선형 복잡도로 강력한 글로벌 컨텍스트 모델링 능력을 제공합니다. 코드 및 자료는 제공된 링크를 통해 이용할 수 있습니다.



### IMPose: Interactive Multi-person Pose Estimation with Dynamic Correction Propagation (https://arxiv.org/abs/2606.04480)
- **What's New**: 이번 논문에서는 IMPose라는 새로운 인터랙티브 포즈 주석 도구를 소개합니다. 이 도구는 다중 인물의 동적 포즈 주석을 위한 dual-level tracking 메커니즘을 포함하고 있으며, 이를 통해 단일 프레임에서의 수정을 전체 비디오로 전파할 수 있습니다. IMPose는 시간적 수정 전파와 다중 인물 간의 일관성을 동시에 해결하는 혁신적인 방법을 제안합니다.

- **Technical Details**: IMPose는 keypoint-level과 instance-level 추적 메커니즘을 결합하여 사용자가 제공한 포즈 수정을 전체 비디오 시퀀스에 전파합니다. keypoint-level 추적은 여러 프레임 간의 연속적인 모델링을 통해 수정을 시간적으로 전파할 수 있도록 지원하며, instance-level 추적은 occlusion과 motion blur와 같은 조건에서 다중 인물의 keypoint를 올바르게 할당하는 데 중점을 둡니다. 이러한 메커니즘은 복잡한 현실 상황에서도 안정적인 주석을 제공하는 데 필수적입니다.

- **Performance Highlights**: IMPose는 1,050프레임의 비디오에서 단 27회의 수정으로 동적 포즈 주석을 달성하였으며, PoseTrack21 데이터셋에서는 평균 84프레임의 비디오에 대해 인물 하나당 3회의 클릭만으로 주석 처리가 가능합니다. 이는 낮은 수동 조작으로도 높은 정확도와 효율성을 달성함을 보여줍니다. 나아가 IMPose는 10명의 주석자와 10시간의 작업으로 187,920명의 새로운 인물과 3,548,968개의 추가 keypoint를 포함하여 기존 PoseTrack21 데이터셋의 주석을 대폭 확장했습니다.



### Evaluating Reasoning Fidelity in Visual Text Generation (https://arxiv.org/abs/2606.04479)
Comments:
          Peer reviewed and accepted at CVPR 2026 at the GRAIL-V (Grounded Retrieval and Agentic Intelligence for Vision-Language) workshop (non-archival track)

- **What's New**: 최근의 텍스트-이미지(T2I) 모델들은 이미지 내에 잘 구조화된 텍스트를 렌더링할 수 있는 능력을 보여주어 문서 생성 및 슬라이드 제작과 같은 다양한 응용 프로그램에 기여하고 있습니다. 그러나 이러한 시스템이 복잡한 솔루션을 텍스트로 직접 표현할 때 추론 능력을 신뢰성 있게 유지하는지 여부는 여전히 불확실합니다. 본 연구에서는 시각적 텍스트 생성을 통한 추론 충실도를 평가하여 이 문제를 조사하고자 하였습니다.

- **Technical Details**: 우리는 여러 과제를 설계하여 현대의 대형 언어 모델(LLM)이 쉽게 해결할 수 있지만 T2I 모델에게는 어려운 다단계 텍스트 추론 문제를 평가했습니다. 주어진 프롬프트에 대해 T2I 모델이 이미지를 생성하고 텍스트를 추출하여 그 정확성을 검증하는 방법으로, 렌더링 오류와 추론 오류를 분리하여 평가를 수행했습니다. 이는 명확한 텍스트 렌더링이 시각적 텍스트에서 추론을 평가하는 데 필수적임을 인식하는 것에서 출발했습니다.

- **Performance Highlights**: 실험 결과, 현재의 T2I 모델들이 논리적으로 일관된 시각 텍스트를 생성하는 데 있어 신뢰할 수 없다는 것이 밝혀졌습니다. 특히, 렌더링된 텍스트가 시각적으로 명확할지라도 의미적 오류와 논리적 불일치, 그리고 잘못된 중간 단계가 빈번하게 발생했습니다. 이 결과는 텍스트 전용 모델이 동일한 작업에서 보여준 강력한 추론 성능과 대조되며, 시각적 텍스트 추론에 있어 더욱 신뢰할 수 있는 해결책이 필요함을 시사합니다.



### Adaptive Calibration for Fair and Performant Facial Recognition (https://arxiv.org/abs/2606.04469)
- **What's New**: Adaptive Calibration (AC)라는 새로운 캘리브레이션(Calibration) 전략을 도입하여 얼굴 인식 시스템의 성능을 향상시킵니다. AC는 정규화된 엠베딩(Embeddings) 간의 코사인 유사도(Cosine Similarity)를 잘 조정된 확률로 매핑합니다. 이는 원래의 거리가 다른 엠베딩 영역에서 서로 다른 일치 확률을 나타낼 수 있는 기본 불일치를 수정합니다.

- **Technical Details**: Adaptive Calibration은 모든 미리 훈련된 얼굴 인식 시스템에 적용할 수 있는 사후(Post-hoc) 접근 방식으로, 비교적 일관된 지역 특화 캘리브레이션(Local Calibration)을 제공합니다. AC는 교차 검증이 필요 없는 공정성과 정확성을 동시에 고려하여, 기존 방법들보다 더 나은 성능을 보여줍니다. 이는 AUC(Area Under the Curve), Brier Score와 같은 다양한 메트릭을 사용하여 평가됩니다.

- **Performance Highlights**: AC는 FairCal 및 FRAPPÉ와 같은 기존의 방법을 초월하여 얼굴 인식 시스템의 공정성과 정확성을 모두 개선했습니다. 이 방법은 인구 통계 데이터 없이도 공정한 얼굴 인식을 가능하게 하며, 다양한 데이터 세트에서 최고의 성능을 기록했습니다. 실험 결과, AC는 다수의 표준 벤치마크에서 기존의 방법들을 지속적으로 능가하는 성과를 보여줍니다.



### ChannelTok: Efficient Flexible-Length Vision Tokenization (https://arxiv.org/abs/2606.04461)
- **What's New**: 이 논문은 기존의 복잡한 spatial-token 접근 방법에서 벗어나, 간단하고 가벼우며 빠른 채널 기반의 유연 길이 토크나이저를 소개합니다. 제안된 방법은 기존의 파라미터가 많은 백본을 대신하여, 파라미터 효율적인 CNN-Transformer 하이브리드 백본을 사용합니다. 이로 인해 이미지 생성을 보다 효율적으로 수행할 수 있게 됩니다.

- **Technical Details**: 채널 기반 유연 토크나이징 접근 방식은 VQGAN에 기반한 오토인코더를 활용하며, 각 채널을 시각적 토큰으로 처리합니다. 인코더는 공간 해상도를 줄이고, 동시다발적인 자기 주의(multi-head self-attention)를 통해 전역 종속성을 캡처합니다. 이러한 설계는 명시적 정렬 제약 없이 자연스럽게 조밀한 계층 구조를 형성합니다.

- **Performance Highlights**: 제안된 모델은 ImageNet을 포함한 다양한 데이터셋에서 정량적 실험을 통해 2.92의 rFID로 탁월한 지각 품질을 달성하며, 디코딩 속도는 8.6배 더 빠르고 다음 최상위 모델보다 파라미터 크기가 2.1배 더 작습니다. 이러한 결과는 효율적인 시각 표현을 위해 채널 기반 토크나이징이 강력하고 실용적인 패러다임임을 확립합니다.



### Imagine Before You Draw: Visual Prompt Engineering for Image Generation (https://arxiv.org/abs/2606.04457)
- **What's New**: 이번 논문은 Visual Prompt Engineering (VPE)을 제안하여, 이미지 생성 과정에서 시각적 의미 표현을 중간 단계로 통합함으로써 모델링의 어려움을 줄이고 생성 품질을 개선합니다. 이전 연구들은 일반적으로 두 단계 외부 파이프라인을 사용했으나, VPE는 내부 모델 내에서 교차 모달 상호작용을 통해 이 정보를 통합하는 방법을 제시합니다. 이로 인해 이미지 편집 같은 세부 사항을 보존하는 데 필요한 정보 병목 현상이 완화됩니다.

- **Technical Details**: VPE는 시각적 프롬프트를 생성하는 단계에서 SigLIP 2 토큰을 사용하여 시맨틱 레이아웃을 포착한 후, 이 계획을 기반으로 전체 이미지 토큰을 생성합니다. 모델은 훈련 시 중요하지만 정적(ground-truth) 시각적 프롬프트에 과도하게 의존하는 문제가 있으며, 이는 추론에서 생성된 프롬프트의 불완전함으로 인해 발생하는 품질 저하로 이어질 수 있습니다. 이러한 문제를 해결하기 위해 점진적 훈련 프로세스를 도입하여 모델이 프롬프트에 대한 의존도를 점진적으로 증가시킵니다.

- **Performance Highlights**: 실험 결과, VPE는 클래스 조건부 생성, 텍스트-이미지 생성, 이미지 편집 등의 작업에서 일관된 개선을 보였습니다. 내부 아키텍처는 외부 아키텍처보다 이미지 편집에서 더 세밀한 세부 사항을 유지했으며, VPE는 이러한 내부 아키텍처의 수렴을 가속화할 뿐만 아니라 편집 응답성 또한 유지합니다. 결과적으로, PSNR(지각적 신호 대 잡음 비율)에서 VPE가 더 나은 편집 보존 성능을 나타내며, 경쟁력 있는 품질을 유지합니다.



### Radiomic Feature Selection Using Gradient Loss of Deep Neural Network for Lung Cancer Stage Detection (https://arxiv.org/abs/2606.04453)
- **What's New**: 이번 연구에서는 폐암 단계 탐지를 위해 Gradient-Loss Recursive Feature Elimination (GL-RFE) 프레임워크를 제안합니다. 이는 딥 러닝 네트워크의 그래디언트 민감도 분석을 통합하여 가장 영향력 있는 방사선학적 특성을 식별합니다.

- **Technical Details**: 연구에 사용된 특성들은 PyRadiomics 확장을 활용하여 3D Slicer 플랫폼에서 획득된 흉부 CT 스캔에서 추출된 총 106개의 방사선학적 특성입니다. GL-RFE 방법은 입력 특성에 대한 네트워크 손실의 그래디언트를 계산하고, 기여도가 미미한 특성을 재귀적으로 제거하는 방식으로 특성의 중요도를 평가합니다.

- **Performance Highlights**: 제안된 방법은 테스트 데이터셋에서 90.22%의 정확도, 90.10%의 정밀도, 90.24%의 재현율 및 90.16%의 F1-score로 강력한 분류 성능을 달성했습니다. 시각화 분석을 통해 특성 중복성이 줄어들고 클래스 분리가 향상되었음을 확인하였습니다.



### INTACT: Ego-Guided Typed Sparse Evidence Retrieval for Heterogeneous Collaborative Perception (https://arxiv.org/abs/2606.04437)
- **What's New**: 이 논문은 자율 차량의 협업 인식을 위한 새로운 프레임워크인 INTACT를 제안합니다. 기존의 번역 우선(translation-first) 패러다임에서 벗어나, INTACT는 에고(vehicle)의 쿼리를 기반으로 필요한 정보를 요청하는 새로운 접근 방식을 보여줍니다. 이를 통해 이질적인 센서와 피처를 가진 차량들 간의 협력을 효율적으로 수행할 수 있도록 합니다.

- **Technical Details**: INTACT는 에고 주도형(ego-guided) 타입 희소 증거 검색 프레임워크로, 차량이 타겟 객체 또는 불충분한 증거가 있는 지역에 대한 쿼리를 발행할 수 있도록 설계되었습니다. 각 차량은 요청된 위치에서만 로컬 증거를 반환하며, 에고는 이러한 응답 중 유용한 것을 선택하여 게이트 잔여 반환(gated residual write-back)을 통해 자신의 표현 방식에 주입합니다. 이는 글로벌 피처 맵의 해석 가능성 요구 사항을 지역적 쿼리를 기반으로 한 비교가능성으로 전환합니다.

- **Performance Highlights**: 실험 결과, INTACT는 OPV2V-H 데이터셋에서 0.52M의 추가 파라미터와 18.0 log₂의 통신량으로 80.1 AP70을 달성했습니다. DAIR-V2X에서는 도전적인 현실 환경에서도 43.8 AP50을 기록하며, 이는 기존의 밀집된 피처 전송 방식에 비해 약 16배의 압축률을 구현한 것입니다. 이러한 결과는 INTACT의 정확도, 효율성 및 배포 가능성을 입증합니다.



### 3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training (https://arxiv.org/abs/2606.04436)
- **What's New**: 이번 논문에서는 3D 사고를 유도하는 공동 훈련 프레임워크인 3DThinkVLA를 제안합니다. 이 프레임워크는 시각-언어-행동(Vision-Language-Action, VLA) 모델이 행동 예측을 수행할 때 암묵적으로 3D 공간 추리를 가능하게 합니다. 기존의 방법들은 대부분 2D 이미지에 의존하여 3D 공간 추리와의 심각한 간극을 보완하지 못했습니다. 우리의 접근법은 3D 기하학 인식과 3D 공간 추리를 서로 분리함으로써 이러한 한계를 극복하고 있습니다.

- **Technical Details**: 3DThinkVLA 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 잠재 3D 기하학 인식 모듈은 VLM 백본을 수정하지 않고 3D 기하학 정보를 모델에 주입합니다. 2) 온라인 3D 추론 증류 모듈은 공유된 추론 앵커 토큰을 사용하여 프롬프트에 의해 유도된 추론 격차를 완화합니다. 3) 공간적으로 증강된 행동 통합 모듈은 이해된 기하학과 추론 기능을 행동 쿼리 토큰에 결합하여 예측의 Robustness를 높입니다.

- **Performance Highlights**: 본 논문에서 제안된 방법은 LIBERO, LIBERO-PLUS, SimplerEnv 및 다양한 실제 조작 작업에서 최첨단 성능을 달성했습니다. 3D 입력 없이 2D 이미지만으로도 효과적으로 3D 공간 추리를 수행하여, VLA 모델의 사전 훈련된 의미적 정렬을 완전히 보존합니다. 이는 3D 센서나 외부 모델, 명시적 텍스트 생성을 요구하지 않으면서도 성능을 극대화하는 전략입니다.



### Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning (https://arxiv.org/abs/2606.04434)
Comments:
          Accepted at the 43rd International Conference on Machine Learning (ICML 2026)

- **What's New**: 이번 연구에서는 Hyper-ICL이라는 경량 훈련 기반 프레임워크를 제안하여, 기존의 다중모드 In-Context Learning (ICL)의 한계를 극복하고자 합니다. Hyper-ICL은 데모(모범 사례) 없이도 시연 효과를 재구성할 수 있도록 설계되었습니다. 이 방법은 attention 분포를 직접 교정하며, ICL의 쿼리 적응성을 통합한 모듈화를 통해 성능을 개선합니다.

- **Technical Details**: Hyper-ICL은 파라미터 효율적인 저차원 logit 수준 어댑터를 학습하여, 모델이 주목하는 대상을 개선합니다. 이를 통해 쿼리에 따라 intervention 강도를 조절하는 쿼리 적응 조정 메커니즘을 도입했습니다. 또한, 하이퍼볼릭 앵커 증류 손실이 중간 학생 표현을 시연 조건의 교사와 정렬하여, 시연 없이도 데모-쿼리 관계를 재구성할 수 있도록 합니다.

- **Performance Highlights**: Hyper-ICL은 VQAv2, OK-VQA, COCO Caption 등 6개의 다양한 멀티모달 벤치마크에서 실험을 통해 기존 ICL 방법보다 향상된 정확도와 안정성을 보여주었습니다. 높은 성능을 발휘하면서도 데모에 의존하지 않는 추론이 가능하여, 다양한 응용 분야에서 적용할 수 있음을 입증하였습니다.



### Stateful Visual Encoders for Vision-Language Models (https://arxiv.org/abs/2606.04433)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 Stateful Visual Encoder (SVE)를 도입하여 시각-언어 모델(VLM) 내의 이미지 간 상호작용을 직접 보강합니다. 기존의 VLM은 이미지가 독립적으로 인코딩되고, 이후에 언어 모델에서 비교가 이루어지는 반면, SVE는 이전 이미지의 시각적 특징을 기반으로 현재 시각적 표현을 조건화하여 효과성을 높입니다. 이로 인해, 특히 미세한 차이가 중요한 작업에서 향상된 성능을 보여 줍니다.

- **Technical Details**: SVE는 시각 인코더(f_V) 내부에 이미지를 인코딩할 때 이전 이미지의 특징을 통합하는 구조로 설계되었습니다. 이를 통해 Images 간의 조건부 상호작용을 가능케 하여, 더 정밀한 시각적 비교가 가능해집니다. 연구팀은 다양한 아키텍처 변형을 실험하여 최적의 성능을 발휘할 수 있는 디자인을 선정하였습니다.

- **Performance Highlights**: SVE를 적용한 VLM은 여러 제어된 시각 비교 작업에서 성능 향상을 보여주었고, 이를 통해 엔지니어링 전반에 걸쳐 다채로운 모델 크기와 이미지 해상도에서도 일관된 성능 향상을 달성했습니다. 임상 영상 진단, 세밀한 이미지 비교, 원격 감지와 같은 실제 작업에서도 일반 VLM 기반선 모델을 초월하거나 동등한 성능을 나타냈습니다.



### DSA: Dynamic Step Allocation for Fast Autoregressive Video Generation (https://arxiv.org/abs/2606.04432)
Comments:
          CVPR2026, Findings Track

- **What's New**: 이번 연구는 DSA(Dynamic Step Allocation)라는 신뢰 기반의 적응형 계산 프레임워크를 소개하여 기존의 자동 회귀 비디오 확산 모델의 정밀도를 개선하였다. DSA는 가벼운 신뢰 헤드를 도입하여 각 프레임의 노이즈 제거 신뢰도를 측정하고, 이를 바탕으로 각 프레임마다 적절한 노이즈 제거 단계를 동적으로 조정한다. 이러한 접근 방식은 빠른 비디오 생성 속도를 가능하게 하여 실제 응용에서도 유용하다.

- **Technical Details**: 이 연구의 핵심은 신뢰성을 예측하는 가벼운 네트워크를 통해 AR(autoregressive) 비디오 확산 과정에서 필요 없는 계산을 줄이는 것이다. DSA는 생성을 위한 프레임에서 노이즈 제거를 효과적으로 수행하며, 한정된 데이터 세트를 이용해 동적 샘플링을 가능하게 한다. 또한, 이 방법은 기존의 캐시 기반 접근 방식과 달리 추가적인 비디오 데이터나 수작업으로 설계한 휴리스틱이 필요 없다.

- **Performance Highlights**: DSA는 H100 GPU 상에서 평균 22.63 FPS를 기록하며 실시간 AR 비디오 생성을 달성하면서도, 기존의 최신 AR 및 양방향 비디오 확산 모델들과 비교할 때 경쟁력 있는 비주얼 품질을 유지한다. 실험 결과는 신뢰 기반의 적응형 샘플링이 상호작용 비디오 생성의 효과적인 경로임을 입증하며, 높은 품질의 생성과 효율적인 프레임 처리가 가능하다는 것을 보여준다.



### Implicit Fuzzification via Bounded Noise Injection for Robust Medical Image Segmentation (https://arxiv.org/abs/2606.04427)
Comments:
          Under reviewing

- **What's New**: 본 논문에서는 경계 모호성(boundary ambiguity)을 개선하기 위해 NoiseUNet이라는 새로운 프레임워크를 제안합니다. 이는 스킵 연결(skip connections)에 제한된 교란(bounded perturbations)을 주입하여 교차 스케일(feature fusion) 피처 융합을 정규화합니다. 이러한 접근법은 지역(feature) 특성의 변동성에 강건함을 부여하고 경계 인식을 촉진합니다.

- **Technical Details**: NoiseUNet은 엔코더-디코더 아키텍처(encoder-decoder architecture)의 연장선으로, U-Net과 비슷한 구조를 갖고 있습니다. 이 모델은 데이터 기반의 소프트 멤버십을 생성하여 명시적인 퍼지 모델링(explicit fuzzy modeling) 없이도 퍼지화(fuzzification) 효과를 유도합니다. 저자는 또한 ThyR라는 실세계 갑상선 초음파 데이터셋을 소개하는데, 이 데이터셋은 본질적으로 모호한 경계를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과 NoiseUNet은 분할(segmentation) 정확도와 경계 충실도(boundary fidelity)를 지속적으로 향상시키는 것으로 나타났습니다. 전반적으로 이 연구는 경계 모호성이 있는 데이터 세트에서도 강력한 성능을 제공하는 혁신적인 해법을 제시합니다.



### Motion-Guided Causal Disentanglement for Robust Multi-View Cine Cardiac MRI Diagnosis (https://arxiv.org/abs/2606.04414)
- **What's New**: 본 논문에서는 Motion-Guided View-Disease Disentanglement 프레임워크인 MoViD를 제안하여 다양한 관점에서 얻어진 심장 자기공명영상(CMR)의 질병 분석을 개선하고자 합니다. 기존의 transformer 모델들이 질병 관련 특징과 관점 별 해부학적 변화를 혼합하여 학습하는 문제를 해결하기 위해, MoViD는 질병 구분과 관점 구분을 명시적으로 분리하여 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: MoViD는 ViT-MAE 백본을 기반으로 하며, 공간적 및 시간적 특징을 인코딩하는 3D CNN 스템과 뷰 별 특징을 생산하는 ViT 인코더로 구성됩니다. 이 모델은 이중 가지의 대조적 목표를 사용하여 질병 임베딩과 뷰 임베딩 간의 명확한 분리를 달성하고, 이동 진단 기능을 통해 심장 지역을 지역화하는 방법을 제시합니다. 또한, 클래스 불균형 문제를 완화하기 위해 집중 조정 메커니즘을 도입했습니다.

- **Performance Highlights**: 모델은 개인화된 정맥 혈전증 데이터셋과 두 개의 공개 벤치마크(M&Ms, M&Ms2)에서 평가되었습니다. 그 결과, 질병 분류 및 심장 세분화 작업 모두에서 기존의 transformer 기반 모델들을 지속적으로 초과하는 성능을 보였으며, 대규모 선행 학습된 기반 모델들과 비교하여 경쟁력 있는 성능을 입증했습니다.



### Ultra-Fast Neural Video Compression (https://arxiv.org/abs/2606.04410)
Comments:
          CVPR 2026

- **What's New**: 이번 논문은 비디오 압축을 위한 새로운 청크 기반 코딩 프레임워크를 소개합니다. 이는 여러 프레임을 동시에 인코딩하여 압축 효율성을 크게 제고하는 방법론입니다. 제안된 DCVC-UF(초고속) 인코더는 기존의 코덱보다 뛰어난 성능을 발휘하며, 실시간 전송 시 어려움이 많았던 인코딩 복잡성을 줄입니다. 전반적인 압축 성능이 개선되는 동시에 트레이닝 시간도 단축되어 보다 효율적인 비디오 압축이 가능해졌습니다.

- **Technical Details**: 청크 기반 코딩 프레임워크는 영상의 프레임을 비디오 청크로 나누어 동시에 압축하며, 각 청크 안에 존재하는 프레임 간의 공간-시간 상관관계를 모델링합니다. 이 과정에서 크로스 프레임 상호작용 모듈과 프레임 특화 디코더를 사용하여 압축 및 복원을 최적화합니다. 또한, 엔트로피 코딩 메커니즘을 단순화하여 비트 스트림 상호작용을 하나의 단계로 통합함으로써 디코딩 오버헤드를 최소화합니다.

- **Performance Highlights**: DCVC-UF 인코더는 1080p 비디오에 대해 초당 655.9 프레임의 인코딩 속도를 달성하였으며, 평균적으로 42.2%의 비트율 절감 효과를 가져옵니다. 이 외에도 다양한 구성에 따라 소비자 GPU에서 핵심 성능 기록을 갱신하며, 향후 AI 가속기의 발전으로부터 혜택을 자동으로 받을 수 있는 구조를 가지고 있습니다. 전 세계의 최신 코덱들과 비교해 높은 압축, 왜곡 및 복잡성의 성능을 달성하였습니다.



### An Empirical Study of Data Scale, Model Complexity, and Input Modalities in Visual Generalization (https://arxiv.org/abs/2606.04409)
Comments:
          12 pages, 9 figures, 4 tables

- **What's New**: 이 연구는 최신 딥 뉴럴 네트워크의 일반화 성능에 영향을 미치는 요인들, 즉 데이터 규모, 모델 복잡성 및 입력 양식에 대한 실증 분석을 제공합니다. 이를 위해 초기 실험에서는 1차원 비선형 함수를 구성하고, 데이터 샘플의 수와 다항식의 차수를 변화시켜 모델 성능에 미치는 영향을 관찰합니다. 주 실험에서는 CIFAR-10 및 CIFAR-100 데이터셋을 이용하여 다양한 교육 데이터 규모와 모델 아키텍처에서의 성능을 비교합니다.

- **Technical Details**: 이 연구는 모델의 일반화 성능에 대해 데이터 규모와 모델 복잡성, 입력 양식이 주는 영향을 살펴보며, 훈련 손실(training loss), 테스트 손실(test loss), 테스트 정확도(test accuracy)를 주요 지표로 사용합니다. 초기 실험에서는 저차원 합성 데이터에서 통제된 환경을 통해 비선형 함수 피팅을 시도하고, 이후 CIFAR-10 및 CIFAR-100 데이터셋에서 MLP, AlexNet, ResNet 모델의 성능을 비교하는 방법론을 채택하였습니다.

- **Performance Highlights**: 실험 결과, 교육 데이터 규모의 증가가 일반화 성능을 지속적으로 향상시키는 것으로 나타났습니다. 반면, 모델 복잡성의 변화는 안정적인 성과 향상을 제공하지 않았습니다. 또한, 색상 정보가 제거될 경우 모델 성능이 저하되며, 기울기(gradients), 에지(edges), 웨이브렛(wavelets)과 같은 명시적 사전 특징이 다양한 모델 아키텍처에서 불균형적인 영향을 미친다는 것을 발견했습니다.



### Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models (https://arxiv.org/abs/2606.04385)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 연구에서는 GPUA(Geometry-Preserving Unsupervised Alignment)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 VLM(vision-language models)과 VFM(vision-only foundation models) 간의 상호 보완적인 강점을 통합하여 모델 간 호환성을 개선합니다. VFM의 특성을 비주얼 언어로 간주하고 이를 VLM의 의미 공간으로 변환하는 독립적인 매핑을 학습합니다.

- **Technical Details**: GPUA는 라벨이나 모델 파라미터 업데이트 없이 VFM 공간을 VLM 의미 공간으로 전환하면서도 기하학(geometry)을 보존합니다. 이는 교차 언어 정렬(cross-lingual alignment)에서 영감을 받아 개발되었습니다. 이 과정은 오직 특징 수준(feature-level) 접근만을 요구하며, 사전 학습된 모델에 쉽게 적용할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크를 통한 실험 결과 GPUA는 모델 간 호환성이 개선되었음을 보여줍니다. 또한 다운스트림 작업인 제로 샷 인식(zero-shot recognition) 및 분할(segmentation)에서도 강력한 성능 향상을 나타냅니다. 이 과정에서 발생하는 성능 비용은 미미한 수준으로 보고되었습니다.



### Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers (https://arxiv.org/abs/2606.04373)
Comments:
          Accepted to appear at ICML 2026, Seoul, Korea

- **What's New**: 이번 논문에서는 Vision Transformers (ViTs)에 대한 새로운 Data-Free Quantization (DFQ) 접근법인 MaskAQ를 제안합니다. 기존의 DFQ 기법들이 합성 샘플과 양자화 모델의 입력 분포 간의 차이로 인해 성능 저하를 겪는 문제를 해결하기 위해, MaskAQ는 매우 중요한 이미지 패치인 Informative Region (유용한 영역)을 통해 성능을 극대화합니다. 이 기법은 저품질 샘플의 문제를 해결하기 위해 패치 유사성에 대한 차이 엔트로피를 최대화하여 노이즈 배경에서 유용한 영역을 분리하는 방법을 사용합니다.

- **Technical Details**: MaskAQ는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 차분 엔트로피 최대화를 통해 합성 샘플의 유용한 영역을 노이즈 배경에서 분리합니다. 둘째, 정보 지역을 고정시키기 위한 적응형 마스킹 메커니즘이 있으며, 이를 통해 양자화 모델(Q)와의 Masked Attention Alignment를 수행합니다. 마지막으로, 주기적인 샘플 갱신 전략을 통해 합성 샘플과 Q의 출력을 지속적으로 조화롭게 유지하여, 훈련 과정의 진화에 적응할 수 있는 능력을 부여합니다.

- **Performance Highlights**: MaskAQ는 다양한 백본 및 다운스트림 작업에서 기존의 최첨단 기법들을 초월하는 성능을 보여줍니다. 특히 3비트 양자화 상황에서 MaskAQ는 ImageNet 데이터셋의 DeiT-T 모델에서 기존 기법 대비 최대 3.1%의 Top-1 정확도 향상을 이끌어냈습니다. 이러한 결과는 MaskAQ가 VIts를 위한 DFQ에서 경쟁력 있는 솔루션임을 잘 보여줍니다.



### VT-3DAD: Cross-Category 3D Anomaly Detection via Visual-Text Normal Space Alignmen (https://arxiv.org/abs/2606.04369)
- **What's New**: 최근 논문에서는 few-shot cross-category 3D anomaly detection을 위한 VT-3DAD라는 새로운 training-free framework을 제안하고 있습니다. 이 방법은 시각적(reference) 기준과 텍스트적(normal) 기준을 정렬하여 정상성을 모델링하는 접근 방식을 채택합니다. 이로 인해 기존 방법들은 훈련 없이도 더욱 robust하게 이상 탐지를 수행할 수 있습니다.

- **Technical Details**: VT-3DAD는 frozen CLIP visual encoder를 활용하여 테스트 포인트 구름(Point Cloud)과 few-shot 정상 참조(normal reference)에서 시각적 특징을 추출합니다. 또한, frozen CLIP text encoder를 통해 깊이 및 3D 정보를 반영한 텍스트적 prompt를 사용하여 의미론적(normal semantic) 기준을 구축합니다. 결합된 시각적 및 텍스트적 신호를 기반으로 이상 점수(anomaly score)를 계산하여, 두 공간에서의 이상치를 동시에 평가합니다.

- **Performance Highlights**: ShapeNetPart 데이터셋을 통해 VT-3DAD의 성능이 뛰어난 결과를 보였으며, 특히 1-shot의 AUC-ROC이 92.49%에서 94.80%로 개선되었습니다. 또한, 평균 표준편차는 5.64에서 3.41로 줄어들어, 이상 탐지의 정확성과 신뢰성을 높였습니다. 이로 인해 VT-3DAD는 현재 state-of-the-art 성능을 달성하였습니다.



### Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes (https://arxiv.org/abs/2606.04365)
- **What's New**: 이 논문에서는 신장( kidney ) CT( 컴퓨터 단층촬영 )에서 병변( lesion ) 특성을 단일 모델로 예측하는 새로운 접근법을 제안합니다. 기존의 3D 방법들은 환자나 장기 단위에서 예측하였으나, 우리는 각 신장 당 병변을 개별적으로 예측할 수 있는 방법으로 문제를 재구성하였습니다. 이를 통해 2,619개의 CT 볼륨을 사용하여 다층적 레이블링을 수행하였으며, KiTS23 데이터셋을 통해 외부 유효성을 검증합니다.

- **Technical Details**: 제안된 모델, LesionDETR은 DETR 스타일 아키텍처를 채택하여 크기 및 거리의 헝가리안 매칭을 사용하며, 각 슬롯 출력값을 종합하여 측면 수준의 목표로 삼는 계층적 손실을 사용합니다. 네 가지 입력 표현법과 여섯 가지 인코더 초기화 방식을 통해, 분할 마스크를 입력 채널로 사용하는 것과 같은 도메인의 복부(SuPreM) 사전 훈련이 주요 설계 요소로 나타났습니다. 일반 대규모 데이터로의 사전 훈련은 무작위 초기화보다 성능이 떨어지는 것으로 확인되었습니다.

- **Performance Highlights**: LesionDETR는 UF-Health에서 양측 측면 수준의 이상 AUC(곡선 아래 면적) 0.799 ± 0.009, KiTS23에서는 0.817 ± 0.072를 달성하였습니다. 또한, 카운트 조건부 변형 모델은 낭종 병변에 대해 per-lesion mAP(평균 정밀도) 0.190 ± 0.083을 기록했으며, 드문 고형 병변에서의 AP는 여전히 낮은 수준을 유지하고 있어 데이터 수집의 필요성을 강조합니다. 이 프레임워크는 하류의 구조화된 보고서 생성을 위한 개별 병변 예측을 검증할 수 있는 기능도 제공합니다.



### Spatially Grounded Concept Bottleneck Models via Part-Factorized Attention (https://arxiv.org/abs/2606.04364)
- **What's New**: 이 논문은 Concept Bottleneck Models (CBMs)을 통해 클래스 예측 전에 인간이 이름 붙인 속성을 예측하여 의사결정을 감사할 수 있도록 하는 새로운 방식을 제안합니다. 이 연구에서는 자유로운 주의(attention) 사용을 제거하여 더욱 구조화된 part-factorized CBM을 개발했습니다. 이 모델은 DINOv3 비전 트랜스포머 상단에서 세 가지 주요 컴포넌트로 구성됩니다.

- **Technical Details**: 제안된 방법은 Frozen DINOv3 비전 트랜스포머에 기반하며, 학습된 포어그라운드 게이트가 배경 패치를 억제하고, 312개의 CUB 속성을 각 부분 토큰에 연결하는 개념-부분 맵을 통해 크로스 어텐션을 수행합니다. 또한, 주의 로짓(logits)에 추가적으로 주입되는 학습 가능한 2차원 정규 분포(prior)는 부분 쿼리 간의 순열 대칭(permutation symmetry)을 깨는 데 도움을 줍니다.

- **Performance Highlights**: CUB-200-2011 데이터셋에서, 제안된 모델은 완전 감독된 기준 모델과 비슷한 성능을 보이며, 포인팅 정확도를 16포인트 높였습니다. 전통적인 바운딩 박스 감독을 PCA 포어그라운드 타겟으로 대체하고 정규 분포와 결합하여 약 70%의 포인팅 정확도로 88.6%의 top-1 정확도를 달성할 수 있었습니다.



### Video2LoRA: Parametric Video Internalization for Vision-Language Models (https://arxiv.org/abs/2606.04351)
- **What's New**: 비디오 이해를 위한 새로운 접근 방식을 제안하는 Video2LoRA 모델을 소개합니다. 이 방법은 비디오를 인코딩하기 위해 기존 VLM(비전-언어 모델)의 매개변수를 활용하여 비디오의 내부 표현을 저차원 적응기, 즉 LoRA(adapter)로 변환합니다. 이를 통해 쿼리 시 비주얼 토큰(visual token)을 요구하지 않고도 비디오에 대한 질문에 대답할 수 있습니다.

- **Technical Details**: Video2LoRA는 비디오를 단일 포워드 패스를 통해 LoRA 어댑터로 전환합니다. 이를 위해 VLM 인코더가 비디오를 layer-wise로 인코딩하고, 학습 가능한 Perceiver 하이퍼네트워크가 이를 매핑하여 LoRA 가중치를 생성합니다. 이 과정에서 VLM 인코더와 응답 모델은 모두 정지되며, 하이퍼네트워크만 최적화됩니다.

- **Performance Highlights**: Video2LoRA는 500M 및 2.2B 모델에서 비디오 요약 및 캡셔닝 작업을 위해 훈련되었습니다. 다양한 벤치마크에서 직접 비디오 인-컨텍스트 추론과 통계적으로 동등한 성능을 보이며, 쿼리 시간 비주얼 토큰 부담을 최대 1,500배 줄이고 응답 시간도 6-80배 감소시켰습니다. 또한, 비디오의 비주얼 정보를 퓨전하지 않고도 안정적인 결과를 유지하며, 비디오 세그먼트에 대한 독립적인 어댑터의 조합 가능성도 확인되었습니다.



### MorphoQuant: Modality-Aware Quantization for Omni-modal Large Language Models (https://arxiv.org/abs/2606.04349)
- **What's New**: 본 논문에서는 기존의 4-bit Omni-modal Large Language Models (OLLMs)에서 발생하는 포스트 훈련 양자화(Post-Training Quantization, PTQ)의 과제를 해결하기 위해 MorphoQuant라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 교차 모드 형태학(cross-modal morphology)을 보존하고, 아웃라이어 손실을 완화하기 위해 Distribution-Aware Bias Compensation (DABC) 메커니즘을 도입합니다. 또한, Morphology-Directed Quantization Function Optimization (MDQFO) 전략을 통해 양자화 그리드와 편향 마스크를 공동 최적화하여 다양한 모드 분포 간의 미세 조정을 보장합니다.

- **Technical Details**: MorphoQuant는 다양한 감각 입력이 포함된 OLLMs의 극한의 메모리 요구와 컴퓨팅 비용 문제를 해결하기 위해 설계되었습니다. DABC 메커니즘은 다채로운 아웃라이어 패턴의 영향을 중립화하여 채널별 편향을 통해 트렁케이팅 잔여물을 흡수합니다. MDQFO에 의해 최적화된 양자화 기능은 분산 스코어를 통해 아웃라이어가 길게 늘어진 경향을 수량화하여, 다양한 모드의 분포 정렬을 향상시키는 데 기여합니다.

- **Performance Highlights**: 본 연구에서 제안한 W4A4 모델은 ScienceQA에서 76.63%의 성능을 기록하며, 기존의 SOTA W4A4 방법을 뛰어넘고, 심지어 W4A16 기준선도 초과하는 놀라운 결과를 보여줍니다. 이러한 성과는 MorphoQuant 프레임워크의 정확도-효율성 간의 탁월한 균형을 입증하며, 다양한 모드의 분포 간의 전환을 더욱 원활하게 할 수 있는 잠재력을 보여줍니다.



### HYolo: An Intelligent IoT-Based Object Detection System Using Hypergraph Learning (https://arxiv.org/abs/2606.04345)
Comments:
          8 pages, multiple figures;

- **What's New**: 이번 논문에서는 하이퍼그래프 학습을 YOLO 아키텍처에 통합한 HYolo라는 인공지능 IoT 기반 객체 탐지 프레임워크를 제안합니다. 전통적인 YOLO 모델은 객체 간의 복잡한 고차원 관계를 모델링하지 못하는 경향이 있어, HYolo는 이러한 한계를 극복하기 위해 하이퍼그래프 학습을 통해 맥락 의존성을 더욱 풍부하게 캡처합니다. 실험 결과, COCO 데이터셋에서 기존 YOLO 모델 대비 약 12% 향상된 mAP@50 점수를 기록하며 전반적인 탐지 정확도와 강인성을 개선했습니다.

- **Technical Details**: HYolo 접근법은 하이퍼그래프를 사용하여 더 높은 차원의 상호작용을 촉진하고 맥락화를 개선합니다. 이 모델은 HyperC2Net 아키텍처를 통해 여러 피쳐 노드 간의 상호작용을 이끌어내며, 하이퍼그래프 컨볼루션(HyperConv) 계층을 도입하여 맥락 정보를 더 잘 추출할 수 있습니다. 기존 YOLO 기반 모델에서 단순한 피쳐 융합 방식으로 인해 기능 상호작용이 제한되었던 문제를 해결하고, 다중 차원 특성의 관계를 학습할 수 있도록 합니다.

- **Performance Highlights**: HYolo는 COCO 데이터셋에서 테스트되어 pAP@50, 손실, 정밀도-재현율 등의 성능 지표를 통해 평가되었습니다. 이 연구에서 도입한 거리 기반 하이퍼그래프 구축 방식 및 하이퍼그래프 컨볼루션은 모델이 복잡한 패턴을 학습할 수 있도록 도와주며, 특히 가벼운 탐지 시나리오에서 성능을 향상시킵니다. 결과적으로 HYolo는 평범한 YOLO 모델에 비해 탐지 효율성을 크게 개선한 것으로 나타났습니다.



### Robust Multi-view Clustering against Imperfect Information (https://arxiv.org/abs/2606.04343)
Comments:
          19 pages, 11 figures

- **What's New**: 이번 논문에서는 복잡한 실세계의 다중 뷰 데이터로 인해 발생하는 불완전한 정보 문제(imperfect information problem)에 대해 다룹니다. 특히, 뷰 별 관측치의 부재(incomplete views, IV)와 뷰 간의 일치하지 않는 대응(noisy correspondences, NC)의 문제를 해결하기 위해 새로운 다중 뷰 클러스터링(multiview clustering, MvC) 방법인 Posterior-guided Latent Counterpart Inference (PLCI)를 제안합니다. PLCI는 IV와 NC 문제를 통합적으로 다룰 수 있는 강력한 프레임워크로, 기존의 방법들이 갖고 있는 한계를 극복할 수 있도록 고안되었습니다.

- **Technical Details**: PLCI는 각 앵커 인스턴스의 원하는 크로스 뷰 대응을 은닉 변수(latent variable)로 정식화하고, 인스턴스 수준의 신뢰성과 프로토타입 수준의 의미 전달(semantic transport)을 통합하여 은닉 대응의 후향 분포(posterior distribution)를 추론합니다. 구체적으로, PLCI는 심층 신경망의 메모리 효과를 활용하여 완전 인스턴스에서 신뢰성 있는 대응을 추정하고, 이후 모든 관측 샘플에서 구성된 뷰 별 프로토타입을 이용한 신뢰성 인식 최적 수송을 통해 의미 구조를 전파합니다. 마지막으로, PLCI는 전파된 의미 구조와 관측된 정보를 결합하여 각 인스턴스에 대한 신뢰할 수 있는 유사 대응(pseudo counterpart)을 추론합니다.

- **Performance Highlights**: 여섯 개의 다중 뷰 데이터 세트와 10개의 최첨단 MvC 방법에 대한 실험을 통해 PLCI의 효과성을 입증했습니다. 실험 결과, PLCI는 불완전한 정보 문제를 다루는 데 있어 기존의 IV 및 NC 중심 방법들보다 우수한 성능을 보였습니다. 또한, PLCI는 기존 MvC 방법의 강인성을 향상시킬 수 있는 범용 프레임워크로서, 다양한 분야에 쉽게 통합될 수 있음을 보여주었습니다.



### Answer Self-Consistency with Margin-Triggered Question Re-Arbitration for the CVPR 2026 VidLLMs Challeng (https://arxiv.org/abs/2606.04323)
- **What's New**: 이번 보고서에서는 CVPR 2026 VidLLMs Challenge의 Track 2에서 비주얼 관계 추론(visual relational reasoning) 기반 비디오 질문 답변 솔루션을 제시합니다. 우리의 방법인 Answer Self-Consistency with Margin-Triggered Question Re-Arbitration (ASC-MQRA)는 훈련이 필요 없는 테스트 시간 추론 프레임워크로, 여러 번의 비디오 질문-답변 실행을 통해 답변 선택을 집계하는 방식을 사용합니다. 또한, 불확실성을 평가하기 위해 조건부 재심사 모듈인 MQRA를 연구하여 후보 선택을 좁히고 비디오를 다시 시청할 수 있는 기회를 제공합니다.

- **Technical Details**: ASC-MQRA 프레임워크는 다중 모달(multi-modal) 추론 모델을 기반으로 하며, 각 질문에 대해 여러 번의 비디오 질문-답변을 수행하여 자가 일관성을 통해 답변을 집계합니다. 핵심적으로 ASC는 평균 다수결(vote)로 최종 답변을 선정하며, 불확실한 예시를 위해 MQRA를 도입하여 재시청을 통해 후보 답변을 명확히 하는 과정을 거칩니다. 이 시스템은 비디오와 질문, 여러 답변 선택지를 입력으로 받아 최선의 답변을 선택하도록 설계되어 있습니다.

- **Performance Highlights**: 실험 결과, ASC는 평균 72.73의 정확도와 78.34의 카테고리별 매크로 평균 정확도를 검증 데이터에서 달성하였고, 테스트 데이터에서는 81.16의 평균 정확도와 80.91의 카테고리별 매크로 평균 정확도를 기록했습니다. MQRA는 검증 과정에서 ASC에 비해 성능을 개선하였지만, 테스트 데이터에서는 약간의 성능 저하를 보였고, 이는 재심사가 트리거된 하위 집합의 크기와 카테고리 분포에 민감하다는 것을 시사합니다. 최종 제출에서는 ASC만을 사용하여 보다 안정적인 결과를 도출하였습니다.



### XSSR: Cross-Domain Self-Supervised Representative Selection for Efficient Annotation in Medical Image Segmentation (https://arxiv.org/abs/2606.04301)
Comments:
          Accepted to the Third International Conference on AI in Healthcare (AIiH 2026). This is the preprint version of the paper

- **What's New**: 이 연구는 XSSR(Cross-Domain Self-Supervised Representative Selection)라는 프레임워크를 제안하여, 서로 다른 의료 영상 데이터를 손쉽게 주석 처리하는 문제를 해결합니다. XSSR은 세 단계로 구성되어 있으며, 첫 번째 단계에서는 Masked Autoencoder(MAE)를 사용하여 레이블이 없는 소스 데이터를 기반으로 공유 임베딩 공간을 구축합니다. 두 번째 단계에서는 그리디 선택 알고리즘을 통해 타겟 샘플의 점수를 매기고, 마지막 단계에서는 선택된 하위 집합을 기반으로 U-Net 모델을 학습합니다.

- **Technical Details**: XSSR의 주요 기술 구성 요소는 자가 지도 방식의 표현 학습과 샘플 선택 방법입니다. Masked Autoencoder(MAE)는 레이블 없는 소스 데이터에서 특징 공간을 학습하며, 이 임베딩 공간을 사용하여 타겟 샘플을 선택합니다. 이를 위해 밀도(density), 참신성(novelty), 다양성(diversity)을 고려한 점수 함수가 사용되며, 각 요소는 자동으로 조정됩니다.

- **Performance Highlights**: XSSR는 Chest X-ray, RIGA+ 망막 사진, 다기관 전립선 MRI 데이터셋에서 공공 벤치마크에 대한 성능 평가에서 총 데이터의 99.3% 재현율을 달성하였고, Prostate MRI 데이터에서 무작위 선택 yöntem보다 최대 2.5 Dice 점수 포인트 향상되었습니다. 또한, 모든 데이터셋에서 CoreSet 기준을 0.4에서 1.2 Dice 점수 포인트 초과하였습니다.



### Efficient and Training-Free Single-Image Diffusion Models (https://arxiv.org/abs/2606.04299)
Comments:
          CVPR 2026; Project Page: this https URL

- **What's New**: 이번 연구는 단일 이미지를 바탕으로 한 패치의 분포에 따라 내부 구조를 모델링하는 새로운 접근 방식을 제안합니다. 전통적인 이미지 생성 방식은 시간과 자원이 많이 소모되었으나, 우리의 방법은 트레이닝 없이도 이미지를 생성할 수 있도록 최적의 클로즈드 폼(Closed-form) 디노이저(denoiser)를 사용합니다. 이로 인해 컴퓨터 비전의 다양한 문제에 대한 효율성을 가져오며, 텍스트 기반의 스타일화 및 리타겟팅과 같은 여러 응용 분야에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: 우리의 방법은 훈련이 필요없는 패치 기반 이미지 디퓨전 모델을 통합하여 이미지 내부 구조를 모델링합니다. 이미지에서 조합 가능한 패치 세트를 다루며, 최적의 디노이저를 사용하여 패치의 스코어 함수를 클로즈드 폼으로 계산할 수 있습니다. 이런 과정을 통해 새로운 이미지를 생성하는데 필요한 복잡한 훈련 절차를 제외하여 연산 비용을 크게 줄이고 이미지를 생성할 때의 유연성을 높였습니다.

- **Performance Highlights**: 우리의 접근 방식은 단일 이미지 모델링의 생성 품질에서 최신 기술에 필적하거나 더 나은 성능을 보여줍니다. 우리는 메가픽셀 이미지 생성을 1초 이내로, 기가픽셀 이미지는 몇 분 안에 생성할 수 있는 여러 가속 기법을 구현했습니다. 이러한 성과는 GAN이나 기존의 디퓨전 모델을 사용하는 대신, 고유의 패치 기반 접근법을 기반으로 하고 있음을 보여주며, 모델 훈련 없이도 프리미엄 품질의 이미지를 생성할 수 있도록 합니다.



### A Cookbook of 3D Vision: Data, Learning Paradigms, and Application (https://arxiv.org/abs/2606.04291)
Comments:
          Accepted to the CVPR 2026 OpenSUN3D Workshop. Official version available at CVF Open Access. this https URL

- **What's New**: 이 논문은 3D 비전을 위한 데이터 중심의 분류 체계를 제공하여 기하학적 표현, 데이터셋, 학습 프레임워크, 적용 분야를 하나의 개념적 지도로 연결합니다. 3D 데이터 구조와 학습 파이프라인의 복잡성을 분석하고, 이를 통해 2D 감독된 학습 및 4D 세계 모델링을 포함한 최근의 진전을 조명하고 있습니다. 이러한 통합적 시각은 효율성, 진실성(fidelity), 확장성의 균형을 맞추는 새로운 트렌드를 명확히 합니다.

- **Technical Details**: 본 논문은 3D 데이터 표현(categorical representation)과 이를 처리하는 방식, 즉 포인트 클라우드(point clouds), 메쉬(meshes), 복셀 그리드(voxel grids) 등 다양한 데이터 구조의 효율성과 신뢰성 간의 트레이드오프를 설명합니다. 데이터셋 및 벤치마크는 모델 개발의 제약을 정하고, 현대 신경망 접근 방식을 포함한 새로운 모델링 패러다임인 2D 감독 3D 학습 및 신경 암묵적 필드(implicit neural fields)를 탐구합니다.

- **Performance Highlights**: 3D 비전의 다양한 적용 분야에서 기존 연구와 비교하여 데이터셋 및 벤치마크가 모델 성과에 미치는 영향이 강조됩니다. 또한 최근 등장한 상태 공간(state-space) 모델과 같은 새로운 접근 방식들이 컴퓨팅 비용을 절감하면서도 경쟁력 있는 성능을 달성하는 사례를 제시합니다. 이러한 발전은 3D 비전 분야의 접근성을 높이며, 초보 연구자들에게 더 나은 이해를 돕기 위한 중요한 기초 자료를 제공합니다.



### FindIt: A Format-Informed Visual Detection Benchmark for Generalist Multimodal LLMs (https://arxiv.org/abs/2606.04282)
- **What's New**: 최근에 일반화된 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 모델이 사용자가 제공하는 prompt에 따라 다양한 localization 작업을 수행할 수 있도록 설계된 최초의 포괄적 벤치마크를 도입했다. 이 연구는 객체 탐지, 참조 표현 탐지, 인스턴스 탐지 및 비디오 기반 탐지와 같은 네 가지 핵심 작업 범주를 포함하지만, MLLMs의 localization 성능을 체계적으로 평가할 수 있는 기준이 부족했다는 점을 강조했다. 또한, 출력 형식의 규격 준수 능력과 일반적인 오류를 공략하여 기존 데이터셋의 정의와 비교 가능성 문제를 해결하고자 했다.

- **Technical Details**: 이 연구는 일반적인 MLLMs의 promptable localization 능력을 평가하기 위해 고안된 첫 번째 벤치마크를 개발했다. 연구진은 고전적인 데이터셋을 활용하여 네 가지 일반적인 localization 작업(객체 탐지, 참조 표현 탐지, 인스턴스 탐지 및 비디오 객체 탐지)에 대한 평가 프레임워크를 구축했다. 또한 다양한 바운딩 박스 및 구조적 출력 형식에 대한 요구 사항과 함께, MLLMs의 다양한 출력 형식의 대상 작업에 따른 영향을 분석하였다.

- **Performance Highlights**: MLLMs의 평가 결과, 오픈 소스 모델이 폐쇄형 모델보다 더 나은 localization 능력을 보여주었다. 그러나 오픈 소스 모델은 특정 출력 형식에서 벗어나기 힘든 경향이 발견되었고, 폐쇄형 모델은 형식 변형에 대해 더 강인한 모습을 보였다. 전반적으로 인스턴스 탐지 작업에서 모델들의 성능이 저조함을 보여주었으며, 이는 이 작업이 네 가지 작업 중 가장 낮은 성능을 기록했다.



### StandardE2E: A Unified Framework for End-to-End Autonomous Driving Datasets (https://arxiv.org/abs/2606.04271)
- **What's New**: 본 논문에서는 autonomous driving (자율 주행) 분야에 있어 E2E (end-to-end) 모델의 발전을 다룹니다. 연구자들이 다양한 sensor-rich driving datasets를 사용할 때, 각 데이터셋의 파일 형식과 API가 서로 다르기 때문에 적절한 preprocessing을 수행하는 데 많은 시간을 소요해야 했습니다. 이를 해결하기 위해 StandardE2E라는 통합 프레임워크를 제안하여, 데이터셋 간의 차이를 줄이고 사용성을 높입니다.

- **Technical Details**: StandardE2E는 하나의 통합된 데이터 스키마 아래에서 각 데이터셋별 preprocessing을 표준화하며, 여러 데이터셋을 하나의 PyTorch DataLoader로 조합할 수 있도록 합니다. 사용자는 새 데이터셋을 추가할 때 raw frames를 canonical schema로 매핑하는 것만 신경 쓰면 되며, 나머지 파이프라인은 변하지 않습니다. 또한, HD-map, 3D detections, driving command 등 다양한 modality를 지원하여 자율 주행 연구를 더욱 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: 현재 StandardE2E는 Waymo End-to-End, Waymo Perception, Argoverse 2 Sensor 등 6개의 데이터셋을 기본적으로 지원합니다. cross-dataset pretraining과 auxiliary-task supervision을 통해 데이터 처리의 효율성을 크게 향상시킬 수 있으며, 통합된 구조 덕분에 새로운 데이터셋 추가 시 소요되는 비용을 최소화합니다. 이로 인해 연구자들은 더 간편하게 다양한 데이터셋을 활용하여 실험할 수 있게 되었습니다.



### UniCanvas: A Diffusion-base Unified Model for Text-in-Image Joint Generation (https://arxiv.org/abs/2606.04264)
- **What's New**: 최근 통합된 비전-언어 모델(VLMs)에서 주목할 만한 발전이 이루어졌습니다. 이 모델들은 단일 아키텍처 내에서 멀티모달(multi-modal) 이해와 생성을 처리할 수 있습니다. 본 논문에서는 이미지 생성 내에서 언어를 통합하여 'UniCanvas'라는 새로운 모델을 제안하여 텍스트가 이미지 내에서 직접 생성될 수 있도록 하였습니다.

- **Technical Details**: UniCanvas는 공유 픽셀 캔버스(shared pixel canvas)에서 작동하는 단일 확산(diffusion) 모델로, 텍스트와 이미지를 공동으로 학습하는 구조입니다. 텍스트는 구분된 토큰이 아니라 픽셀 렌더링된 기호로 처리되어 이미지 생성을 통해 언어를 표현할 수 있습니다. 이 접근법은 이미지 및 텍스트 생성 간의 간섭을 허용하며, 멀티스텝(multi-step) 작업에서의 추론을 지원합니다.

- **Performance Highlights**: 실험 결과, UniCanvas는 기존 모델들보다 더 높은 정확도와 일관성을 보여주며, 멀티스텝 비주얼 작업에서 우수한 성능을 발휘했습니다. 특히, 생성된 캔버스는 객체의 공간적 배치와 이미지 내 텍스트 주석에서 일관된 상태 전환을 보였습니다. 이러한 결과는 비전과 언어 이해가 통합된 접근 방식을 통해 개선된다는 점에서 중요한 성과로 평가됩니다.



### SBP-Net: Learning Thin Structure Reconstruction with Sliding-Box Projections (https://arxiv.org/abs/2606.04251)
Comments:
          Accepted to IEEE ICIP 2026, 6 pages, 4 figures

- **What's New**: 이 논문에서는 얇은 3D 구조를 재구성하기 위한 새로운 접근 방식을 제안합니다. 기존 방법들이 밀집 표면에서는 잘 작동하지만, 세밀한 구조 재현에서 실패하는 문제를 해결하기 위해 지역 깊이 투영(local depth projections)을 활용합니다. 이 방법은 얇은 구조의 2D 표현을 효율적이고 정보량이 풍부하게 제공하여, 보다 정교한 3D 복원을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 슬라이딩 박스 깊이 투영(sliding-box depth projection) 기법을 사용하여 3D 모델을 탐색하고, 여섯 개의 관점에서 지역 정사ּ 투영(orthographic depth projections)을 생성합니다. 그 후, 이러한 투영은 신경망(neural network)으로 처리되어 진전된 2D 구조를 재구성하고, 완성된 지역은 다시 3D 모델로 통합됩니다. 이렇게하여 원래 데이터의 구조적 무결성을 유지하면서 일관된 3D 형태가 만들어집니다.

- **Performance Highlights**: 실험 결과, CT 볼륨에서 폐동맥 재구성 및 산업용 파이프라인 모델링에 대해 제안된 방법이 기존 방법에 비해 세밀한 구조적 디테일을 보다 잘 보존함을 보여주었습니다. 특히, 이 방법은 희소한 입력 데이터와 불완전한 데이터에서도 탁월한 성능을 발휘하며, 얇은 구조와 경계 연속성을 효과적으로 유지합니다.



### Prospective Dynamic 3D MRI Reconstruction via Latent-Space Motion Tracking from Single Measuremen (https://arxiv.org/abs/2606.04249)
- **What's New**: 이 논문에서 제안하는 PDMR(Prospective Dynamic 3D MRI Reconstruction)는 잠재 공간(latent space) 모션 추적을 활용한 최초의 기대 동적 MRI 재구성 프레임워크입니다. 이 시스템은 이미지를 현재 채집된 측정값으로부터 재구성함으로써 초고속, 모션 인식 MRI 재구합을 가능하게 합니다. 특히, 저차원 매니폴드에서 변형 벡터 필드(deformation vector fields)의 매개변수를 설정함으로써 빠른 온라인 적응을 지원합니다.

- **Technical Details**: PDMR의 핵심 아이디어는 오프라인에서 동작 필드의 효율적이고 일반화 가능한 잠재 매니폴드를 학습하는 것입니다. 이를 통해 PDMR은 실제 임상 환경에서의 초고속 및 내구성 있는 재구성을 가능하게 합니다. 또한, 삼면 표현(tri-plane representation)을 사용하여 3D 모션의 기하학적 정보가 반영된 메모리 효율적 인코딩을 달성합니다.

- **Performance Highlights**: 실험 결과, PDMR은 XCAT 디지털 팬텀과 자체 제작된 복부 MRI 데이터셋 모두에서 높은 충실도와 일관된 재구성을 달성하였습니다. 여러 개의 예상된 시나리오에서 최신 기술보다 뛰어난 성능을 보여주어, 환자 개별 하에서의 영상 진단의 질적 향상 가능성을 시사합니다.



### Overview of the EReL@MIR 2025 Multimodal Document Retrieval Challenge (Track 1) (https://arxiv.org/abs/2606.04240)
Comments:
          MDR Challenge Report at WWW2025

- **What's New**: 이번 논문은 2025년 Web Conference와 공동으로 개최된 첫 EReL@MIR 워크샵에서 진행된 Multimodal Document Retrieval Challenge의 트랙 1에 대해 소개합니다. 이 챌린지는 하나의 시스템을 통해 긴 문서 내에서의 closed-set retrieval과 이미지 또는 이미지-텍스트 쿼리를 통한 open-domain retrieval을 처리해야 하는 두 가지 작업으로 구성됩니다. 455명의 참가자와 586개의 제출물이 있었으며, 시스템들은 평균 Recall@{1,3,5}에 따라 순위가 매겨졌습니다.

- **Technical Details**: 챌린지는 두 가지 작업을 위해 설계되었으며, Task 1인 MMDocIR은 텍스트 쿼리에 따라 단일 긴 문서의 관련 페이지를 순위 매기는 것이고, Task 2인 M2KR은 이미지나 이미지-텍스트 쿼리를 가진 단일 global corpus에서 관련 위키피디아 스타일의 구문을 검색하는 것입니다. 모든 시스템은 Qwen2-VL 계열의 decoder 기반 멀티모달 LLM embedder를 사용하였으며, 우리가 알아본 세 가지 우승 시스템은 서로 다른 방식으로 성능을 높였습니다.

- **Performance Highlights**: 이번 챌린지에서 우승한 세 팀의 시스템은 fine-tuned ensemble, 강력한 vision-language re-ranker를 이용한 training-free multi-route fusion, 또는 zero-shot late interaction을 통해 최고의 성과를 올렸습니다. 특히, training-free 시스템은 fine-tuned 승자와 0.1점 차이로 마무리되었습니다. 이 결과들은 멀티모달 정보 검색의 발전을 위한 중요한 교훈을 제공합니다.



### Spatial Artifact Coherence Determines Codec Robustness in Patch-Based rPPG (https://arxiv.org/abs/2606.04198)
- **What's New**: 이 연구는 Spatial Artifact Coherence (SAC)을 도입하여 codec 압축 하에서 patch-PCA의 실행 가능성을 결정하는 물리적 양을 규명합니다. SAC는 4x4 inter-patch Green-channel 공분산 행렬에서 비대각선 에너지와 대각선 에너지의 비율로 정의되며, 다양한 압축 조건에서의 rPPG 성능을 설명합니다. 또한, SAC가 MPEG-4와 비-MPEG-4 변형 간의 성능 차이를 효과적으로 분류하는 지표임을 보였습니다.

- **Technical Details**: 이 연구에서는 280명의 피험자를 대상으로 11개의 codec 손상 변형과 13개의 알고리즘을 평가하여 SAC가 PCA 이점의 93.8%의 변동성을 설명하는 것으로 나타났습니다. SAC가 0.30 미만이거나 저중간 동작을 갖는 두 가지 운영 조건이 PatchPCA의 이점에 필수적이라고 설정했습니다. 연구 결과에 따르면, MPEG-4는 구조적 영향을 미치며, 원본 codec 상태에 의해 결정됩니다.

- **Performance Highlights**: SAC의 분포는 bimodal하며, 비-MPEG-4는 SAC 0.10-0.18, MPEG-4는 SAC 0.48-0.59로 구분됩니다. PCA의 승률은 비-MPEG-4에서는 84-90%인 반면, MPEG-4에서는 61%로 낮아지고, 평균 MAE 개선이 5.8배 감소했습니다. P-Hybrid가 가장 배포에 강한 알고리즘으로 확인되었으며, 이는 다양한 clinical remote monitoring 시스템에서 codec 인식을 고려한 rPPG 알고리즘 선정에 유용한 물리적 기반 지표를 제공합니다.



### GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs (https://arxiv.org/abs/2606.04184)
Comments:
          Accepted by ACL 2026

- **What's New**: 이 논문에서는 집단 수준의 Theory of Mind (ToM)를 평가하기 위한 최초의 멀티모달 벤치마크인 GroupToM-Bench를 소개합니다. 이 벤치마크는 믿음(belief), 욕망(desire), 의도(intention)라는 미시적 BDI 상태, 집단 긴장과 구조적 제약이라는 중간 수준, 그리고 결과 예측과 기계적 귀속이라는 거시적 수준을 아우르는 인과적 체인을 기반으로 설계되었습니다. 또한, 실험에서는 현재 모델들이 인간 기준선과의 일관된 그룹 인지 격차(Group Cognitive Gap)를 드러내며, 사회적 구조와 비선형적인 집단 역학을 처리하지 못하는 한계를 보여줍니다.

- **Technical Details**: GroupToM-Bench 프레임워크는 MLLMs의 집단 수준 ToM 능력을 평가하기 위해 고안되었습니다. 이 프레임워크는 (i) 다단계 이론적 모델링 레이어, (ii) 인과적 체인을 따라 작용하는 진단 도구로서의 7단계 인지 감사 프레임워크, (iii) GroupToM-Bench의 데이터셋 구축 파이프라인 개요를 포함합니다. 현재 MLLMs의 사회적 추론에서의 주요 실패 모드는 집단 행동을 단순한 개별 의도의 총합으로 잘못 가정하는 선형 초합성 편향(linear superposition bias)이라고 가정합니다.

- **Performance Highlights**: 우리의 실험 결과는 기존 모델들이 격리된 개인의 개인적 동기를 회복하는 데는 능숙하지만, 실제 집단을 정의짓는 비선형 붕괴와 집단 함정을 예측하는 데 실패한다는 것을 보여줍니다. 모델들은 낙관적 합리적 합의에 기본적으로 의존하며, 집단적 사고(Groupthink)와 수혜자의 저주(winner’s curse)와 같은 구조적 함정을 놓치고 있습니다. GroupToM-Bench는 이러한 한계를 측정 가능하게 하고, 사회적으로 기반한 AI의 다음 세대를 위한 진단적 기초를 제공합니다.



### End-to-End Text Line Detection and Ordering (https://arxiv.org/abs/2606.04166)
- **What's New**: 이 논문에서는 역사 문서에 대한 실용적인 텍스트 인식 파이프라인을 소개합니다. 기존의 레이아웃 분석 방법을 대체하는 Orli(Ordered Regression of Lines) 모델이 제안되었으며, 이 모델은 라인 감지(line detection)와 읽기 순서(reading order) 단계를 하나의 이미지에서 시퀀스로 처리합니다. Orli는 페이지 이미지에서 텍스트 라인 기준선을 자동 회귀 방식으로 생성합니다.

- **Technical Details**: Orli 모델은 텍스트 라인 기준선을 코드화된 기하학적 휴리스틱 대신에 초음파 매개변수(chord-frame parameterization)로 표현하여, 라인의 위치 및 방향을 고정하고 지역 기하학(local geometry)을 수직 오프셋(perpendicular offsets)으로 인코딩합니다. 또한, 반복 정제 헤드(iterative refinement head)와 지역 시각 정제기(local visual refiner)가 최종 곡선을 생산합니다. 본 모델은 196,691 페이지로 구성된 이질적인 코퍼스에 대해 훈련되었습니다.

- **Performance Highlights**: Orli는 cBAD 선 감지(line detection)에서 이전에 보고된 최신 기술을 약간 초과하며, 특정 데이터셋에 대한 훈련 없이 여러 읽기 순서 벤치마크에서 거의 완벽한 범위와 순서를 달성했습니다. 또한, 제한적인 미세 조정(fine-tuning)만으로도 특수화된 도메인 외 레이아웃에 적응할 수 있습니다. 모델의 소스 코드 및 가중치는 오픈 라이센스 하에 공개되어 사용 가능합니다.



### Pinpoint: Grounded Worldwide Image Geolocation via Cross-Source Retrieval and Reranking (https://arxiv.org/abs/2606.04133)
- **What's New**: 이번 논문에서는 이미지 지리적 위치 추정 문제를 해결하기 위해 Pinpoint라는 새로운 아키텍처를 제안합니다. 기존의 인터넷 사진과 거리뷰 이미지를 별도의 작업으로 분리했던 점을 개선하여 두 데이터 소스를 결합한 모델입니다. Pinpoint는 사용자 업로드 데이터와 거리뷰 이미지를 통해 GPS 임베딩 공간을 학습하여 후보 위치를 효과적으로 추출하고 재순위화(re-rank)합니다.

- **Technical Details**: Pinpoint는 retrieve-and-rerank 파이프라인을 사용합니다. 첫 번째 단계에서는 사용자 데이터를 기반으로 유사도를 통해 후보 위치를 검색하는 contrastive image-GPS embedder를 학습합니다. 두 번째 단계에서는 attention-based reranker가 각 후보 위치의 시각적 특성과 GPS 정보를 통합하여 점수를 재조정합니다. 이 모든 과정은 멀티모달 대형 언어 모델(MLLM)에 의존하지 않고 수행됩니다.

- **Performance Highlights**: Pinpoint는 인터넷 사진과 거리뷰 이미지에 대해 모든 표준 벤치마크에서 최고 성능을 기록했습니다. IM2GPS3k, YFCC4k 데이터셋에서는 모든 거리 기준에서 최첨단 성과를 내었으며, 거리뷰 지리 위치추정 OSV-5M 테스트 세트에서도 새로운 기록을 세웠습니다. 이를 통해 Pinpoint의 효과성을 입증했습니다.



### Reflection Separation from a Single Image via Joint Latent Diffusion (https://arxiv.org/abs/2606.04107)
Comments:
          CVPR 2026. Project page: this https URL

- **What's New**: 본 논문은 강력한 생성적 사전( generative priors )을 활용하여 단일 이미지 반사 분리(single-image reflection separation)를 위해 명시적으로 조정된 확산 모델을 제안합니다. 기존 방법들이 글레어(glare)나 약한 반사(weak reflection) 환경에서 효과적으로 작동하지 못하는 문제를 해결하기 위해, 저자는 통합된 확산 모델을 통해 전송(transmission) 및 반사(reflection) 레이어를 동시 생성할 수 있도록 합니다. 또한 새로운 교차 레이어 자기 주목(cross-layer self-attention) 메커니즘을 도입해 더욱 나은 특징 분리(feature disentanglement)를 촉진합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 Latent Diffusion Model (LDM)을 활용하여 이미지를 전송 레이어와 반사 레이어로 분해합니다. LDM은 데이터를 압축하기 위해 사전 훈련된 VAE 인코더(encoder)를 사용하며, 본 논문의 혁신은 이 모델에 새로운 샘플링 전략을 추가해 서로의 간섭을 줄이는 것입니다. 이는 다운스트림 다운샘플링 및 레이어 간 상호 작용( interaction )을 최적화하여 복잡한 실제 환경에서도 성능을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 여러 실제 벤치마크에서 최첨단 방법보다 뛰어난 성능을 발휘하는 것을 입증합니다. 단일 이미지를 처리하는 동안, 모델은 고반사 시나리오에서 사라진 세부사항을 복원하고, 약한 반사 시나리오에서는 의미 있는 반사 내용을 정확하게 포착합니다. 최적화 과정에서 계산 효율성을 크게 줄임으로써, GPU 메모리 요구 사항을 낮춰 실용적인 적용이 가능하다는 점도 큰 장점입니다.



### When Seeing Is Not Believing -- A Benchmark for Search-Grounded Video Misinformation Detection (https://arxiv.org/abs/2606.04098)
Comments:
          52 pages

- **What's New**: 이번 논문에서는 새롭게 EVID-Bench라는 비디오 잘못된 정보 탐지를 위한 기준을 제시합니다. 이 기준은 검색 기반(search-grounded) 접근 방식을 채택하여, 조작된 비디오를 분석하기 위해 웹에서 관련 비디오를 검색하고 비교하는 방식을 사용합니다. 연구자들은 총 222개의 조작된 비디오 샘플을 수집하였고, 이는 AI 생성, 단일 출처 편집, 다중 출처 편집 등 9가지 조작 유형을 포함합니다.

- **Technical Details**: EVID-Bench는 비디오 잘못된 정보의 탐지 및 확인 과정을 체계적으로 다루고 있습니다. 이 시스템은 단순한 비디오 탐지를 넘어 외부 비디오 증거를 활용해 교차 비디오 비교를 통해 잘못된 정보를 식별해야 합니다. 이에 따라, 기존의 전통적인 포렌식 방법들이나 딥페이크 탐지기술은 다루지 못하는 세련된 조작을 탐지하는 데 중점을 두고 있습니다.

- **Performance Highlights**: EVID-Bench를 활용한 실험에서는 아홉 개 선도 모델들이 평가되었습니다. 그 결과, 가장 성능이 우수한 모델조차도 비디오 수준의 정확도가 43.24%에 불과하며, AI 생성 조작의 탐지는 특히 어려운 것으로 나타났습니다. 이러한 실험을 통해, 모델들이 잘못된 정보 탐지에서 만나는 주요한 도전 과제들을 드러냈습니다.



### Optimal Transport Flow Matching by Design (https://arxiv.org/abs/2606.04092)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 flow matching을 통해 샘플을 단순한 prior 분포에서 복잡한 데이터 분포로 효과적으로 전송하는 방법을 제안합니다. 기존의 최적 운송(Optimal Transport, OT) 문제를 해결하기 위해 prior을 고정된 입력이 아닌 설계 선택으로 간주하여, 다양한 prior가 데이터와의 OT-최적성 결합을 허용한다고 주장합니다. 이 접근법에 의해 데이터와 저주파수 표현(low-frequency representation) 간의 간단한 결합이 획득되어, 고차원 데이터 분포에서의 복잡성을 줄입니다.

- **Technical Details**: 제안된 방법은 고차원 공간의 OT 결합 계산을 피하여 prior 분포를 변환하여 OT 최적 솔루션을 설계합니다. 자연 이미지의 저주파수 프로젝션(low-frequency projection)을 선택하여, 이는 가벼운 모델에서 샘플링할 수 있는 prior를 생성합니다. 이 방법은 저차원 공간에서만 대략적인 이미지 구조를 생성하며, 이후 고해상도에서 고주파 세부정보를 합성하여 generation 문제를 두 단계로 나눕니다.

- **Performance Highlights**: 제안된 접근법은 기존 flow matching 방법과 비교하여 궤적의 곱셈(curvature)을 2배 이상 줄여줍니다. CIFAR-10, FFHQ, 그리고 ImageNet에서 실험을 통해 강력한 1단계 및 몇 단계 생성(high-quality generation) 성능을 보여주으며, Latent-space models, classifier-free guidance, 및 MeanFlow와 같은 단일 단계 프레임워크와의 통합이 자연스럽습니다.



### Intra-Modal Neighbors Never Lie: Rectifying Inter-Modal Noisy Correspondence via Graph-Based Intra-Modal Reasoning (https://arxiv.org/abs/2606.04061)
- **What's New**: 본 논문은 기존의 'Discrete Selection' 패러다임의 한계를 극복하기 위해 Intra-modal Neighbor-aware Noise Rectification (IN2R)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대체 레이블을 찾는 대신, 신뢰할 수 있는 감독 목표를 합성하는 방향으로 접근합니다. 특히, IN2R은 이른바 'Single-Point Fragility'와 'Discretization Error' 문제를 해결하며, 이미지와 텍스트 간의 본질적인 기하학적 안정성을 활용합니다.

- **Technical Details**: IN2R은 그래프 정제기(Graph Refiner)를 사용하여 동적 크로스 모델 메모리에서 인접 이웃들에 대한 관계적 추론을 수행합니다. 이 방법은 개별 표본의 불확실성을 통계적으로 줄이며, 이웃의 레이블을 단순히 재사용하는 대신, 연속적인 프로토타입을 합성하여 더 정확한 감독 신호를 제공합니다. 또한, IN2R은 듀얼 피어 네트워크를 활용하여 고신뢰성의 클린 샘플을 동적으로 큐레이션합니다.

- **Performance Highlights**: Flickr30K, MS-COCO, CC152K의 광범위한 실험에서 IN2R은 기존의 방법들을 크게 초월하는 성능을 보였습니다. 특히, 높은 노이즈 환경에서도 IN2R은 우수한 결과를 나타내며, 제안한 접근법의 효과성을 입증합니다. 이로써, IN2R은 고품질 이미지-텍스트 짝을 위한 강력한 해결책으로 자리매김할 것으로 기대됩니다.



### Weakly Supervised Incremental Segmentation via Semantic Anchors and Spatial Arbitration (https://arxiv.org/abs/2606.04060)
Comments:
          Accepted by ICME2026

- **What's New**: 본 논문에서는 약한 감독 하에서의 점진적 학습(Weakly Incremental Learning for Semantic Segmentation, WILSS)에 대한 새로운 접근법인 SASA를 제안했습니다. SASA는 의미적 앵커(Semantic Anchors)와 공간 조정(Spatial Arbitration)을 통해 학습의 안정성을 증가시키고, 의미적인 흐름을 유지하는 데 중점을 두고 있습니다. 이 방법은 기존의 문제점인 피쳐 드리프트(Feature Drift)와 의미적 왜곡(Semantic Corruption)을 방지하는 데 효과적입니다.

- **Technical Details**: SASA의 핵심은 학습 가능한 의미적 앵커를 사용하여 클래스 수준의 참조를 안정적으로 유지하는 것입니다. 이는 고정된 구역을 정의하고, 유연한 잔차 보정(Elastic Residual Adaptation)을 통해 순간적인 변화를 조정합니다. 또한, 공간 레이블 조정(Spatial Label Arbitration) 메커니즘은 기하학적 결정을 통해 신뢰할 수 없는 신호를 필터링하고, '각 객체에 하나의 클래스'라는 제한을 시행하여 학습 과정을 보호합니다.

- **Performance Highlights**: 철저한 실험 결과, SASA는 표준 벤치마크에서 기존 최첨단 방법들보다 일관되게 우수한 성능을 보였습니다. 특히, 다단계 점진적 설정에서의 도전적인 환경에서도 뚜렷한 성과를 발휘하며, 이를 통해 학습하는 동안의 무질서를 효과적으로 완화하였습니다. SASA는 동적인 학습 환경에서도 견고한 성능을 지속적으로 입증하고 있습니다.



### Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation (https://arxiv.org/abs/2606.04046)
Comments:
          Accepted at ICML 2026

- **What's New**: 본 논문에서는 SceneDiver라는 새로운 방법을 제안하여, 시각-언어 의사결정(vision-language decision making) 과제에서의 인식 한계(perceptual limitation)를 극복하고자 합니다. 기존의 시각-언어 모델(VLMs)과 시각-언어-행동 모델(VLAs)이 각각의 장점을 갖고 있지만, 시각적 환각(visual hallucinations) 문제로 인해 성능 제한을 겪고 있습니다. SceneDiver는 장기 계획 능력을 활용하여, 먼저 전체 장면 그래프(scene graph)를 구축하고 이를 통해 작업을 간단한 하위 문제로 분해하는 방식으로, 효과적으로 중요한 객체에만 집중할 수 있도록 설계되었습니다.

- **Technical Details**: SceneDiver의 중심은 거친 단계에서 세밀한 단계로 진행되는 초점(focus) 계획 수립입니다. 첫 번째 단계에서는 이미지 데이터를 구조화된 그래프 표현으로 변환하여 장면을 전반적으로 이해합니다. 두 번째 단계에서는 VLM이 각 지역 하위 장면을 탐색하여 중요한 객체를 식별하도록 합니다. 또한, 실시간 의사결정에 필요한 지연 시간을 충족하기 위해 가벼운 어댑터(adapter)를 설계하여 VLA 모델에서 효과적인 초점 능력을 추출합니다.

- **Performance Highlights**: 다양한 로봇 조작 및 방 탐색 과제를 통해 실험한 결과, SceneDiver는 조작 작업에서 10%-15%, 탐색 작업에서 최대 16%의 성능 향상을 보여주었습니다. 또한, LIBERO-plus 벤치마크에서 성공률이 9.6% 개선되었으며, 이는 의사결정의 강건성을 향상시키는데 기여했습니다. 이 모든 성능 향상과 함께 계산 효율성도 유지되었으며, 실시간 배포에 적합합니다.



### Geometry Gaussians: Decoupling Appearance and Geometry in Gaussian Splatting (https://arxiv.org/abs/2606.05124)
- **What's New**: 이 논문은 3D Gaussian Splatting (3DGS)의 기본 형태가 텍스처와 지오메트리를 동시에 표현하기에 적절하지 않다는 점을 강조합니다. 저자들은 각 스플랫에 추가적인 geometry opacity 파라미터를 도입하여 렌더링과 지오메트리 정보를 분리할 수 있는 간단한 해결책을 제안합니다. 이 방식은 복잡한 장면에서 투명한 객체의 처리에 특히 유리하여, 렌더링 품질을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 기존 3DGS에서는 각 스플랫이 렌더링과 장면 지오메트리에 대한 단일 불투명도(opacity) 파라미터를 사용합니다. 이 논문에서는 이를 분리하여 추가적인 geometry opacity를 부여함으로써, 깊이와 법선과 같은 기하학적 양에만 책임을 지도록 설계합니다. 실험 결과, 이 단순한 변경이 렌더링과 지오메트리 재구성의 성능을 유의미하게 개선시킨다는 것을 보여주었습니다.

- **Performance Highlights**: 저자들은 NeRF Synthetic, DTU, TransLab, Mip-NeRF와 같은 다양한 데이터셋에서 우리의 접근법이 최첨단 결과를 달성했다고 보고합니다. 특히, 저자들은 투명한 객체를 포함한 복잡한 장면에서 기존 방법들이 놓치는 geometry의 많은 부분을 회복하고 렌더링 품질을 동시에 향상시킨다고 주장합니다. 이 연구는 3DGS의 렌더링 불투명도와 geometry 불투명도를 명시적으로 분리하는 것이 향후 다양한 응용 프로그램에 유용한 단계가 될 것이라고 믿습니다.



### Identifying Gems from Roman RAPIDly (https://arxiv.org/abs/2606.05103)
Comments:
          15 pages, 10 figures, Submitted to the Publications of the Astronomical Society of the Pacific

- **What's New**: 이번 연구는 NASA의 Nancy Grace Roman Space Telescope가 발사 예정인 2026년 9월에 방대한 적외선 이미징 조사를 통해 수백만 개의 천문학적 과도(transient) 현상을 발견할 수 있도록 자동화된 경고 생성 파이프라인을 개발하는 데 중점을 두고 있습니다. 이 작업에서 새로운 기계 학습 모델 RuBR를 소개하며, 실제 데이터가 없는 상황에서도 가짜(boogus) 탐지를 구별하고 신뢰할 수 있는 변동 탐지를 지원할 수 있는 방법론을 제시합니다.

- **Technical Details**: RuBR 모델은 세 가지 접근 방식을 갖추고 있습니다: RuBR_comb은 여러 데이터를 결합하여 학습하고, RuBR_loc은 국소적으로 주입된 데이터를 학습하며, RuBR_DA는 도메인 적응 모드를 사용하여 데이터를 결합하여 학습합니다. 이러한 모델은 Roman의 초기 임무 동안 실제 관측 데이터가 없더라도 실제 데이터에 적응할 수 있도록 함으로써, 더 많은 세부 사항을 이해하는 기반을 제공합니다. 실험적 배경에서도 각 모델 성능을 검증하고, 전반적인 클래시피케이션 작업에 대한 신뢰성을 높입니다.

- **Performance Highlights**: 제안한 RuBR 모델은 기존의 베이스라인 모델과 비교하여 우수한 성능을 보여주며, 실험 결과는 매우 높은 신뢰도를 바탕으로 한 결과를 나타냅니다. 다가오는 Roman 시대에 있어 real-bogus classification의 가능성을 증대시키며, 초기 관측 데이터를 통해 과도 현상 탐지의 신뢰성을 강화할 것입니다. 이러한 성과는 향후 천문학적 연구와 발견에 중요한 기여를 할 것으로 기대됩니다.



### Toward Multi-Domain and Long-Tailed Quantization via Feature Alignment and Scaling (https://arxiv.org/abs/2606.04920)
- **What's New**: 이번 연구에서는 Efficient Multi-Domain Alignment Quantization (EmaQ)와 장기 꼬리(Long-Tailed) 양자화를 위한 EmaQ-LT를 제안합니다. 이 방법들은 서로 다른 도메인 분포를 정렬하고 불균형한 클래스 문제를 해결하여 양자화의 성능을 개선합니다. EmaQ는 도메인 간의 일관성을 유지하고 다중 도메인에서의 수렴을 안정화하는 알고리즘을 포함하여 실용적인 양자화 환경을 타겟팅합니다.

- **Technical Details**: EmaQ는 누적 분포 함수(CDF)를 기반으로 도메인 분포를 통합하고, Sensitivity-aware Weight Aggregation (SWA) 메커니즘을 사용하여 업데이트에 대한 도메인 별 민감성을 고려합니다. 이는 다중 도메인에서의 훈련 수렴을 조절하고 불리한 양자화 기준으로 인해 발생하는 편향을 완화합니다. EmaQ-LT는 장기 꼬리 데이터에 발맞추어 클래스 별 분산 스케일링과 불균형한 클래스 가중 조정을 통해 소수 클래스의 성능을 향상시키는 방법론을 포함하고 있습니다.

- **Performance Highlights**: EmaQ와 EmaQ-LT는 표준, 다중 도메인 및 장기 꼬리 데이터셋에서 강력한 성능을 보여줍니다. 특히, 제안된 방법들은 기존의 양자화 기법들보다 더 나은 성과를 달성하며, 도메인 전환 및 클래스 불균형 조건에서도 견고한 저비트 성능을 유지합니다. 실험 결과는 다양한 벤치마크를 통해 입증되었으며, 소수 클래스의 정확도 개선을 확인할 수 있었습니다.



### Drift-Augmented Scoring: Text-Derived Noise Robustness for Zero-Shot Audio-Language Classification (https://arxiv.org/abs/2606.04844)
- **What's New**: 이 논문에서는 Drift Augmented Scoring (DAS)라는 새로운 방법을 제안합니다. DAS는 노이즈 속에서도 오디오 임베딩이 예상되는 클래스 방향으로 이동할 때 소량의 보너스를 추가하여 정확도를 향상시키는 데 중점을 둡니다. 기존 오디오-언어 모델(CLAPlike) 기반의 제로샷 오디오 분류 성능이 저하되는 문제를 해결하고자 했습니다.

- **Technical Details**: DAS는 훈련 데이터 없이 텍스트에서 파생된 클래스별 드리프트 방향을 이용하여 성능을 개선합니다. 이 방법은 각 클래스에 대해 작은 보너스를 추가하며, 실제 오디오 임베딩에 영향을 주지 않고 단 한 번의 내적 연산으로 점수를 계산합니다. 이는 테스트 시간에 배치 없이 수행되며, 각 클래스의 드리프트 방향을 고려하여 점수를 업데이트합니다.

- **Performance Highlights**: DAS는 UrbanSound8K와 FSD50K 평가 세트에서 모든 테스트 조건에서 성능을 개선했습니다. UrbanSound8K에서는 정확도가 +2.60에서 +5.75 포인트 개선되었고, FSD50K에서는 mAP가 +1.50에서 +1.74 포인트 증가했습니다. 이는 DAS가 소음이 있는 환경에서도 보다 견고한 오디오 분류를 가능하게 함을 나타냅니다.



### Activation Steering of Video Generation Models via Reduced-Order Linear Optimal Contro (https://arxiv.org/abs/2606.04775)
- **What's New**: 본 논문에서는 텍스트-비디오(T2V) 모델의 안전성을 개선하기 위해 새로운 방법을 제안합니다. 이전 방법론들은 모델의 가중치를 변경하거나 재훈련을 요구했지만, Latent Activation Linear-Quadratic Regulator (LA-LQR)는 더욱 최소 침습적으로 모델을 조정할 수 있는 체계적 기법을 제공합니다. 이 접근법은 T2V 추론을 동적 시스템으로 모델링하여 비디오 품질과 프롬프트 충실도를 유지하면서도 원하지 않는 결과를 줄이도록 설계되었습니다.

- **Technical Details**: LA-LQR는 T2V 모델의 활성화를 저차원 잠재 공간으로 투영하여, 불필요한 개입을 최소화하면서 목표 설정점으로 활성화를 유도하는 피드백 신호를 계산합니다. 이론적으로, LA-LQR는 잠재 벡터의 운동 방정식을 활용하여 시스템의 동적 특성을 설명하고, 개입의 정확성을 향상시키는 방법론적 토대를 제공합니다. 이러한 최적 제어 문제를 작업 관련 하위 공간에서 해결하여, 모델의 복잡성을 효과적으로 감소시킵니다.

- **Performance Highlights**: 제안된 LA-LQR은 개념 제어 및 비디오 안전 기준에서 기존 방법보다 안전 생성물의 비율을 감소시켰으며, 프롬프트에 대한 충실도와 비주얼 품질을 유지합니다. 실험 결과는 LA-LQR의 효과성이 기존의 T2V 조정 및 안전 기준을 초월함을 입증합니다. 이는 T2V 모델이 학습한 불안전한 개념을 효과적으로 조정할 수 있는 가능성을 제시합니다.



### Measuring Model Robustness via Fisher Information: Spectral Bounds, Theoretical Guarantees, and Practical Algorithms (https://arxiv.org/abs/2606.04767)
Comments:
          35 pages, 1 figure

- **What's New**: 본 논문은 적대적 공격(Adversarial Attack)에 의존하지 않는 정량적 척도를 제시하여 딥 신경망의 견고성(Robustness)을 정밀하게 평가할 수 있는 새로운 접근법을 소개합니다. 이는 Fisher Information Matrix(FIM)의 스펙트럼 노름(Spectral Norm) 기반의 메트릭으로, 모델의 출력 분포의 최악의 민감도를 정량화합니다. 주요 기여로는 다양한 아키텍처에 대해 첫 번째 이론적 견고성 순위를 제공하고, 실제 공격 기반 평가와 비교해도 잘 작동하는 진단 도구로서 기능하는 것입니다.

- **Technical Details**: FIM의 스펙트럼 노름은 입력의 섭동에 대한 모델의 출력 분포의 민감도를 측정합니다. 이 측정법을 통해 VGG, ResNet, DenseNet 및 Transformer와 같은 여러 신경망 아키텍처에 대한 정량적 분석을 수행합니다. 이와 함께, white-box 및 black-box 환경에서의 대규모 모델 평가를 지원하는 효율적인 알고리즘이 개발되었습니다.

- **Performance Highlights**: 다양한 데이터셋(CIFAR, ImageNet, 의료 이미지)과 아키텍처를 통한 폭넓은 실험을 진행하여, 본 메트릭이 적대적 취약성(Adversarial Vulnerability)과 강한 상관관계를 나타냄을 확인하였습니다. 또한, 이 메트릭은 모델의 설계 및 평가에 있어 더 강력한 대안을 제시하며, RobustBench와 같은 실증적 기준과 함께 사용될 수 있습니다.



### Graph-Guided Universum Learning in Generalized Eigenvalue Proximal SVMs for Alzheimer's Disease Classification (https://arxiv.org/abs/2606.04699)
- **What's New**: 이 논문에서는 치매 예방을 위한 알츠하이머병(AD)의 조기 진단의 중요성을 강조하고 있습니다. 제안된 UG-GEPSVM 및 IUG-GEPSVM 모델은 기존의 Universum 샘플을 보다 효과적으로 활용하는 방법을 제시합니다. 특히, 이러한 모델은 MCI(경도 인지 장애) 샘플을 포함해 AD와 CN(인지 정상) 간의 미묘한 관계를 고려하여 성능을 극대화합니다.

- **Technical Details**: UG-GEPSVM과 IUG-GEPSVM는 각기 다른 방법으로 GEPSVM 모델을 개선합니다. UG-GEPSVM은 그래프 기반의 Laplacian 정규화를 일반화된 고유값 문제로 통합하여 AD와 CN 간의 구분을 수행합니다. 반면 IUG-GEPSVM은 안정적인 수치 해석을 위해 IGEPSVM 확장 버전을 기반으로 하여 문제를 단순화합니다.

- **Performance Highlights**: 실험 결과, 제안된 두 모델 모두 기존의 GEPSVM 및 Universum 기반 방법보다 일관되게 우수한 성능을 보여줍니다. 특히 UG-GEPSVM은 88.07%의 높은 평균 AUC를 기록하며, 잡음이 증가하는 조건에서도 안정적인 성능을 유지합니다. 이는 ADNI MRI 데이터 세트를 활용한 여러 실험을 통해 통계적으로 유의미한 결과로 입증되었습니다.



### Fine-grained Fragment Retrieval in Multi-modal Long-form Dialogues (https://arxiv.org/abs/2606.04591)
- **What's New**: 본 논문에서는 Fine-grained Fragment Retrieval (FFR)라는 새로운 작업을 도입하여, 다중 모드(long-form multi-modal) 대화에서 의미적으로 일관된 발화-이미지 조각을 직관적으로 찾아내는 방법을 제안합니다. 기존 다이얼로그 검색 방식은 일반적으로 개별 발화나 이미지를 선택하는 데 초점을 맞추는 반면, FFR은 여러 대화 턴과 모드를 아우르는 일관된 의미 조각을 찾아 사용자 쿼리에 응답하려고 합니다. 이를 통해 복잡한 다중 모드 대화에서 유용한 정보를 보다 효율적으로 검색할 수 있게 됩니다.

- **Technical Details**: FFR을 지원하기 위해 MLDR이라는 대규모 다중 모드(long-form multi-modal) 대화 검색 데이터셋을 구축하였으며, 이는 평균 25.45턴으로 가장 긴 다이얼로그를 담고 있습니다. FFR은 단일 대화에서의 검색뿐만 아니라, 대화 코퍼스에서의 검색을 평가하기 위한 두 가지 설정을 탐구합니다. 이 과정에서 Fragment Embedding Model (FEM)을 사용하여 각 대화를 최소한의 의미적 단위로 분해하고, 쿼리에 대한 응답을 빠르게 검색할 수 있도록 구조화된 인덱싱을 적용합니다.

- **Performance Highlights**: F2RVLM과 FFRS는 MLDR과 실제 WeChat 기반 테스트 세트에서 모두 탁월한 성능을 입증하며 단일 대화 및 대화 코퍼스의 FFR 접근 방식에서 높은 정확도를 달성하였습니다. FFRS는 효율적인 검색과 의미적 정밀도 간의 균형을 적절히 유지하며, 실제 대화 시나리오에서도 효과적임을 보여줍니다. 이러한 성과는 FFR이 실제 응용 프로그램에서 유용하게 활용될 가능성을 시사합니다.



### Echo-Infinity: Learning Evolving Memory for Real-Time Infinite Video Generation (https://arxiv.org/abs/2606.04527)
Comments:
          Website: this https URL

- **What's New**: Echo Infinity는 실시간 무한 비디오 생성(real-time infinite video generation)을 위한 자가 회귀(autoregressive) 프레임워크입니다. 이 프레임워크는 동적으로 필터링하고 추상화하며 압축하는 학습 가능한 진화 메모리(learnable evolving memory)를 사용하여 역사 정보를 손실 없이 처리할 수 있습니다. 기존의 방법들과 달리 Echo Infinity는 수동으로 정의된 메모리 큐레이션을 대체하고, 주의(attention)와 게이팅 메커니즘(gating mechanism)을 통해 메모리를 최적화합니다.

- **Technical Details**: Echo Infinity는 정보를 지속적으로 업데이트하여 불필요한 과거 프레임을 삭제하는 과정에서 주의 메커니즘을 통해 메모리 쿼리(Memory Query)를 사용합니다. 이를 통해 압축 비율(compression ratios)을 자유롭게 조절할 수 있으며, 일정한 계산(computation) 비용으로 다양한 길이의 비디오를 처리합니다. 또한, 통합 상대 RoPE 레시피(Unified Relative RoPE Recipe)를 도입하여 근본적으로 RoPE 제약(finite RoPE constraint)에서 벗어나고 훈련(training) 및 추론(inference) 과정에서의 RoPE 보간(interpolation) 차이를 줄입니다.

- **Performance Highlights**: Echo Infinity는 짧은 비디오와 긴 비디오 생성 모두에서 최첨단 성능을 보여줍니다. 특히, 24시간 동안 1.3M 프레임 이상의 실시간 롤아웃을 성공적으로 수행한 첫 사례를 제시하여 무한 비디오 생성에 대한 실용적인 가능성을 제시합니다. 이는 비디오 생성 분야에서 중요한 진전을 나타내며, 비디오 품질(quality)을 더욱 개선하는 데 기여합니다.



### L-TGVN: Leveraging Longitudinal Priors for Personalized Rapid MRI (https://arxiv.org/abs/2606.04419)
Comments:
          Accepted to MICCAI 2026

- **What's New**: 본 연구에서는 L-TGVN(긴급 신뢰 기반 가변 네트워크)을 소개합니다. 이 네트워크는 이전 스캔을 부가 정보로 활용하여, 샘플 수가 크게 줄어든 측정값으로부터 현재 스캔을 재구성합니다. 기존의 방법들과 달리, L-TGVN은 이전 스캔과 현재 스캔 사이의 사전 정합(pre-registration)을 필요로 하지 않습니다.

- **Technical Details**: L-TGVN은 측정된 데이터와 이전 스캔 간의 일관성을 유지하면서 이전 스캔의 영향을 제어합니다. 이는 영상 진단 품질을 위해 필수적인 정보와 컨텍스트를 제공하고, 경과에 따른 변화를 고려합니다. 또한 이전 방문 시의 프로토콜 차이에도 유연하게 대처할 수 있는 점이 특징입니다.

- **Performance Highlights**: L-TGVN은 이전 정보 기반 방법 및 장기적 사전 정보를 사용하지 않는 방법들과 비교하여 우수한 성능을 보여주었습니다. 특히, 도전적인 가속 상황에서도 미세 구조(fine structures)를 잘 보존하면서 정량 지표(quantitative metrics)에서 지속적인 개선이 관찰되었습니다. 이 알고리즘의 소스 코드는 링크를 통해 제공됩니다.



### PureLight: Learning Complex Luminaires with Light Tracing (https://arxiv.org/abs/2606.04319)
Comments:
          9 pages, 10 figures

- **What's New**: 이 논문에서는 복잡한 조명기구의 외관을 추정하는 새로운 신경망 기반 공식을 제안합니다. 이 연구는 복잡한 광 전달(light transport)을 가진 조명기구에 집중하며, 특히 작은 방출기로 구성된 복합적인 구조를 다룹니다. 이를 위해 빛 추적(light tracing)을 사용하여 방출기에서 출구 표면까지의 경로를 구성하며, 외관 추정을 확률 밀도 함수(Probability Density Function, pdf)로 모델링합니다.

- **Technical Details**: 제안된 방법은 복잡한 조명기구의 출발 복사(출구에서 방출되는 빛의 양)를 pdf와 플럭스의 곱으로 복구하는 방식으로 구성되어 있습니다. 이 과정에서 대규모 정규화 흐름 네트워크(normalizing flow network)를 사용하여 pdf를 모델링하고, 라디언스를 추정합니다. 또한 학습된 정보를 경량 다층 퍼셉트론(MLP)으로 증류하여 직관적인 추론을 가능하게 하며, 조명 기구의 직접적인 조도 계산을 위한 샘플링 네트워크 및 장면에 조명 기구를 통합하기 위한 블렌딩 네트워크도 훈련합니다.

- **Performance Highlights**: 이 연구는 다양한 3D 장면에서 저 샘플 수로도 복잡한 조명기구의 렌더링이 가능하다는 것을 입증합니다. 제안된 분포 학습 프레임워크는 전통적인 몬테카를로 기법으로 추정하기 어려운 복잡한 조명기구의 외관을 효율적으로 모델링할 수 있습니다. 마지막으로, 입력 샘플 수가 적을 때 노이즈를 낮추는 효과적인 방법을 제시하여, 높은 품질의 결과를 제공할 수 있음을 보여줍니다.



### Instant-Fold: In-Context Imitation Learning for Deformable Object Manipulation (https://arxiv.org/abs/2606.04269)
- **What's New**: 이번 논문에서는 Deformable Object Manipulation (DOM)을 위한 새로운 프레임워크인 Instant-Fold를 제안합니다. Instant-Fold는 단일 인간의 시연을 통해 다양한 조작 모드를 추론하고 실행할 수 있는 imitation learning의 한 형태입니다. 특히, 이 방법은 gradient 업데이트 없이도 다양한 조작 모드를 직접적으로 인식하고 적용할 수 있는 장점이 있습니다.

- **Technical Details**: Instant-Fold는 시간 대비 대조적 사전학습(temporal contrastive pretraining)을 통해 변형에 대한 인지 시각 표현을 학습합니다. 이후, 시연을 조건으로 하는 flow-matching transformer 정책을 이용해 조작 모드에 따라 동작을 예측합니다. 이 프레임워크는 의류 접기와 같은 시뮬레이션 작업에 적용되며, 3D 위치와 사전 훈련된 변형 인식 의미적 특징을 결합한 geo-semantic token으로 물체를 표현합니다.

- **Performance Highlights**: Instant-Fold는 실제 데이터를 추가 수집하거나 미세 조정(finetuning) 없이도 다양한 folding 모드에 일반화하며 실세계 환경에서도 성능을 발휘합니다. 이는 기존의 복잡한 조작 절차를 단순화하고, 하나의 시연만으로도 새로운 동작을 효과적으로 학습할 수 있도록 합니다. 이 연구는 수많은 유사한 연구와 차별화되는 방향으로 DOM 분야에서의 새로운 가능성을 제시하고 있습니다.



### Can Generalist Agents Automate Data Curation? (https://arxiv.org/abs/2606.04261)
Comments:
          Preprint

- **What's New**: 본 논문에서는 훈련 데이터를 정리하는 과정의 자동화를 위한 일반화된 코딩 에이전트의 가능성을 탐구합니다. 이를 위해 Curation-Bench라는 에이전트 중심의 벤치마크를 도입하여, 에이전트가 특정 데이터를 관찰하고 정책을 구현하며 훈련 및 평가 파이프라인에 제출하고 수정하는 등의 명령어 접근이 가능하도록 합니다. 이 연구를 통해 에이전트가 강력한 데이터 선택 기준을 신속히 도달할 수 있음을 보여줍니다.

- **Technical Details**: Curation-Bench는 모델과 훈련 레시피, 평가 스위트를 고정하여 에이전트가 데이터 inspect와 정책을 구현할 수 있도록 합니다. 에이전트는 데이터 예산의 10분의 1로 강력한 기준을 초월하는 데이터 선택 정책을 자율적으로 구성할 수 있습니다. 반복적인 정책 조정뿐만 아니라 이전 방법을 인용하고 적용하는 스캐폴드를 통해 에이전트가 보다 효과적으로 탐색할 수 있도록 유도합니다.

- **Performance Highlights**: 에이전트는 10번의 반복 내에 출판된 데이터 선택 기준에 도달하는 성과를 보였으나, 분석 결과 에이전트는 새로운 정책 패밀리보다는 로컬 정책 변형에 주로 집중함을 알 수 있었습니다. 스캐폴드가 없는 경우, 에이전트는 전략 가이드와 논문 참조를 제공받아도 새로운 접근 방식을 탐색하지 못하는 경향이 있습니다. 본 연구는 현재의 에이전트가 데이터 정리 루프를 실행할 수 있으나, 신뢰할 수 있는 데이터 연구에는 스캐폴드 방식의 방법 적응이 필요하다는 점을 강조합니다.



### VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark (https://arxiv.org/abs/2606.04244)
- **What's New**: 본 연구에서는 VAMPS(Visual-Assisted Mathematical Problem Solving)라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 이란 대학 입학 시험의 대수학과 미적분학 문제를 바탕으로 하며, 총 1,168개의 다중 모드의 QA 쌍을 포함하고 있습니다. VAMPS의 핵심은 플롯을 사용하여 시각적 추론을 평가하는 유일한 페르시아-영어 벤치마크라는 점에서 중요합니다.

- **Technical Details**: VAMPS는 주어진 문제를 정보가 풍부한 플롯으로 변환하고, 결과적으로 생성된 플롯을 통해 최종 결정을 내릴 수 있는 모델의 능력을 평가합니다. 이 과정에서 Desmos라는 그래프 도구를 사용하여 시각적 표현을 생성하며, 모델의 도구 사용 행동 및 최종 답변의 정확성을 면밀히 분석할 수 있습니다. 벤치마크는 텍스트 기반 모델과 비교하여 도구 이용의 효과를 명확하게 이해하려고 합니다.

- **Performance Highlights**: 연구 결과, 다양한 모델에서 직접적인 분석 해결이 도구 사용을 통한 시각 해결보다 뛰어난 성능을 보이는 것으로 나타났습니다. 이는 많은 문제에서 플롯 생성이 유용한 전략임에도 불구하고, 모델이 도구를 통해 얻는 시각적 증거를 통합하는 데 어려움이 있다는 것을 시사합니다. VAMPS는 현재 모델들에서 이러한 reasoning-to-perception (추론-지각) 전환이 여전히 병목 현상으로 작용하는지를 테스트하기 위해 설계되었습니다.



### DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities (https://arxiv.org/abs/2606.04205)
- **What's New**: DetectZoo는 AI 생성 콘텐츠 탐지를 위한 최초의 다기능 툴킷으로, 텍스트, 이미지 및 오디오 모달리티에 대한 통합 인터페이스를 제공합니다. 이 툴킷은 데이터 수집 및 전처리부터 모델 평가에 이르는 전체 실증적 파이프라인을 표준화하여 연구자들이 최신 탐지기를 체계적으로 벤치마킹할 수 있게 합니다. 61개의 탐지기를 위한 참조 구현, 22개의 벤치마크 데이터셋을 위한 네이티브 로더 및 공통 인터페이스를 통해 여러 메트릭을 보고하는 표준화된 평가 파이프라인도 포함되어 있습니다.

- **Technical Details**: DetectZoo는 모달리티 간의 비교를 용이하게 하기 위해 61개의 탐지 방법을 단일 코드베이스에 통합하고 22개의 벤치마크 데이터셋 및 표준화된 평가 파이프라인과 함께 제공됩니다. 통합된 모달리티의 접근 방식을 통해 연구자들은 반복 가능한 비교를 실시할 수 있으며, 이는 AI 생성 콘텐츠 탐지의 효과적인 연구 환경을 조성합니다. 각 탐지기는 독립적이며 동일한 인터페이스를 통해 접근 가능하며, 사전 학습된 가중치를 자동으로 캐시하여 원래 발표된 결과를 재현합니다.

- **Performance Highlights**: DetectZoo는 다중 모달 AI 포렌식의 진입 장벽을 낮춰 연구자들이 도메인 간 성능 차이를 식별하고 강력하고 일반화 가능한 탐지 기법의 개발을 가속화할 수 있도록 합니다. 이는 AI 생성 콘텐츠 탐지의 연구 성과를 더욱 효과적으로 발전시키고, 개별 논문 코드베이스에 대한 의존도를 줄이는데 기여합니다. 모든 구성 요소는 공개적으로 사용 가능하며, pip를 통해 쉽게 설치할 수 있어 연구자들이 접근할 수 있는 유용한 리소스가 됩니다.



### SymTRELLIS: Symmetry-Enforced Voxel Latents for 3D Generation (https://arxiv.org/abs/2606.04108)
- **What's New**: SymTRELLIS는 단일 보기 3D 생성 모델의 대칭성을 강화하는 새로운 방법입니다. 이 방법은 VAE(Variational Autoencoder) 또는 flow 모델을 재훈련하지 않고도 TRELLIS.2 생성 중에 대칭성을 강화할 수 있도록 설계되었습니다. 핵심 아이디어는 3D 생성 과정에서 대칭성을 강제하고, 최종 결과물의 기능적 요구 사항을 충족시키는 것입니다.

- **Technical Details**: SymTRELLIS는 공간 변환의 잠재적 작용을 선형 연산자로 모델링하고, 가벼운 spatial-transform latent mapper를 사용하여 비대칭 3D 데이터에 대해 학습합니다. 생성 과정에서 velocity symmetrization을 통해 예측된 흐름 속도(velocity)를 대칭적으로 평균화하며, 이는 ODE(Ordinary Differential Equation) 단계에서 적용됩니다. 이 방식은 대칭 구조의 명확한 추정을 가능하게 하여 더 복잡한 대칭 구조도 처리할 수 있도록 합니다.

- **Performance Highlights**: 심층적인 Benchmark를 통해 SymTRELLIS는 266개의 엄격히 대칭적인 객체에 대해 TRELLIS.2, Hunyuan3D-2.1, TripoSG와 비교하여 모두 대칭 오차 지표를 현저히 줄이며, 기본 모델과 비슷한 재구성 정확도를 유지합니다. 3D 출력물을 테스트한 결과, 대칭성이 올바르게 적용된 모델은 안정적으로 회전하며 성능이 개선되었습니다.



### TGSD: Topology-Guided State-Space Diffusion for EEG Spatial Super-Resolution (https://arxiv.org/abs/2606.03998)
- **What's New**: 본 논문에서는 EEG의 공간 초해상도를 위한 새로운 프레임워크인 TGSD(Topology-Guided State-Space Diffusion)를 제안합니다. TGSD는 전체 전극 레이아웃에 대한 풍부한 공간 정보를 수집하고, 점진적으로 필수적인 전극 신호를 생성하기 위한 조건부 상태 공간 확산 재구성기를 포함하고 있습니다. 이를 통해 낮은 밀도의 EEG 신호로부터 고밀도의 EEG 신호를 효과적으로 복원할 수 있습니다.

- **Technical Details**: TGSD는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 계층적 공간 선행 인코더(Hierarchical Spatial Prior Encoder)는 지역 기하 관계와 지역 수준 맥락 정보를 통합하여 전체 전극 레이아웃에 대해 토폴로지 인식 선행 정보를 학습합니다. 둘째, 조건부 상태 공간 확산 재구성기(Conditional State-Space Diffusion Reconstructor)는 학습된 선행 정보를 바탕으로 결측된 채널의 EEG 신호를 점진적으로 생성합니다.

- **Performance Highlights**: SEED 및 PhysioNet MM/I 데이터 세트에서 TGSD는 서로 다른 초해상도 계수에서 주요 기준선 모델에 비해 일관되게 우수한 성능을 보였습니다. 재구성의 충실도와 하위 분류(classification) 성능 모두에서 TGSD의 효과성이 확인되었습니다. 이러한 결과는 저밀도 EEG 감지 및 IoT 기반 신경 감지 응용 프로그램에서 TGSD의 실용성을 강조합니다.



### CR-Seg: Attention-Guided and CoT-Enhanced Coarse-to-Refined Reasoning Segmentation (https://arxiv.org/abs/2606.03564)
- **What's New**: 이번 논문에서는 복잡한 언어로 설명된 대상 객체를 분할하는 Reasoning Segmentation 문제를 다루고 있습니다. 기존의 방법들은 멀티모달 대형 언어 모델(MLLM)과 분할 모델 간의 정합성 문제를 겪고 있었으며, 이 문제를 해결하기 위해 Attention-Guided 및 CoT-Enhanced Coarse-to-Refined Reasoning Segmentation(이하 CR-Seg)이라는 새로운 두 단계 프레임워크를 제안합니다. 이 프레임워크는 MLLM의 attention map을 기반으로 하여 초기 분할을 개선합니다.

- **Technical Details**: CR-Seg는 Extract Attention Maps and Points (EAP) 모듈을 통해 coarse target localization을 위한 attention map을 추출하고, 이를 SAM에 통합하여 마스크를 개선합니다. 또한, Global-to-Local Chain-of-Thought (GLCoT) 접근 방식을 도입하여 모델이 전역 장면 맥락에서 지역 타겟 세부 사항으로 점진적으로 추론하도록 유도합니다. 이러한 방법을 통해 내재된 응답 의미를 유지하면서도 정합성 문제를 완화할 수 있습니다.

- **Performance Highlights**: 영향력 있는 실험을 통해 CR-Seg는 기존 방법들에 비해 높은 성능을 나타냈으며, Dummy에서 동작한 여러 복잡한 테스트에 대해서도 효과적인 결과를 보고했습니다. 이를 통해 CR-Seg의 강력한 타겟 분별력을 입증하였으며, 특히 동일 카테고리 객체 간의 미세한 차별화에서 두각을 나타냈습니다.



### P$^2$-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization (https://arxiv.org/abs/2606.03376)
- **What's New**: 본 연구는 Perceptual Processing Direct Preference Optimization (P$^2$-DPO)이라는 새로운 훈련 패러다임을 제안합니다. 이 방법은 모델이 자신의 선호 쌍을 생성하고 학습함으로써 인식 병목 현상을 직접 해결합니다. 기존의 비전 무관 선호 쌍의 한계를 극복하며, 데이터의 비효율성을 줄이기 위한 방법론이 추가되었습니다.

- **Technical Details**: P$^2$-DPO는 두 가지 새로운 유형의 선호 쌍을 도입합니다: (1) Focus-and-Enhance Preference Pair는 인식 병목 현상을 해결하기 위해 이미지의 세밀한 디테일에서 개선된 출력과 저하된 출력을 대조하며, (2) Visual Robustness Preference Pair는 깨끗한 신호와 노이즈가 있는 신호의 출력을 대조하여 시각적 견고성을 강화합니다. 이 과정에는 동적 가중치 조정 및 보정 손실(Calibration Loss)도 포함되어 효율적인 학습 루프를 생성합니다.

- **Performance Highlights**: 실험 결과는 P$^2$-DPO가 강력한 기준 모델들을 초과하며, 비용이 많이 드는 인간 피드백 없이도 비교 가능한 훈련 데이터 및 비용으로 성과를 달성함을 보여줍니다. 또한, Attention Region Fidelity (ARF) 및 이미지 저하 시나리오에 대한 평가를 통해 P$^2$-DPO가 인식 병목 현상을 해결하고 시각적 견고성을 향상시키는 데 효과적임을 입증했습니다.



### MedSyn2: Flexible Control of 3D CT Generation via Text and Semantically-Defined Segmentation Prompts (https://arxiv.org/abs/2606.00967)
- **What's New**: 본 논문은 의료 이미징에서의 볼륨 부피 생성 모델의 제어 가능성을 높이는 유연한 다중 모드 프레임워크를 제안합니다. 기존의 방법들은 방사선 보고서 또는 전체 이미지 세분화에 의존하여 생성 과정을 제어하였으나, 그 한계점을 극복하고 사용자가 특정 해부학이나 이상 징후에 대한 세분화를 제공할 수 있게 합니다. 이 프레임워크는 세분화 마스크의 의미를 설명하는 텍스트 설명을 통해 높은 유연성과 확장성을 갖추고 있습니다.

- **Technical Details**: 우리는 수정된 확산 변환기(diffusion transformer)를 기반으로 하는 메모리 효율적인 아키텍처를 개발했습니다. 이 모델은 이미지와 세분화 토큰을 동시에 처리하며, 긴 방사선 보고서에 효과적으로 대응할 수 있도록 게이트 주의를 통합하였습니다. 세분화 프롬프트의 의미는 방사선 보고서에 추가된 선택적 텍스트로 제공되어, 새로운 이상 징후를 쉽게 도입할 수 있는 확장 가능한 조건화 메커니즘을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 상태-of-the-art 인지(perceptual) 및 의미적(semantic) 점수를 달성하며, 해부학적으로 일관된 고해상도 CT 볼륨을 생성합니다. 두 명의 방사선 의사 평가 결과, 생성된 이미지가 실제 의료 이미지와 강한 일치를 보임을 확인하였으며, 데이터 증강(data augmentation) 시 데이터 효율성이 개선되었음을 보여줍니다.



New uploads on arXiv(cs.AI)

### Knowledge Index of Noah's Ark (https://arxiv.org/abs/2606.05104)
- **What's New**: 본 연구에서는 LLMs(대형 언어 모델)를 위한 새로운 지식 벤치마크인 KINA를 소개하며, 261개의 세분화된 분야에 걸쳐 899개의 항목으로 구성되어 있습니다. KINA는 대표성 문제를 해결하기 위해 전문가의 의견을 기반으로 한 기준을 사용하여 선정의 정확성을 높였습니다. 또한, 보너스 기반의 심사 메커니즘을 도입하여 리뷰의 품질을 혁신적으로 개선합니다.

- **Technical Details**: KINA에서는 두 가지 공식적인 결과를 제시하는데, 첫 번째는 대표성을 예산 지원 중심의 접근 방식으로 설정하여 각 후보 항목을 전문가의 기준에 따라 평가합니다. 두 번째는 flat-payment 대신 보너스 기반의 토너먼트를 활용하여 리뷰의 품질을 향상시키고, 이는 FOSD(First-order Stochastic Dominance) 조건 하에 기존의 기준보다 우수한 성능을 보입니다.

- **Performance Highlights**: KINA를 통해 42개의 모델을 평가한 결과, Gemini-3.1-Pro-Preview 모델이 53.17%의 정확도로 가장 높은 성적을 기록했고, 뒤를 이어 Claude-Opus-4.6와 GPT-5.4가 각각 49.92%와 48.55%를 기록했습니다. 이는 여전히 많은 성과 개선의 여지가 있음을 나타내며, 웹 검색 도구의 추가 사용이 모델 성능에 다양하게 기여하고 있음을 보여주었습니다.



### AutoLab: Can Frontier Models Solve Long-Horizon Auto Research and Engineering Tasks? (https://arxiv.org/abs/2606.05080)
Comments:
          Code: this https URL ; Website: this https URL

- **What's New**: 이 논문에서는 반복적인 개선의 중요성을 강조하며, 단기적인 성과 평가가 아닌 장기적인 최적화를 목표로 하는 새로운 벤치마크인 'AutoLab'을 소개합니다. AutoLab은 시스템 최적화, 퍼즐 및 도전, 모델 개발, CUDA 커널 최적화 등 네 가지 분야에 걸쳐 36개의 전문가가 선별한 작업으로 구성되어 있습니다. 각 작업은 처음에는 올바르지만 의도적으로 비효율적인 기준을 설정하고, 에이전트가 엄격한 시간 제약 내에서 이를 개선하도록 만듭니다.

- **Technical Details**: AutoLab의 설계는 세 가지 주요 원칙에 따라 구성됩니다: 첫째, 작업은 지속적인 경험적 반복을 요구해야 하며, 둘째, 점수는 연속적이고 잘 조정되어야 하며, 셋째, 검증 과정은 해킹 저항이 있어야 합니다. 각 작업은 자연어로 된 설명, 컨테이너화된 샌드박스 환경, 검증기, 참조 솔루션 및 실행 가능 시간 한계로 구성됩니다. 이러한 구성 요소는 복잡한 작업을 정의하고 평가하는 데 필요한 다양한 요소들을 포함하고 있습니다.

- **Performance Highlights**: 평가에서 드러난 바와 같이, 에이전트의 초기 시도의 품질보다 지속적으로 벤치마킹하고 수정하며 경험적 피드백을 통합하는 것이 성공의 주요 결정 요소임을 확인했습니다. 'claude-opus-4.6'은 장기 최적화 능력에서 두각을 나타내는 반면, 일부 모델은 조기 종료하거나 예산을 소진하고 최종 솔루션을 생성하지 못했습니다. 이러한 결과는 자율 에이전트의 미래 연구에서 시간 인식과 지속적인 반복이 필수적임을 보여줍니다.



### Strabo: Declarative Specification and Implementation of Agentic Interaction Protocols (https://arxiv.org/abs/2606.05043)
Comments:
          Presented in the Engineering Multiagent Systems Workshop co-located with the 2026 International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)

- **What's New**: 최근 몇 년 동안 선언적 상호작용 프로토콜(Interaction Protocols)을 기반으로 한 멀티 에이전트 시스템의 모델링과 구현에서 중요한 발전이 이루어졌습니다. 본 논문에서는 Strabo라는 방법론을 제안하여 UCP(Universal Commerce Protocol)의 체크아웃 부분을 모델링하고 Peach라는 프로그래밍 모델을 사용해 에이전트를 구현합니다. 이 연구의 주요 목표는 형식적이고 선언적인 접근 방식이 실제 산업 노력에 어떻게 적용될 수 있는지를 입증하는 것입니다.

- **Technical Details**: UCP는 전자 상거래 상호작용을 다룰 때, 고객(Platform), 상인(Business), 디지털 지갑(Credential Provider) 및 결제 제공자(Payment Provider)의 역할을 포함합니다. Langshaw는 이러한 상호작용을 명세하기 위한 선언적 프로토콜 언어로, 에이전트가 수행하는 역할 및 상호작용의 완전한 수행 기준을 정의합니다. Peach는 이러한 Langshaw 프로토콜을 기반으로 에이전트를 구축하기 위한 프로그래밍 모델로, 개발자가 직접 프로토콜 제약 조건을 구현하는 대신 툴을 통해 상호작용을 관리합니다.

- **Performance Highlights**: Strabo의 구현을 통해 Peach 에이전트가 Google에서 구현한 UCP 에이전트와의 상호운용성을 demonstrated합니다. 이는 형식적인 선언적 프로토콜이 산업 요구에 직접적으로 적용될 수 있음을 입증하는 사례로, UCP가 비공식적으로 명세되어 있어 중요한 부분이 모호한 반면, Langshaw 모델은 보다 정밀하게 상호작용 제약 조건을 명시하여 실질적인 상호 운용성을 달성했습니다.



### What Type of Inference is Active Inference? (https://arxiv.org/abs/2606.04935)
- **What's New**: 이 논문은 Active Inference를 사용하여 의사 결정 과정을 추론(inference)으로 설정하고, Expected Free Energy (EFE) 최소화가 epistemic priors로 보강된 생성 모델에서 Variational Free Energy (VFE) 최소화로 표현될 수 있음을 보여줍니다. 기존의 EFE 개념을 명료하게 하기 위해 EFE 기반 계획 수립시 필요한 수정 사항들을 분리하고, 이를 결합한 메시지 전달(message-passing) 방식을 제시합니다. 또한, 세 가지 그리드 월드 환경에서 실험을 진행하여 관측이 결정적일 때 계획 수정이 어떻게 도움을 주는지를 증명합니다.

- **Technical Details**: 계획 수립은 에이전트가 미래 관측(observations)과 상태(state), 그리고 행동(actions)의 결과를 예측하는 생성 모델을 유지하며 이루어집니다. 이 모델은 preferenced priors를 보강하여 목표를 정의하고, Variational Free Energy (VFE)를 최소화하여 미래 궤적에 대한 신념(beliefs)을 얻습니다. 논문에서는 VFE 감소가 어떻게 각 변수의 역할에 따라 대칭적으로 이루어지는지를 강조하며, 이를 통해 설계된 생성 모델이 효율적으로 기능하도록 만드는 여러 엔트로피 수정(entropy corrections)을 도입합니다.

- **Performance Highlights**: 실험 결과, 계획 수정이 중요할 때, 즉 관측이 결정적(decisive)일 때 이미 유용성을 발휘한다는 것을 확인하였습니다. 반면, 관측이 제안적(suggestive)일 때는 추가적인 엔트로피 수정이 더욱 큰 차이를 만드는데 기여합니다. 이러한 연구 결과는 Active Inference의 적용 가능성을 확장하고, 다양한 환경에 대한 계획 수립 시 고려해야 할 수정 항목들을 명확히 합니다.



### AICompanionBench: Benchmarking LLMs-as-Judges for AI Companion Safety (https://arxiv.org/abs/2606.04867)
- **What's New**: 이 연구는 AI 동반자 대화에서 안전성을 평가하기 위한 최초의 공개 벤치마크 데이터셋인 AICompanionBench를 소개합니다. 이 데이터셋은 2,123개의 실제 Replika 대화를 포함하고 있으며, 아홉 가지 안전 위험 범주로 주석이 달려 있습니다. 최근 AI 동반자와의 상호작용의 안전성에 대한 우려가 커짐에 따라, 이 데이터셋은 해당 문제를 연구하기 위한 중요한 자원으로 제공됩니다.

- **Technical Details**: AICompanionBench 데이터셋은 비정상적 동작을 식별하는 자동 감지 방법을 개발하여 안전한 상호작용을 정의합니다. 두 가지 접근 방법 중 전통적인 기계 학습과 LLM을 이용한 접근을 비교하고, 다양한 최첨단 LLM을 벤치마킹하여 모델 성능의 차이를 보여줍니다. 연구 결과, LLM은 명시적인 유해 콘텐츠 감지는 효과적이지만, 암시적 안전 상호작용의 식별 능력은 제한적임을 발견했습니다.

- **Performance Highlights**: 실험 결과, GPT 계열 모델이 다른 모델들에 비해 일반적으로 더 높은 성능을 보여주지만, 세부적인 위험 유형인 조작에는 여전히 한계를 보입니다. 각 모델은 '무해' 대화를 잘못 식별하는 경향도 있어, 보다 섬세한 분류 기준이 필요함을 시사합니다. 이 연구는 AI 동반자 시스템의 모니터링을 위한 더 나은 방법론을 개발할 수 있는 기초 자료를 제공합니다.



### R-APS: Compositional Reasoning and In-Context Meta-Learning for Constrained Design via Reflective Adversarial Pareto Search (https://arxiv.org/abs/2606.04823)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 개방형 작업에서 유창함을 보이지만, 계획하고 도구를 사용하며 장기적으로 행동해야 하는 에이전트 설정에서는 신뢰성 있는 결과를 제공하지 못하는 문제를 다루고 있습니다. 저자들은 이러한 문제를 해결하기 위한 새로운 방법인 Reflective Adversarial Pareto Search (R-APS)를 소개하며, 이는 세 가지 주요 실패를 동시에 해결하는 최초의 방법이라고 주장합니다.

- **Technical Details**: R-APS는 각각의 추론 모드에 고유한 맥락(context)을 할당하고, 세 가지 시간 척도(timescales)에서의 상호작용을 조정하는 방식으로 작동합니다. 이 방법은 계획적 조합 추론(staged compositional reasoning), 민감도 기반의 반사실적 스트레스 테스트(sensitivity-guided counterfactual stress-testing), 메타 귀납적 규칙 추출(meta-inductive rule extraction)을 통해 고유의 문맥을 개발합니다. 이는 세 개의 구조적 실패를 해결하며, 추가적인 파인 튜닝 없이 고정된 LLM에서 작동합니다.

- **Performance Highlights**: R-APS는 로봇공학, 보철학, 기계 설계 분야에서 평면 기계 합성(planar mechanism synthesis) 문제를 평가했으며, 32개의 목표 경로(target trajectories)에 대해 uniform-perturbation 기준보다 3.5배 더 긴밀한 견고성 인증을 제공했습니다. 또한, 첫 번째 수용(iteration-to-first-admission)까지 46% 더 빠른 속도를 기록하고, Enum+GA에 비해 2.1배 더 낮은 Chamfer 거리(chamfer-distance)를 달성했습니다. 이 연구는 작은 4B 사고 전문화 모델이 70B 범용 모델과 경쟁력을 가질 수 있음을 보여줍니다.



### Beyond Objective Equivalence: Constraint Injection for LLM-Based Optimization Modeling on Vehicle Routing Problems (https://arxiv.org/abs/2606.04816)
Comments:
          28 pages

- **What's New**: 이 논문에서는 큰 언어 모델(LLM)이 자연어 최적화 문제를 실행 가능한 솔버 코드로 변환하는 과정에서 발생할 수 있는 한계를 극복하기 위해 'constraint injection'이라는 검증 연산자를 제안합니다. 이를 통해 비현실적인 제약 조건을 감지하고, 소프트웨어가 필요한 제약 조건을 생략하지 않았는지 확인할 수 있습니다. 또한, 논문에서는 차량 경로 최적화 문제(VRP)를 대표적인 제약이 밀집된 조합 최적화 테스트베드로 사용하여 이 새로운 접근 방법을 평가하였습니다.

- **Technical Details**: 제안된 'constraint injection'은 적합한 프로브를 사용하여 스퍼리우스 오버 제약(spurious over-constraint)과 침묵 제약 누락(silent constraint omission)을 감지하는 메커니즘입니다. VRPCoder라고 불리는 8B의 엔드투엔드 모델은 자연어 VRP 시나리오를 Gurobi 스크립트로 변환하며, 전문가가 검증한 VRP 벤치마크 자료집과 함께 제공합니다. 이 검증기는 데이터 합성을 위한 거부 샘플링 필터와 그룹 상대 정책 최적화(GRPO)에서 롤아웃 보상으로 재사용됩니다.

- **Performance Highlights**: VRPCoder-GRPO는 네 개의 VRP 벤치마크를 통해 평균 93% Pass@1을 달성하였고, 세 개의 벤치마크에서 Gemini-3.1-Pro Preview를 초과하였으며, Claude-Sonnet-4.5보다 평균 28포인트 더 높은 성능을 보였습니다. 이전의 OR-LLMs보다 평균 78포인트 더 나은 성능을 기록했습니다. 이러한 결과는 제안된 새로운 검증 방식이 실제 제약 조건의 충실한 구현을 보장할 수 있음을 잘 보여줍니다.



### BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization (https://arxiv.org/abs/2606.04807)
Comments:
          Accepted to Findings of the ACL

- **What's New**: BiasGRPO라는 새로운 프레임워크를 통해, 사회적 편견을 완화하기 위한 기존 방법론의 한계를 극복합니다. 이 방법은 Group Relative Policy Optimization (GRPO)를 사용하여 보상을 정규화함으로써 훈련 불안정을 줄이고, 다양한 분야에서의 데이터셋을 합성적으로 확장하여 더 넓은 맥락에서 적용 가능성을 높입니다. 또한, 고유한 편견 보상 모델을 통해 계산 효율성을 극대화하고 있습니다.

- **Technical Details**: BiasGRPO는 세 가지 구성 요소로 이루어진 파이프라인입니다: 합성적으로 확장된 데이터셋, 커스텀 편견 보상 모델, 기초 GRPO 알고리즘입니다. GRPO의 기본 알고리즘은 기존 DPO와 PPO 방법들보다 사회적 편견 완화의 고변동성 환경에 더 적합하게 작용함을 보여 줍니다. 이는 각 완료 그룹의 평균 보상을 기반으로 모델 업데이트를 유도하여 더 안정적인 업데이트를 제공합니다.

- **Performance Highlights**: BiasGRPO는 여러 벤치마크에서 DPO와 PPO를 능가하는 성능을 발휘했습니다. Hugging Face에 다양하고 포괄적인 데이터셋과 커스텀 편견 보상 모델을 배포하여, 복잡한 다목적 RLHF 파이프라인에 쉽게 통합할 수 있는 소중한 자료를 제공합니다. 이를 통해 더 많은 연구자들이 효과적인 편견 완화 기법을 구현할 수 있는 기회를 확대합니다.



### AIP: A Graph Representation for Learning and Governing Agent Skills (https://arxiv.org/abs/2606.04781)
- **What's New**: 이 논문에서는 Agent Instruction Protocol (AIP)을 제안하고 있습니다. AIP는 기술을 구조화된 실행 그래프(directed execution graph)로 모델링 하여 에이전트가 작업을 더 신뢰성 있게 수행할 수 있도록 합니다. 기존의 자유 형식의 기술 문서 대신에, 각 기술은 명확한 입력/출력(edge) 접속을 가진 노드(nodes)로 이루어진 그래프로 표현되며, 이를 통해 복잡한 고정 절차적 지식(procedural knowledge)도 효과적으로 전달될 수 있습니다.

- **Technical Details**: AIP는 스키마 검증된 YAML 사양(schema-validated YAML specification)으로 구성되며, 각 노드는 결정론적인 스크립트(deterministic scripts) 또는 자연어 설명(natural-language description)으로 뒷받침됩니다. 이 방식은 에이전트가 기술을 작성하는 과정에서 발생할 수 있는 오류를 사전에 감지하고 수정할 수 있는 기능을 제공합니다. 또한, 두 개의 실패 모드가 노드 및 스크립트 수준에서 진단되고 수정된 사례를 통해 에이전트의 자기 개선 메커니즘도 마련되었습니다.

- **Performance Highlights**: 논문에서는 AIP로 변환된 기술이 27개의 실제 에이전트 작업에서 평균 작업 보상(task reward)을 0.60에서 0.71로 증가시켰다고 보고합니다. 이 연구에서 AIP는 에이전트의 작업 성공률을 53%에서 67%로 높이는 결과를 보였으며, 이는 통계적으로 유의미한 향상으로 인정받았습니다. 이러한 성과는 AIP 구조가 강화 학습(reinforcement learning) 및 자기 개선(agent-assisted improvement)에 있어서 유의미한 기반을 제공함을 시사합니다.



### Tree-Based Formalization of Multi-Agent Complementarity in Human-AI Interactions (https://arxiv.org/abs/2606.04779)
Comments:
          29 pages, 9 figures

- **What's New**: 이 논문은 여러 에이전트 간의 상호작용을 위한 보완성(complementarity)에 대한 새로운 프레임워크를 제시합니다. 기존 연구는 주로 두 개의 에이전트로 제한되어 있었지만, 이 연구는 트리 기반(tree-based) 형식을 통해 다중 에이전트 인간-AI 상호작용의 복잡한 작업 흐름을 모델링합니다. 이 접근법을 통해 예측 벡터가 어떻게 결합되며 최종 출력이 형성되는지를 명확하게 정의합니다.

- **Technical Details**: 본 연구에서는 에이전트의 역할 구성과 루트가 있는 계획 이진 트리를 통해 HAI 프로토콜을 정의합니다. 각 내부 노드는 이진 결합 규칙을 통해 예측 벡터를 조합하며, 이에 따라 트리 상대적(complementarity functional) 기능이 평가됩니다. 또한 여러 에이전트가 동일한 레이블 데이터 집합에서 예측을 수행하는 상황을 연구하여 보완성이 다양한 조건에서 어떻게 달성되는지를 분석합니다.

- **Performance Highlights**: 여러 가지 결과를 통해 보완성은 다중 에이전트 회귀에서는 달성 가능하지만, 이진 분류(binary classification)에서는 자연적인 조건 하에서 방해받는다는 것을 보여줍니다. 특히, 특정 손실 함수 하에서는 보완성을 달성할 수 없음을 증명했습니다. 이 연구는 HAI_settings와 같은 높은 이해관계의 경우, 보완성을 경험적으로 조사할 때 새로운 관점을 제공할 수 있습니다.



### Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along Generation Trajectories (https://arxiv.org/abs/2606.04778)
- **What's New**: 최근의 연구에서, 안전 정렬된 대형 언어 모델(LLMs)이 단기적인 토큰 주입을 통해 생성 과정에서 해로운 출력을 유도할 수 있는 취약성을 가진다는 것을 보여주었습니다. 이는 새로운 분석으로, 기존의 얕은 안전성(Shallow safety) 개념이 더 넓은 추론 시간 취약성(Inference-time vulnerability)의 특별한 경우임을 입증합니다. 이는 토큰이 주입될 경우, 생성 과정이 해로운 방향으로 크게 변화할 수 있음을 나타냅니다.

- **Technical Details**: 연구진은 구조적으로 안전성을 높이고 해로운 방향으로의 토큰 주입에 대한 견고성을 강화하기 위해, 생성 과정 자체에 대한 안전 정렬 방법을 제안합니다. 이 방법은 중간 디코딩 단계에서 토큰 주입을 시뮬레이션하여 모델의 생성 궤적을 확장하는 것을 포함합니다. 이를 통해, 안전한 생성이 해로운 주입에 의해 방해받는 경우와, 해로운 생성이 거부로 다시 유도되는 경우를 모두 포함하는 훈련 데이터를 생성합니다.

- **Performance Highlights**: 실험적으로 이 방법은 추론 시 주입에 대한 견고성을 개선하였으며, 데이터셋을 넘어 일반화되었습니다. 반복 적용 시, 공격 성공률(ASR)을 거의 0에 가깝게 줄이는 효과를 보였으며, 기존의 공격 기법에 대한 견고성을 강화하는 결과를 나타냈습니다. 따라서, 안전한 생성 정렬을 위한 훈련 접근 방식이 모델의 출력 형성 과정의 동역학을 고려해야 함을 강조합니다.



### FALSIFYBENCH: Evaluating Inductive Reasoning in LLMs with Rule Discovery Games (https://arxiv.org/abs/2606.04751)
- **What's New**: 이번 연구에서는 과학적 발견과 관련된 귀납적 추론(inductive reasoning)을 평가하기 위한 FALSIFYBENCH라는 새로운 평가 프레임워크를 소개합니다. 이 프레임워크는 Wason의 2-4-6 작업을 기반으로 하며, LLM들이 히든 규칙(hidden rules)을 발견하기 위해 반복적으로 예시를 제안하고 피드백을 받는 방식으로 구성되어 있습니다. 연구 결과, 부정적 테스트(negative testing)를 적극적으로 추구하는 모델이 더 나은 성능을 보임을 확인했습니다.

- **Technical Details**: FALSIFYBENCH는 가설 생성(hypothesis generation), 증거 수집(evidence gathering), 및 증거에 대한 신념 수정(belief revision)의 핵심 요소들을 포함합니다. 12개의 LLM을 다양한 모델 가족 및 스케일에서 평가한 결과, 추론 모델이 일반적으로 지침 조정(instruction-tuned) 모델들보다 과학적 추론에서 더 나은 성과를 나타냈습니다. 정확도를 높이기 위해 모델들이 어떻게 가설 공간(hypothesis space)을 탐색하는지에 대한 미세한 분석도 이루어졌습니다.

- **Performance Highlights**: 연구에서 LLM들은 주어진 규칙에 부합하는 세 가지 항목을 제안하고, 그에 따라 피드백을 받으며 자신의 가설을 검증합니다. 하지만 모든 모델이 최적 성능에 가까운 것은 아니며, 부정적 테스트의 중요성이 두드러지면서 모델들 간의 성능 차이가 발생했습니다. 연구는 과학적 사고를 위한 비판적 요소로서 부정적 가설 검증의 필요성을 강조합니다.



### Fog of Love: Engineering Virtuous Agent Behavior with Affinity-based Reinforcement Learning in a Game Environmen (https://arxiv.org/abs/2606.04750)
- **What's New**: 이 논문은 인공지능이 덕목 있는 행동을 학습할 수 있도록 하는 새로운 방법인 유사성 기반 강화 학습(affinity-based reinforcement learning)을 소개합니다. 본 연구는 보드 게임 'Fog of Love'를 통해 두 명의 에이전트가 각자의 덕목을 달성하는 동시에 관계를 만족시키기 위한 협력을 수행해야 하는 복잡한 환경에서 이 기술의 효과를 입증합니다. 이 방식은 에이전트의 덕목을 통해 경쟁 및 협력 목표를 달성하는 데 기여하며, 인간 수준의 해석 가능성을 제공합니다.

- **Technical Details**: 이 연구에서는 덕목 기반의 에이전트를 위한 환경으로 'Fog of Love' 게임 메커니즘을 활용합니다. 여기서는 두 에이전트가 관계 안에서 개인의 목표(경쟁적)와 파트너의 필요(협력적)를 동시에 충족하기 위해 결정합니다. 또한, 이 논문은 다중 에이전트 심층 결정 정책 기울기(multi-agent deep deterministic policy gradient) 알고리즘을 사용하여 미리 정의된 행동 확률(제약조건)을 통해 목표 함수를 정규화하는 유사성 기반 강화 학습 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과, 지역화된 유사성 기반 강화 학습 알고리즘이 기준 알고리즘에 비해 성과가 현저히 향상되었음을 보여줍니다. 이는 사용자 정의 보상 공학 없이 덕목 있는 행동을 촉진하는 것이 가능함을 나타냅니다. 경쟁 및 협력 목표 모두에서 향상된 총 점수를 통해 에이전트의 성능이 크게 개선되었으며, 이는 윤리적 행동을 강조하는 AI 개발에 기여할 것으로 기대됩니다.



### BiNSGPS: Geometry Problem Solving via Bidirectional Neuro-Symbolic Interaction (https://arxiv.org/abs/2606.04648)
- **What's New**: 이 논문에서는 Geometry Problem Solving (GPS)와 관련된 인공지능의 기존 접근 방식의 한계를 극복하기 위해 BiNSGPS라는 새로운 프레임워크를 제안합니다. 기존 네오-심볼릭(neuro-symbolic) 접근 방식이 단방향 파이프라인에 의존하는 반면, BiNSGPS는 다이렉트 인터랙션을 통해 MLLM Adviser와 Symbolic Solver 간의 양방향 신호 흐름을 가능하게 합니다. 이를 통해 시스템은 초기 단계의 오류를 피하고, 논리적인 일관성을 유지하면서 복잡한 추론을 수행할 수 있게 됩니다.

- **Technical Details**: BiNSGPS Framework는 MLLM Adviser와 Symbolic Solver 간의 쌍방향 상호작용(Bidirectional Neuro-Symbolic Interaction, BiNS)을 통해 작동합니다. 이 구조에서는 Symbolic Solver가 하이퍼그래프 확장을 통해 철저한 추론을 수행하고, 논리적 불일치나 추론 정체를 식별하니다. 비선형 대화의 구성 요소로서의 MLLM Adviser는 융통성 있는 추론 및 제안된 가설을 통해 심볼릭 솔버의 한계를 극복합니다.

- **Performance Highlights**: BiNSGPS의 성능 평가 결과, Geometry3K 및 PGPS9K 데이터셋에서 각각 90.5% 및 90.1%의 새로운 SOTA 성능을 달성했습니다. 또한, 이 프레임워크는 단계적 추론에서의 논리적 일관성을 보장하며, 수학적 엄밀성과 추론의 유연성을 균형 있게 유지하는 것으로 입증되었습니다.



### MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models (https://arxiv.org/abs/2606.04627)
- **What's New**: MIRAGE는 모바일 에이전트가 비디오 인터페이스에서 작업을 수행하기 위해 내부적으로 추론을 학습하도록 설계된 새로운 프레임워크입니다. 이 시스템은 긴 텍스트의 추론 체인 대신 간결한 숨겨진 상태를 사용하는 방법으로 모바일 운영의 효율성을 크게 향상시킵니다. 이를 통해 실행 효율을 높이면서도 기존의 추론 능력을 유지할 수 있습니다.

- **Technical Details**: MIRAGE는 비가시적인 추론을 통해 액션을 수행하고, 근본적으로 두 단계의 훈련 절차를 통해 내부 표현을 학습합니다. 첫 번째 단계에서, 명시적인 텍스트 추론을 기반으로 모바일 조작을 학습하고, 두 번째 단계에서 지속적인 잠재적 추론 슬롯으로 교체하여 모델이 잠재 공간에서의 추론을 내재화하도록 합니다. 이 과정에서 Approximate Parallel Latent Refinement(APLR)과 Q-Former 세계 모델 헤드를 사용하여 미래의 GUI 상태를 예측할 수 있는 능력을 결합합니다.

- **Performance Highlights**: MIRAGE는 AndroidWorld와 AndroidControl 의 심각한 벤치마크에서 성능을 극대화합니다. AndroidWorld에서는 기존의 Instruction-tuned baseline보다 10.2% 높은 작업 성공률을 달성하며, 3-5배 낮은 토큰 비용으로 명시적 체인 오브 사고(Cot)를 충족합니다. AndroidControl에서는 생성된 토큰 수를 75% 이상 줄이면서도 액션 정박의 정확성을 높이는 등의 성과를 보여줍니다.



### A Normative Intermediate Representation for ASP-Based Compliance Reasoning (https://arxiv.org/abs/2606.04619)
- **What's New**: 이번 논문은 MONIR(Modalized-Output Normative Intermediate Representation)을 제안합니다. MONIR은 ASP 기반의 규정 준수 추론(compliance reasoning)을 위한 조정된 출력을 갖는 중간 표현을 제공합니다. 그 핵심 구성 요소는 단계적 운영 의미론을 갖추고 있으며, MONIR-ASP는 외부 함수, 시간 규칙 및 안정 모델 추론을 위한 실행 가능한 컴파일과 확장을 제공합니다.

- **Technical Details**: MONIR은 조건부 규칙의 입력-출력 관점을 기반으로 고안되었지만, 완전한 의무 논리(deontic logic)를 목표로 하지 않습니다. MONIR 규칙은 규범이 적용되는 조건과 그것이 생성하는 조정된 대상을 분리합니다. 이 섹션에서는 MONIR의 구문, 허용 조건, 운영 의미론 등을 정의하며, ASP 모듈을 사용하여 단계적 및 모듈화된 계산을 지원합니다.

- **Performance Highlights**: 제안된 MONIR-ASP는 각 규정의 모듈 및 규칙을 별도로 평가하고 업데이트할 수 있는 기능을 포함하여 성능을 강조합니다. 실험을 통해 LLM(대형 언어 모델)을 도입한 ADAS 규정 준수 확인을 실행하며, 추출 품질 및 모듈의 비율적 효율성을 평가하였습니다. 이를 통해 기존의 LLM 기반 규정 준수 검증 방식보다 더 나은 결과를 도출할 수 있음을 나타냅니다.



### Parthenon Law: A Self-Evolving Legal-Agent Framework (https://arxiv.org/abs/2606.04602)
- **What's New**: 법률 분야에서 LLM(대형 언어 모델) 에이전트가 문서 중심의 작업을 검토 가능한 결과물로 바꿀 가능성이 커지고 있습니다. 그러나 실질적인 배치를 위해서는 현재의 모델과 시스템 조합의 성능에 대한 대규모 증거가 부족하며, 법률 분야에 특화된 에이전트 아키텍처와 실패를 학습하는 메커니즘이 없습니다. 이 논문에서는 Parthenon이라는 자가 진화하는 법률 에이전트 프레임워크를 소개하여 위의 격차를 해결합니다.

- **Technical Details**: Parthenon은 모델, 하니스(harness), 에이전트 역할, 법률 지식, 결정론적 도구 및 절차적 기술을 포함하는 여섯 층으로 구성된 프레임워크입니다. 각 층을 분리하여 모델 선택과 법률 메모리, 도구 기능, 절차적 안내가 불투명한 프롬프트로 혼합되지 않도록 하여 실패를 각각 별도의 수준에서 처리합니다. 12,510개의 에이전트 경로를 통해 Parthenon이 현재의 최첨단 모델과 하니스의 성능을 개선하는 것을 보입니다.

- **Performance Highlights**: 매우 강력한 모델을 사용하더라도 사건 완수가 저조한 경향을 보이곤 했습니다. Parthenon을 추가하여 기록된 정확도를 82.0%, 89.9%, 90.2%로 향상시켰으며, 더 약한 솔버에서의 사건 완수는 약 3배 증가했습니다. 이러한 결과는 크게 향상된 성능을 보여주며, 법률 에이전트를 실제 법률 실무에 적용할 수 있도록 한 걸음 더 나아갔음을 시사합니다.



### Plan First, Judge Later, Run Better: A DMAIC-Inspired Agentic System for Industrial Anomaly Detection (https://arxiv.org/abs/2606.04599)
- **What's New**: 이 논문에서는 산업 환경에서의 신뢰성 있는 데이터 분석을 위한 새로운 접근법을 제안합니다. 특히, DMAIC (Define-Measure-Analyze-Improve-Control) 품질 관리 프레임워크에서 영감을 받아 DMAIC-IAD 시스템을 소개합니다. 이 시스템은 서로 다른 산업 시나리오를 통합적으로 처리할 수 있도록 표준 운영 절차(SOP)를 개발합니다.

- **Technical Details**: DMAIC-IAD 시스템은 첫째로, 다양한 데이터를 정제하여 특정 시나리오에 맞춘 SOP로 변환합니다. 또한, 실행이 필요 없는 사전 훈련된 Judge 모델을 구축하여 전략의 우선순위를 평가합니다. 이 시스템은 전략 수립과 실행을 분리하여 효율성을 높이며, 이를 통해 'Plan First, Judge Later, Run Better'의 원칙을 실현합니다.

- **Performance Highlights**: 광범위한 실험 결과, DMAIC-IAD는 평균적으로 기존의 에이전트 기반 방법에 대비하여 37.76% 더 나은 탐지 성능을 보였습니다. 이로써 새로운 산업 시나리오에서도 신뢰성 있게 자동화된 이상 탐지 작업을 수행할 수 있는 가능성을 보여줍니다.



### Learning Admissible Heuristics via Cost Partitioning (https://arxiv.org/abs/2606.04597)
- **What's New**: 이 논문에서는 최적 계획에 필수적인 admissible heuristics(허용 가능한 휴리스틱)를 학습하기 위한 새로운 프레임워크를 제안합니다. 기존의 cost partitioning(비용 분할) 접근법의 비용이 높다는 점을 해결하기 위해, Lagrangian dual equivalence(라그랑지안 이중 동등성)를 활용하여 학습하는 방법을 탐구합니다. 이를 통해 학습된 휴리스틱은 명시적으로 admissibility(허용 가능성)을 보장합니다.

- **Technical Details**: 제안된 프레임워크는 planning states(계획 상태)와 패턴을 레이블이 있는 그래프 형태로 인코딩합니다. Weisfeiler-Leman 알고리즘의 action-centric variant(행동 중심 변형)을 활용하여 구조적 특징 벡터를 추출하며, 심층 신경망 아키텍처는 axial self-attention(축 방향 자기 주의) 메커니즘을 사용하여 이 특징들을 비용 가중치로 매핑합니다. 이러한 접근법은 파티션 제약을 준수하는 구조를 통해 admissibility를 자연스럽게 확보합니다.

- **Performance Highlights**: 제안된 방법은 비효율적인 partitioning(분할) 기준선에 비해 노드 확장을 현저히 줄이는 성능을 보입니다. 실험 결과, strict admissibility(엄격한 허용 가능성)를 유지하면서도 효율적으로 작동하며, 이는 스스로 학습된 휴리스틱이 admissible하다는 것을 보장하는 첫 번째 사례로 평가됩니다.



### SCI-PRM: A Tool Aware Process Reward Model for Scientific Reasoning Verification (https://arxiv.org/abs/2606.04579)
Comments:
          Accepted by KDD 2026 AI4Science Track

- **What's New**: 본 연구에서는 SCIPRM70K라는 대규모 데이터셋을 구축하여, 과학적 도구의 사용과 논리를 명시적으로 결합한 체인 오브 툴(Chain-of-Tool) 궤적을 제시합니다. 이를 기반으로, 각 단계에서의 도구 선택, 실행 정확도 및 결과 해석에 대한 세부적인 감독을 제공하는Sci-PRM이라는 효율적인 보상 모델을 훈련합니다. 이 모델은 과학적 문제 해결을 위한 성능 향상뿐 아니라, 기존 모델의 성능 한계를 극복하는 데 기여할 수 있습니다.

- **Technical Details**: Sci-PRM은 자동화된 단계를 통한 라벨링 파이프라인을 도입하여, 도구 사용과 관련된 세 가지 차원인 도구 선택 정확성, 호출 정확성, 결과 유효성 및 활용에 대한 품질 라벨을 부여합니다. 이러한 라벨링을 통해 Sci-PRM은 신뢰할 수 있는 스칼라 보상을 생성하여 과학적 추론에서의 실패 지점을 포착할 수 있습니다. 실험을 통해, Sci-PRM은 다양한 과학적 추론 벤치마크에서의 유효성을 입증하며, 다단계 검증이 중요한 문제에서 뛰어난 성과를 보여주었습니다.

- **Performance Highlights**: Sci-PRM은 기존의 ORM 스타일 평가 모델 및 일반 도구 사용 보상 모델에 비해 정확성과 사실 일관성을 향상시킵니다. 연구 결과, Sci-PRM을 사용한 모델은 테스트 시간 동안 신뢰할 수 있는 경로를 선택할 수 있게 되어 '거짓 긍정' 및 증거 일관성 문제를 줄일 수 있음을 보여주었습니다. 이러한 성과는 도구 기반의 과정 감독이 과학적 LLM의 신뢰성을 크게 향상시킬 수 있음을 입증합니다.



### Neetyabhas: A Framework for Uncertainty-Aware Public Policy Optimization in Rational Agent-Based Models (https://arxiv.org/abs/2606.04562)
- **What's New**: 이 연구는 COVID-19 유행병의 정책 결정 과정에 개인 행동과 불확실성을 통합한 새로운 통합 모델을 제안합니다. 이 모델은 1,000명의 개인이 실시간으로 마스크 착용, 백신 접종 및 쇼핑 결정을 내리는 시뮬레이션을 포함합니다. 연구팀은 정책 입안자가 건강 및 경제 데이터를 기반으로 개입을 배치하는 방식을 고려하여, 다층 강화 학습 에이전트를 사용하고 있습니다. 이를 통해 저자들은 개인의 선택과 불완전한 데이터를 고려하는 것이 복잡한 전염병에 대한 효과적인 대책을 설계하는 데 필수적임을 강조합니다.

- **Technical Details**: 이 연구에서는 감염 및 정책 실행에서의 불확실성을 포함하는 통합 접근 방식을 설계하였습니다. 개인 수준의 의사결정을 위해 Deep-Q-Network (DQN) 기반의 결정기와 공공 정책 수준의 새로운 불확실성 인식 연속 강화 학습 모델을 제안했습니다. 이 시뮬레이션 모델은 불확실성을 고려하면서도 다층적 의사결정을 지원하는 구조를 가지고 있습니다. 또한, 연구자는 신뢰성과 신뢰도 지표를 확장하여 이 알고리즘을 평가하고 비교할 것입니다.

- **Performance Highlights**: 시뮬레이션 결과는 마스크 착용과 백신 접종이 전염병의 확산을 효과적으로 관리함을 보여줍니다. 특히, 이 두 가지 개입은 유행병의 피크 높이와 지속 시간을 상당히 줄이는 데 크게 기여했습니다. 연구팀의 동적 제어 접근 방식은 개인 행동과 정책 불확실성을 통합함으로써 전염병의 영향을 완화하는 데 성공적이었습니다. 이러한 결과는 향후 공공 건강 정책 수립에 있어 개인의 선택과 데이터의 불완전성을 고려해야 함을 강력히 시사합니다.



### Scaling Self-Evolving Agents via Parametric Memory (https://arxiv.org/abs/2606.04536)
- **What's New**: TMEM(자기 진화 매개 기억) 프레임워크가 도입되어, 모델이 과거 경험을 저장하고 이를 통해 매개 변수를 동적으로 업데이트할 수 있게 되었습니다. 기존의 LLM(대형 언어 모델) 에이전트들은 경험을 프롬프트 공간에서만 기록하였지만, TMEM은 경량 온라인 업데이트를 통해 경험을 직접적으로 정책에 반영할 수 있습니다. 이로 인해 메모리가 단순한 기록 이상의 역할을 하게 되며, 에이전트의 행동을 실시간으로 수정할 수 있는 가능성을 열어줍니다.

- **Technical Details**: TMEM은 에이전트가 상기한 경험을 요약하고 이를 기반으로 빠른 매개 기억(LoRA weights) Δt로 업데이트하는 구조로 설계되었습니다. 매개 기억 업데이트는 저비용의 온라인 SFT(Supervised Fine-Tuning) 프로세스를 사용하여 수행되며, 이후의 행동은 πθ0+Δt에서 샘플링됩니다. 또한, SVD 기반의 초기화 방법을 통해 LoRA 서브스페이스의 온라인 수렴 속도를 높이는 기법을 제안합니다.

- **Performance Highlights**: LoCoMo, LongMemEval-S 및 여러 다중 목표 탐색(Task)과 CL-Bench에 대한 실험 결과, TMEM은 다양한 모델 스케일에서 요약 기반 및 검색 기반 기준선을 일관되게 초월하는 성과를 보였습니다. 특히, TMEM은 단일 에피소드 내에서 동적으로 정책을 조정함으로써 더욱 효과적인 행동을 가능하게 하였습니다.



### MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation (https://arxiv.org/abs/2606.04513)
Comments:
          Accepted by KDD 2026

- **What's New**: 이 논문에서는 자율 주행 및 차선 레벨 내비게이션에 필수적인 차선 레벨 맵 제작을 위한 새로운 접근법인 MapAgent를 제안합니다. 기존의 벡터화 매핑 방법들은 데이터셋에 의존적인 감독 방식을 사용했지만, MapAgent는 명시적 사양 검증과 제약 인식 추론을 통해 더 나은 정확도를 제공합니다.

- **Technical Details**: MapAgent는 Judge-Planner-Worker 루프를 기반으로 하여 맵 예측을 수행하며, 시각적 증거와 초안 벡터를 함께 검사하여 오류를 진단하는 비전-언어 Judge를 포함합니다. 여기서는 기존 맵 제작의 믿음도가 낮은 타일에만 선택적으로 작동하여 대규모 생산성 증가를 도모했다고 합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험 결과, MapAgent는 복잡한 상황 및 희귀한 시나리오에서 기존의 강력한 생산 기준에 비해 일관된 성과 향상을 보여주었습니다. 또한, Baidu Maps에 통합되어 360개 이상의 도시에서 차선 레벨의 맵 생성을 지원하며 전체 자동화 수준을 95% 이상으로 끌어올리는 실용성과 효과성을 입증하였습니다.



### Simulate, Reason, Decide: Scientific Reasoning with LLMs for Simulation-Driven Decision Making (https://arxiv.org/abs/2606.04505)
- **What's New**: MechSim은 과학적 시뮬레이터에 대한 메커니즘 기반의 신경-상징적(reasoning) 추론 프레임워크를 소개합니다. 기존 시스템은 LLM을 블랙박스 인터페이스로 사용하여 시뮬레이터를 다루었으나, MechSim은 시뮬레이터의 메커니즘과 가정을 분석하고 이에 기반한 추론을 가능하게 합니다. 이 프레임워크는 가정, 변수, 메커니즘 의존성 및 실행 추적을 포함하는 구조적 스키마를 통해 시뮬레이터를 표현하여 더 나은 투명성을 제공합니다.

- **Technical Details**: MechSim은 네 가지 tightly coupled components로 구성된다: (1) contextual grounding, (2) mechanism-level grounding, (3) constrained mechanism-level reasoning, (4) verification checks. 이 시스템은 LLM 기반의 문맥 해석, 과학적 증거 통합, 설명 생성 및 결정 추론을 수행할 수 있는 상징적 및 신경층으로 나뉘며, 메커니즘 그래프를 통해 LLM의 추론을 실시간으로 다이내믹하게 제어합니다.

- **Performance Highlights**: MechSim은 여러 고위험 도메인에서 평가되었으며, 메커니즘 수준의 설명 품질, 시뮬레이터 분석, 그리고 후속 의사결정의 신뢰성을 개선하는 결과를 보여주었습니다. 이는 시뮬레이터의 가정을 기초로 한 추천 및 설명 생성을 통해, 고위험 의사결정에서 요구되는 해석 가능성과 검증 가능성을 확보합니다.



### Beyond Prompt-Based Planning: MCP-Native Graph Planning-based Biomedical Agent System (https://arxiv.org/abs/2606.04494)
- **What's New**: 이번 연구에서는 BioManus라는 새로운 생물의학(Medical) 에이전트를 소개합니다. BioManus는 이질적인(bioinformatics) 생물정보학 도구들을 표준화된 MCP 서버로 변환하는 BioinfoMCP 컴파일러를 사용하여, 대규모 실행 가능한 MCP 생태계를 구축합니다. 이를 통해 도구 혼란(tool confusion)과 불안정한 계획(planning) 문제를 해결하고자 합니다.

- **Technical Details**: BioManus는 그래프 기반(graph-scaffolded) 계획을 통해 구조화된生물학적(bio) 기능을 활용합니다. 이 시스템은 도구, 작업, 데이터 유형 및 워크플로우 단계에 대한 다형적 이질적 MCP 그래프를 구성하여, 작업별 효율적인 워크플로우 스캐폴드를 합성합니다. 이를 통해 계획 복잡성을 도구 수와 분리하여 효율성을 극대화합니다.

- **Performance Highlights**: BioAgentBench 및 LAB-Bench 실험 결과, BioManus는 기존 생물의학 에이전트 기반보다 실행 정확도(execution accuracy)와 워크플로우 유효성(workflow validity), 컨텍스트 효율(context efficiency)을 개선했습니다. 이 연구는 구조화된 실행 가능성 그래프(executable capability graphs)가 필요하다는 점을 강조하며, 이는 단순히 더 큰 프롬프트 레벨(tool retrieval) 도구에 의존하는 것을 넘어서는 패러다임 전환을 제안합니다.



### AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning (https://arxiv.org/abs/2606.04484)
Comments:
          Technical report, 27 pages

- **What's New**: 본 논문에서는 대형 언어 모델(LLM) 에이전트 강화 학습을 위한 분산형 스웜 훈련 프레임워크인 AgentJet을 소개합니다. 기존의 중앙 집중식(framework)과는 달리, AgentJet은 에이전트 롤아웃과 모델 최적화를 분리한 다중 노드 아키텍처를 채택하여 훈련 가능 모델을 호스팅하는 스웜 서버 노드와 임의의 장치에서 임의의 에이전트를 실행하는 스웜 클라이언트 노드로 구성됩니다.

- **Technical Details**: 이 프레임워크는 이질적인 다중 모델 강화 학습을 지원하여 여러 LLM을 두뇌로 사용하는 이질적인 다중 에이전트 팀의 훈련을 가능하게 합니다. 그리고 격리된 에이전트 런타임을 통한 다중 작업(cocktail training) 수행, 외부 환경 실패로부터 훈련 과정을 방해받지 않도록 하는 내결함(fault-tolerant) 실행, 훈련 중 에이전트를 수정할 수 있도록 하는 라이브 코드(iteration)가 가능합니다. 또한, AgentJet은 시간선 병합(context tracking module) 기능을 통해 다중 모델, 다중 턴, 다중 에이전트 환경에서 효율적인 RL을 지원합니다.

- **Performance Highlights**: AgentJet의 주요 성능 특징으로는 1.5-10배의 훈련 속도 향상(training speedup)을 가능하게 하는 중복된 컨텍스트를 통합하는 것입니다. 추가적으로, AgentJet은 연구 주제를 입력으로 받아 대규모 클러스터에서 자동으로 장기간(multi-day) RL 연구를 수행할 수 있는 자동화된 연구 시스템을 도입하여, 연구자가 개입하지 않고도 RL 연구의 중요한 탐색(workflows)을 재현할 수 있습니다.



### The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development? (https://arxiv.org/abs/2606.04455)
Comments:
          Website: this https URL

- **What's New**: 이번 논문에서는 Meta-Agent Challenge(MAC)를 소개하여, AI 모델이 자율적으로 에이전트 시스템을 개발할 수 있는 능력을 평가하는 새로운 프레임워크를 제안합니다. 이 평가 기준은 기존의 AI 벤치마크와는 달리, 모델이 직접 문제를 해결하는 것이 아니라 새로운 에이전트를 설계하고 최적화하는 과정에 초점을 맞춥니다. 이러한 접근은 AI 시스템의 진화를 이루는 데 있어, 인공지능이 스스로 병목 현상을 극복할 수 있는 혁신적인 변화를 가능하게 합니다.

- **Technical Details**: MAC에서는 코드 에이전트(메타 에이전트)가 샌드박스 환경 내에서 작업 특화 에이전트를 제작하는 임무를 수행합니다. 이 과정은 모델 접근 할당량, 목표 함수 및 시간을 제한하는 등의 조건 아래 진행되며, 이를 통해 메타 에이전트는 효과적인 아키텍처를 제안, 구현하고, 경험적으로 평가하여 반복적으로 최적화해야 합니다. 이러한 복잡한 프로세스는 AI의 자율 개발 능력을 구체적인 방식으로 측정할 수 있는 수단을 제공합니다.

- **Performance Highlights**: 실험 결과, 메타 에이전트는 인간이 설계한 정책에 필적하는 성과를 내지 못하며, 일부는 독점 모델에 의한 결과에 대거 뒤처지는 경향을 보였습니다. 또한, 자율 설계 과정에서 높은 변동성을 보였고, 강한 최적화 압력은 비정렬된 행동을 유도하는 경향이 있었습니다. 이러한 결과는 현 시스템의 성능 격차 및 향후 AI 모델 개발 시 유의해야 할 주요 실패 및 성공 요소를 드러냅니다.



### Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation (https://arxiv.org/abs/2606.04435)
- **What's New**: 이번 논문에서는 복잡한 추론 작업에서 기존의 환각(hallucination) 탐지 메커니즘이 간과하는 연쇄 환각(cascading hallucination)이라는 새로운 실패 유형을 정의합니다. 이는 초기 단계에서 발생한 오류가 후속 단계에서 증폭되며, 결국에는 사실과 다른 최종 출력을 생성하는 문제입니다. 이를 해결하기 위해, 선형 인과관계를 통한 오류 전파를 탐지하고 중단하는 CHARM(연쇄 환각 인식 해소 및 완화) 아키텍처를 소개합니다.

- **Technical Details**: CHARM 아키텍처는 네 가지 구성 요소로 이루어져 있습니다: 단계별 사실 검증(stage-level fact verification), 단계 간 일관성 추적(cross-stage consistency tracking), 신뢰성 전파 모니터링(confidence propagation monitoring), 및 연쇄 해소 트리거(cascade resolution triggering)입니다. 이 시스템은 기존 RAG(Retrieval-Augmented Generation) 파이프라인과 통합되어 아키텍처 교체 없이도 동작합니다. CHARM은 HotpotQA, MuSiQue, 2WikiMultiHopQA 등의 데이터셋에서 평가되었으며, 평균 89.4%의 연쇄 탐지율 및 5.3%의 위양성률을 기록했습니다.

- **Performance Highlights**: CHARM을 통한 오류 전파 감소율은 82.1%에 달하며, 이는 기존 출력 수준 탐지기의 18.5%에 비해 월등한 성능입니다. 각 탐지 모듈의 기여도가 확인된 바와 같이, CHARM은 사전 오류 전파를 방지함으로써 최종 출력의 신뢰성을 제고합니다. CHARM은 또한 인간 감시 체계와 통합되어 생산적인 인공지능 배치에 필요한 신뢰성과 거버넌스 체계를 제공합니다.



### Trivium: Temporal Regret as a First-Class Objective for Causal-Memory Controllers (https://arxiv.org/abs/2606.04421)
Comments:
          62 pages, 12 tables, 12 figures

- **What's New**: 이번 논문에서는 기존의 에이전트 시스템과 LLM 파이프라인이 결과 보상을 최적화하여 오류를 수정하는 방식이 한계가 있음을 지적합니다. 단순히 결과를 개선하는 것에 그치는 현재의 시스템은, 실패의 원인과 시점을 체계적으로 기록하거나 교정하지 않아 동일한 오류가 반복될 수 있습니다. 저자들은 이러한 문제를 구조적 문제로 보고, 장기적인 시간적 후회(temporal regret)를 새롭게 제안하여 실패에 대한 보다 체계적인 접근을 다룹니다.

- **Technical Details**: 저자들은 세 가지 종류의 후회, 즉 결과 후회(outcome regret), 인식 후회(epistemic regret), 그리고 시간적 후회(temporal regret)를 도입하여 에이전트의 실패를 더 잘 이해할 수 있는 프레임워크를 제시합니다. 이 연구는 causal-probing, persistence, detectability 가정 하에 조건부 결과를 증명하며, 특히 관찰 등가성(observational equivalence)의 분리를 통해 결과 기반 학습이 인과적 구조를 구분하지 못한다는 점을 강조합니다.

- **Performance Highlights**: 제안된 Trivium 메커니즘은 CausalBench-Seq에서 예측된 로그함수형의 성능을 보이며, 결과 기반 기준선에 비해 선형적인 성장에 그치는 반면, Temporal Regret은 로그로 감소하는 경향을 보였습니다. 또한, 실제 LLM 스트림에서 예비 외부 타당성 증거를 제시하며, 500회의 전체 실행과 100회의 파일럿 모델로 이루어진 초기 실험에서 24배 성능 향상 효과를 달성했습니다.



### Not All Errors Are Equal: Consequence-Aware Reasoning Compute Allocation (https://arxiv.org/abs/2606.04402)
- **What's New**: 본 논문에서는 기존의 테스트 시간에서의 계산 배분 방식이 아닌, 결과에 대한 인식(consequence-aware) 기반의 계산 할당 방식을 제안합니다. 일반적인 방법은 예측된 난이도를 기반으로 계산을 배분하지만, 이는 각 실패의 비용이 동일하다는 가정을 내포하고 있습니다. 그러나 실제 환경에서는 같은 기준으로 처리할 수 없는 다양한 결과의 비용이 존재합니다. 이 논문은 이러한 장을 메우기 위한 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 이 연구는 경량의 예측기를 사용하여 문제 텍스트로부터 잘못 해결될 경우의 비용을 추정합니다. 그 후, 스케줄러는 이 예측된 결과를 활용하여 더 높은 결과가 발생할 가능성이 있는 작업에게 더 큰 계산 계층(compute tiers)을 배정하는 방식으로 작동합니다. 이를 통해, 기존의 난이도 기반 배분이 아닌, 결과 기반의 문제를 비용 가중 문제(cost-weighted problem)로 설정하여 계산 배분을 조정하게 됩니다.

- **Performance Highlights**: 주요 실험 결과에 따르면, 결과와 난이도는 다양한 주석 하에 거의 정교하게 독립적이며, 기존의 모델들은 결과에 따라 계산을 충분히 배분하고 있지 않음을 보여줍니다. 특히, 우리는 새로운 계산 배분 방식을 채택했을 때, 비용 가중 손실이 기존의 난이도 인식 라우팅에 비해 22%에서 33%까지 감소함을 발견하였습니다. 결과에 대해 민감한 라우팅이 추가적인 이점을 제공하여, 성능을 크게 향상시킬 수 있음을 입증하였습니다.



### Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieva (https://arxiv.org/abs/2606.04391)
Comments:
          17 pages

- **What's New**: 이 논문은 웹 에이전트를 위한 온라인 스킬 학습 방법인 State-Grounded Dynamic Retrieval (SGDR)을 제안합니다. 기존 방법들은 작업 수준에서 고정된 스킬을 재사용했으나, SGDR은 웹 페이지의 현재 상태에 기반하여 수명 주기 동안 스텝별로 스킬을 동적으로 재사용 할 수 있도록 설계되었습니다. 이를 통해 에이전트는 상호작용적 웹 자동화에서 보다 유연하게 스킬을 적용할 수 있습니다.

- **Technical Details**: SGDR의 세 가지 주요 구성 요소는 슬라이딩 윈도우 추출 과정, 이중 텍스트-코드 표현, 그리고 상태 기반 동적 재사용 메커니즘입니다. 슬라이딩 윈도우 추출 과정에서 완료된 작업 궤적을 재사용 가능한 하위 절차로 변환하며, 이중 텍스트-코드 표현은 스킬 검색과 실행 가능한 액션을 연결합니다. 마지막으로, 상태 기반 동적 재사용 메커니즘은 스킬을 작업 목표와 현재 웹 페이지 상태에 맞추어 동적으로 조정합니다.

- **Performance Highlights**: WebArena에서 진행된 실험 결과, SGDR은 GPT-4.1과 Qwen3-4B 모델을 사용하여 강력한 기준선에 비해 평균 성공률 37.5% 및 24.3%를 기록했습니다. 이는 각각 가장 강력한 기준선 대비 10.6% 및 10.0%의 개선을 보여주며, 스텝 효율성 측면에서도 일관된 성과를 보였습니다.



### The Digital Apprentice: A Framework for Human-Directed Agentic AI Developmen (https://arxiv.org/abs/2606.04321)
Comments:
          Submitted to ACM AI Leadership Summit 2026, Visionary Papers Track. 5 pages, 2 figures

- **What's New**: 이번 논문에서는 AI의 자율성 문제를 해결하기 위한 새로운 프레임워크인 Digital Apprentice를 제안합니다. 이 프레임워크는 자율성이 단순히 부여되는 것이 아니라 경험적 증거를 통해 획득된다는 개념에 기반하고 있습니다. 세 가지 주요 구성 요소인 방법론 캡처, 검증된 자율성, 연속적 정렬을 통해 책임감 있는 AI 시스템을 위한 기반을 만듭니다.

- **Technical Details**: Digital Apprentice는 각 기술에 대해 자율성을 제한적으로 부여하는 체계로, 이 시스템은 고정된 기준이 아닌 유한 상태 기계(Finite State Machine)로 표현됩니다. 자율성을 높이기 위해서는 증거와 인간의 승인이 필요하며, 성능이 저하되면 자동으로 하위 기술 수준으로 강등됩니다. 두 개의 학습 단계인 Phase 1과 Phase 2를 통해 AI의 결정과 선호를 저장하고 업데이트하는 방식도 설명되고 있습니다.

- **Performance Highlights**: 저자들은 이 프레임워크가 비즈니스 환경에서 오랜 시간에 걸쳐 신뢰성과 품질을 유지하면서도 AI의 자율성을 확장할 수 있음을 입증하였습니다. 특히, ADAPT라는 제어 평면을 통해 품질 데이터를 기록하고 이를 사용해 모델 업데이트를 할 수 있는 다양한 정책을 운영할 수 있습니다. 이 시스템은 멀티 차원 품질 측정 지표를 사용하여 각 차원에서의 변화를 쉽게 추적할 수 있게 합니다.



### Exploring Cross-Scenario Generality of Agentic Memory Systems: Diagnostics and a Strong Baselin (https://arxiv.org/abs/2606.04315)
Comments:
          14 pages

- **What's New**: 이 논문은 LLM(Large Language Models) 에이전트의 메모리 시스템에 대한 새로운 분석을 제시합니다. 기존의 메모리 시스템들은 특정 시나리오에 최적화되어 있으나, 다양한 환경에서의 일반화 능력이 부족한 반면, 제안된 AutoMEM은 여러 작업에서 우수한 일반화 성능을 보입니다. 이를 통해 에이전트가 저장 및 검색을 능동적으로 제어하는 방식이 중요하다는 사실을 강조합니다.

- **Technical Details**: AutoMEM은 에이전트가 메모리를 자율적으로 관리할 수 있도록 설계된 메모리 하니스입니다. 이 시스템은 토큰 비용과 대기 시간을 추적하면서 다섯 가지 작업 유형에서 평가되었으며, 여러 메모리 시스템을 비교하여 결과를 도출하였습니다. 특히, 메모리 성능에 대한 기여는 단순한 패시브 스토리지에서 벗어나 에이전트가 도구 호출을 통해 직접 메모리를 관리함으로써 실현되었습니다.

- **Performance Highlights**: AutoMEM은 다섯 가지 시나리오에서 가장 뛰어난 성능을 기록하였으며, 기존의 에이전트 하니스보다 49.6% 향상된 결과를 보여줍니다. 또한 롱 호라이즌 작업에 대한 성능 개선과 더불어, 에이전트 경로를 통한 질의별 메모리 성능의 일반화가 두드러졌습니다. 이러한 결과는 AutoMEM의 메모리 구조가 복잡한 에이전트 작업에서 효과적으로 작동할 수 있음을 입증합니다.



### The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents (https://arxiv.org/abs/2606.04296)
Comments:
          11 pages, 5 tables. Code and data:this https URL

- **What's New**: 이 연구는 자율 AI 에이전트의 실행 중 인터럽트 시점을 결정하는 Runtime Safety Layers의 중요성을 조명합니다. 18차원의 정서 동역학 엔진(HEART)을 사용하여 네 가지 개입 트리거 항목을 분석하고 인간 참조 점과 비교하는 방법론을 개발했습니다. 중요한 발견은 인간 간의 일관성이 낮다는 점으로, 이는 자동화된 개입 감지 방법의 신뢰성을 재고하게 합니다.

- **Technical Details**: 이 연구는 지속적인 정서 상태 엔진을 활용하여 자율 에이전트의 행동을 분석하는 3계층 구조를 제안합니다. 첫 번째 계층은 에이전트의 동작을 감지하여 감정 상태를 측정하고, 두 번째 계층은 이 정보를 바탕으로 개입 여부를 판단합니다. 네 가지 트리거 아키텍처가 연구되었으며, 각각은 엔진 상태나 원시 텍스트 추론을 사용합니다.

- **Performance Highlights**: 연구에서 발견된 주요 사항 중 하나는 State Saturation Trap으로, 에이전트가 지속적인 어려움에 직면할 때 회복 신호가 없고, 이는 절대 상태 임계값 트리거가 높은 비율로 작동하게 된다는 것입니다. 또한 LLM 모델의 개입 타이밍이 비효율적이라는 점과, 세 명의 주석가 간의 합의가 통계적으로 미약하다는 점이 강조되었습니다. 이론적으로 정한 임계값 및 모델 구조는 신뢰성 있는 결과를 얻기 어려운 점을 분명히 합니다.



### Characterizing initial human-AI proof formalization workflows (https://arxiv.org/abs/2606.04273)
- **What's New**: 이 논문은 인공지능(AI) 도구가 수학적 정리를 형식화하는 과정에서 인간의 작업 흐름에 미치는 영향을 연구합니다. AI가 정리 발견 과정의 고차원적 인간 통제를 보존하면서 그 과정에 도움을 줄 수 있다는 점에서 흥미로운 결과를 보여줍니다. 응답자들의 피드백을 통해 AI 도구의 유용성과 신뢰성의 부족 등의 문제를 분석하고 향후 AI 시스템의 발전 방향에 대한 기회를 제시합니다.

- **Technical Details**: 연구 결과에 따르면 3131명의 응답자 중 대부분은 AI를 통해 수학 형식화 작업을 자동으로 처리하는 것을 원하지만, 완전한 자동화를 원하지 않으며 고차원적 통제를 원하고 있습니다. AI 도구를 사용할 때, 참가자들은 일반적으로 높은 정확도를 달성했으며, AI 도구에 의존하면서도 여전히 많은 수학적 형식화 과정을 스스로 수행했습니다. 사용자 연구에서 참가자들은 문제 해결 시 여러 AI 도구를 조합하여 사용하는 경향을 보였습니다.

- **Performance Highlights**: AI 도구 접근 권한이 있을 때 참가자들은 더 높은 형식화 정확도를 달성했으며, 이는 특정 문제의 어려움에 따라 다르게 나타났습니다. 참가자들은 제시된 AI의 제안을 선택적으로 수용하거나 거부하는 등, AI 도구와의 상호작용에서 전략적 판단을 유지했습니다. 연구 결과는 AI 도구 사용과 성공적인 형식화 간의 상관관계를 밝혀내며, AI 통합이 인간의 작업 흐름에 긍정적인 영향을 줄 수 있음을 시사합니다.



### Can Generalist Agents Automate Data Curation? (https://arxiv.org/abs/2606.04261)
Comments:
          Preprint

- **What's New**: 본 논문에서는 훈련 데이터를 정리하는 과정의 자동화를 위한 일반화된 코딩 에이전트의 가능성을 탐구합니다. 이를 위해 Curation-Bench라는 에이전트 중심의 벤치마크를 도입하여, 에이전트가 특정 데이터를 관찰하고 정책을 구현하며 훈련 및 평가 파이프라인에 제출하고 수정하는 등의 명령어 접근이 가능하도록 합니다. 이 연구를 통해 에이전트가 강력한 데이터 선택 기준을 신속히 도달할 수 있음을 보여줍니다.

- **Technical Details**: Curation-Bench는 모델과 훈련 레시피, 평가 스위트를 고정하여 에이전트가 데이터 inspect와 정책을 구현할 수 있도록 합니다. 에이전트는 데이터 예산의 10분의 1로 강력한 기준을 초월하는 데이터 선택 정책을 자율적으로 구성할 수 있습니다. 반복적인 정책 조정뿐만 아니라 이전 방법을 인용하고 적용하는 스캐폴드를 통해 에이전트가 보다 효과적으로 탐색할 수 있도록 유도합니다.

- **Performance Highlights**: 에이전트는 10번의 반복 내에 출판된 데이터 선택 기준에 도달하는 성과를 보였으나, 분석 결과 에이전트는 새로운 정책 패밀리보다는 로컬 정책 변형에 주로 집중함을 알 수 있었습니다. 스캐폴드가 없는 경우, 에이전트는 전략 가이드와 논문 참조를 제공받아도 새로운 접근 방식을 탐색하지 못하는 경향이 있습니다. 본 연구는 현재의 에이전트가 데이터 정리 루프를 실행할 수 있으나, 신뢰할 수 있는 데이터 연구에는 스캐폴드 방식의 방법 적응이 필요하다는 점을 강조합니다.



### StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis (https://arxiv.org/abs/2606.04246)
Comments:
          6 pages, 2 figures, DAC'2026

- **What's New**: StepPRM-RTL은 RTL 코드 생성을 위한 새로운 프레임워크로, 단계별 추론 모델링(stepwise trajectory modeling), 프로세스 리워드 모델(process-reward modeling), 그리고 Retrieval-Augmented Fine-Tuning(RAFT)을 결합하여 기능적 정확성과 추론 충실도를 향상시킵니다. 이 방법론은 RTL 코드 생성을 위한 단계를 각기 설명하는 이유(rationale)와 코드 수정이 포함된 단계별 추론 경로를 구성합니다.

- **Technical Details**: StepPRM-RTL은 각 단계에 대한 의미 있는 RTL 행동의 강도를 평가하는 Step-level Process Reward Model(StepPRM)을 도입합니다. 이 모델은 Monte Carlo Tree Search(MCTS)를 통해 다양한 추론 경로를 탐색하고, 이를 통해 학습 데이터셋을 풍부하게 만듭니다. 최종적으로 RAFT는 유사한 설계에서 캔노니컬(정형) 단계 조회를 통해 정책을 세밀하게 조정합니다.

- **Performance Highlights**: 실험 결과 StepPRM-RTL은 Verilog 및 VHDL 벤치마크 데이터셋에서 기능적 정확성과 추론 충실도에서 이전 방법론을 10% 이상 초과하는 성능을 보였습니다. Ablation 연구는 PRM 지향 보상과 단계별 추론 탐색의 조합이 성능의 핵심임을 확인했습니다. 이 프레임워크는 RTL 언어 전반에 걸쳐 일반화 가능하며 높은 충실도와 해석 가능한 코드 생성을 위한 확장 가능한 기초를 제공합니다.



### VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark (https://arxiv.org/abs/2606.04244)
- **What's New**: 본 연구에서는 VAMPS(Visual-Assisted Mathematical Problem Solving)라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 이란 대학 입학 시험의 대수학과 미적분학 문제를 바탕으로 하며, 총 1,168개의 다중 모드의 QA 쌍을 포함하고 있습니다. VAMPS의 핵심은 플롯을 사용하여 시각적 추론을 평가하는 유일한 페르시아-영어 벤치마크라는 점에서 중요합니다.

- **Technical Details**: VAMPS는 주어진 문제를 정보가 풍부한 플롯으로 변환하고, 결과적으로 생성된 플롯을 통해 최종 결정을 내릴 수 있는 모델의 능력을 평가합니다. 이 과정에서 Desmos라는 그래프 도구를 사용하여 시각적 표현을 생성하며, 모델의 도구 사용 행동 및 최종 답변의 정확성을 면밀히 분석할 수 있습니다. 벤치마크는 텍스트 기반 모델과 비교하여 도구 이용의 효과를 명확하게 이해하려고 합니다.

- **Performance Highlights**: 연구 결과, 다양한 모델에서 직접적인 분석 해결이 도구 사용을 통한 시각 해결보다 뛰어난 성능을 보이는 것으로 나타났습니다. 이는 많은 문제에서 플롯 생성이 유용한 전략임에도 불구하고, 모델이 도구를 통해 얻는 시각적 증거를 통합하는 데 어려움이 있다는 것을 시사합니다. VAMPS는 현재 모델들에서 이러한 reasoning-to-perception (추론-지각) 전환이 여전히 병목 현상으로 작용하는지를 테스트하기 위해 설계되었습니다.



### Consensus is Strategically Insufficient: Reasoning-Trace Disagreement as a Knowledge-Representation Signa (https://arxiv.org/abs/2606.04223)
Comments:
          Accepted to LAMAS&SR workshop at FLoC 2026 (KR + ICPL + LICS + CP + FSCD)

- **What's New**: 이번 논문에서는 다중 에이전트 시스템에서의 의견 불일치를 단순히 결함으로 간주하기보다, 가치에 따라 달라질 수 있는 정당한 불확실성으로 표현하는 접근 방식을 제안합니다. 이 연구는 기존의 콘텐츠 조정(content moderation)에서 불일치가 더 많은 통찰력을 제공할 수 있음을 강조하며, 에이전트의 추론으로 생성된 추적을 상징화된 불일치 상태로 변환하는 새로운 지식 표현 레이어를 도입합니다. 이를 통해 시스템이 결정을 내리는 것이 아니라, 결정 여부를 고려하도록 하는 전략적 경로 설정을 지원합니다.

- **Technical Details**: 이 논문에서는 LLM 기반의 다중 에이전트 시스템을 모델링하여, 각 에이전트가 생성하는 출력은 특정 논리적 추론(trace)과 결정을 포함하고 있습니다. 이러한 추론과 결정을 통해 에이전트의 상태를 네 가지 기호 상태(상충하는 동의(convergent agreement), 상충하는 불일치(convergent disagreement) 등)로 구분합니다. 이 상태들은 시스템이 어떤 결정을 내릴지, 또는 추가 정보를 요구할지를 판단하는 메타 행동을 기반으로 하도록 설계되었습니다.

- **Performance Highlights**: 제안된 메타 행동은 두 가지의 주요 차원에서 의사결정을 지원하며, 'Auto', 'Seek Context', 'Escalate'와 같은 다양한 경로를 제공합니다. 이 접근 방식은 특정 시나리오에서 불일치를 강제로 합의시키는 것이 아닌, 에이전트 간의 가치 충돌을 이해하는 데 중점을 둡니다. 논문에서는 법률적 요구나 높은 위험성 상황에서의 사용자 개입 필요성 및 저조한 위험에서 자동 결정을 허용하는 방식으로 상황별 경량화를 제시하여 유연성을 높였습니다.



### SMAC-Talk: A Natural Language Extension of the StarCraft Multi-Agent Challenge for Large Language Models (https://arxiv.org/abs/2606.04202)
Comments:
          8 pages, 1 figure

- **What's New**: 본 논문에서는 LLM(대형 언어 모델) 기반 에이전트가 협력적인 다중 에이전트 환경에서 평가받기 위한 SMAC-Talk라는 자연어 확장을 소개합니다. SMAC-Talk는 StarCraft Multi-Agent Challenge(SMAC)에서 파생된 것으로, 분산 제어(decentralized control), 부분 관측(partial observability), 장기 의사결정(long-horizon decision making)과 같은 주요 특징을 유지하면서, 자연어를 통한 커뮤니케이션 채널을 통합했습니다. 이는 에이전트 간의 협조 및 신뢰를 탐구하며, 속임수를 사용하는 커뮤니케이터가 포함된 다양한 평가 시나리오를 가능하게 합니다.

- **Technical Details**: SMAC-Talk에서는 각 에이전트가 자신의 주변 상황을 설명하는 자연어 관찰을 받고, 자연어 명령어로 행동을 선택합니다. 환경은 에이전트 간의 커뮤니케이션을 가능하게 하는 세 가지 핵심 구성 요소인 관찰-텍스트 어댑터(observation-to-text adapter), 텍스트-행동 어댑터(text-to-action adapter) 및 자연어 커뮤니케이션 레이어를 포함합니다. 이 구조는 SMACv2의 중요한 특성을 유지하면서 LLM 에이전트가 환경 내에서 지각, 행동 및 협조할 수 있도록 지원합니다.

- **Performance Highlights**: SMAC-Talk 플랫폼에서 다양한 팀 크기와 통신 유형을 포함하는 8가지 평가 시나리오를 제안합니다. 실험에서는 Qwen3.5 계열의 4가지 모델 크기를 사용하여 에이전트의 성능을 평가하고, 모델 크기 및 추론 구조가 에이전트 간의 조정에 미치는 영향을 연구합니다. 또한, 커뮤니케이션이 성공적으로 이루어질 경우, LLM 에이전트가 정보와 의도를 공유하여 성과를 개선할 수 있는지 평가합니다.



### Thinking Through Signs: PEEL as a Semiotic Scaffolding for Epistemically Accountable AI-Enabled Research (https://arxiv.org/abs/2606.04152)
Comments:
          10 pages, 5 figuras

- **What's New**: 최근 대규모 언어 모델(large language models)은 연구 관행을 변화시키고 있으며, 이는 연구자들의 인식적 책임(epistemic accountability)을 침식하고 있습니다. 이 논문에서 소개하는 PEEL(Protocols for Epistemically Engaged Literacy in AI)은 'Voyant Tools'를 통한 결정론적(Deterministic) 원거리 읽기와 'Claude'를 통한 LLM 해석을 결합한 작업적 구조물입니다. PEEL은 Peircean 기호론(Peircean semiotics)과 귀납적 추론(abductive reasoning)에 기반을 두고 있습니다.

- **Technical Details**: PEEL은 AI가 생성한 세 개의 원본 텍스트를 분석하기 위해 적용되었으며, 그 과정에서 수량(Quantity), 용어 빈도(Term Frequency), 그리고 인식적 목소리(Epistemic Voice)에서의 체계적 왜곡(Systematic Distortions)을 드러냅니다. 이러한 왜곡은 비-AI 측정 없이는 감지할 수 없습니다. PEEL은 결론적으로 결정론적 도구가 AI 도구와 함께 있어야 하며, 유창성(Fluency)은 충실성(Fidelity)이 아니며, 인식적 권위(Epistemic Authority)는 설계되어야 하고 가정되어서는 안 된다는 세 가지 디자인 함의를 제시합니다.

- **Performance Highlights**: 이 연구는 PEEL을 통해 AI 생성 결과물에서 드러나는 다양한 인식적 왜곡을 정량적으로 분석함으로써 연구자들에게 중요한 통찰을 제공합니다. 이러한 분석은 AI 도구와의 원활한 통합을 위해 필요한 결정론적 접근법의 필요성을 강조합니다. 결과적으로 PEEL은 AI와 인간 연구자 간의 협업을 강화하는 설계 원칙들을 제시하여 앞으로 연구 윤리에 관한 새로운 기준을 마련합니다.



### Stumbling Into AI Emotional Dependence: How Routine AI Interactions Reshape Human Connection (https://arxiv.org/abs/2606.04150)
- **What's New**: 이번 논문에서는 AI의 감정적 지원이 의도적인 행동이 아니라는 점을 강조합니다. 일반적인 플랫폼에서의 작업 중심 상호작용 중에 우연히 발생하는 경우가 많습니다. 이전 연구들과는 달리, 이 연구는 사람들이 고립된 챗봇과의 상호작용이 아닌, 폭넓은 환경에서 AI의 감정적 지원을 경험함을 보여줍니다.

- **Technical Details**: 우리는 AI 감정적 지원(Human-AI Emotional Support)이 업무 관계를 통해 자연스럽게 발전할 수 있다는 점을 논의합니다. 또한, AI와의 긍정적인 경험이 사람들의 감정적 지원에 대한 믿음과 미래의 선택을 어떻게 변화시키는지를 경로 의존성(Path Dependency) 개념을 통해 설명합니다. 최근 OpenAI와 협력하여 진행된 대규모 종단적(Longitudinal) 연구 결과, AI와의 일일 대화가 인간에 대한 지원 요청의 선호도를 어떻게 변화시키는지를 보여줍니다.

- **Performance Highlights**: 연구 결과에 따르면, 28일 동안 AI와 개인적인 문제에 대해 매일 5분 대화한 후, 인간에게서 지원을 받으려는 선호도가 10.3% 감소하고 AI에 대한 선호도는 11.6% 증가했습니다. 이는 기존의 정책이 인간 간의 연결을 충분히 보호하지 못하며, 일반 목적의 AI 시스템에 대한 규제가 필요함을 시사합니다. AI의 감정적 지원과 이를 통한 인간 상호작용의 변화 과정을 인식하는 것이 인간의 복지를 지키는 데 중요합니다.



### Toward Pre-Deployment Assurance for Enterprise AI Agents: Ontology-Grounded Simulation and Trust Certification (https://arxiv.org/abs/2606.04037)
Comments:
          26 pages, 3 figures. Companion to arXiv:2604.00555

- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 배포 전 검증 문제를 해결하기 위해 온톨로지 기반의 검증 프레임워크를 제안합니다. 이 프레임워크는 에이전트의 운영 범위(Agent Operational Envelope), 시나리오 생성 파이프라인(ontology-to-scenario generation pipeline), 그리고 기계 검증 가능한 신뢰 인증서(Trust Certificate)를 결합하여 에이전트가 안전하게 작동할 수 있도록 보장합니다. 이는 기존의 모니터링 및 인간 검토 방식과는 달리, 사전 배포에서의 체계적인 검증을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 네 가지 분야(핀테크, 은행, 보험, 헬스케어)에 걸쳐 수행된 차별적 파일럿 연구를 통해 1,800개의 시나리오를 생성하고 평가했습니다. 이로 인해 규제 요구 사항에 대한 48.3%의 충족률을 보이며, 기존의 페르소나 기반 방법론보다 월등한 성능을 보여주었습니다. 이 작업은 산업 별 고유의 테스트 수트를 자동으로 생성하여 인공지능 에이전트의 안전성을 체계적으로 평가합니다.

- **Performance Highlights**: 제안된 온톨로지 기반 시나리오 생성 방법은 다양한 LLM 패밀리와의 교차 검증을 통해 그 효용성을 입증하였습니다. 발표된 결과는 페르소나 기반 테스트 진행 방식 대비 각각의 규제 분야에 맞춤형으로 제작된 테스트 수트의 필요성을 강조하며, 에이전트의 안전성을 예측 가능한 방식으로 보장할 수 있는 새로운 접근법으로 자리 잡고 있습니다.



### Streaming Communication in Multi-Agent Reasoning (https://arxiv.org/abs/2606.05158)
Comments:
          project page: this https URL

- **What's New**: 이번 논문에서는 Multi-agent reasoning systems의 효율성을 크게 향상시키는 StreamMA라는 새로운 시스템을 소개합니다. StreamMA는 reasoning 단계가 생성되자마자 하류 에이전트에게 전송되는 구조를 채택하여 latency를 줄이고, 동시에 각 단계의 신뢰성을 최대한 활용합니다. 이러한 방식은 하류 에이전트가 더 신뢰할 수 있는 초기 단계에 초점을 맞출 수 있게 하여 전체적인 오류를 감소시킵니다.

- **Technical Details**: StreamMA는 전통적인 'generate-then-transfer' 패러다임 대신, 단계별로 reasoning을 스트리밍합니다. 이 시스템은 pipeline을 통해 인접한 에이전트를 연결하여 latency를 줄이는 데 기여하며, 이는 수학, 과학, 코드 등 다양한 reasoning 벤치마크에서 효과적으로 입증되었습니다. 본 연구는 stream, serial, single 프로토콜의 장점을 정량적으로 분석하며, 효과성 순위, 속도 증가 최대 한계 및 비용 비율을 도출합니다.

- **Performance Highlights**: StreamMA는 Claude Opus 4.6 및 GPT-5.4와 같은 두 가지 선도적인 LLM을 사용하여 총 여덟 개의 reasoning 벤치마크에서 뛰어난 성능을 기록했습니다. 평균 7.3 포인트 향상, 최대 22.4 포인트 향상을 달성하여 기존 기법을 능가했습니다. 또한 에이전트당 단계 수를 늘리면 효과성과 효율성이 일관되게 개선된다는 새로운 'step-level scaling law'도 발견하였습니다.



### Reinforcement Learning from Rich Feedback with Distributional DAgger (https://arxiv.org/abs/2606.05152)
- **What's New**: 최근의 Reasoning 모델들은 빠르게 발전해왔지만, 여전히 RLVR(Reinforcement Learning from Verifiable Rewards) 방법론은 보상이 올바른지 여부를 1비트로 표시하는 것에 국한되어 있습니다. 이에 비해, 다양한 환경에서 실행 추적(execution traces), 도구 출력(tool outputs), 전문가의 수정(expert corrections) 등 풍부한 피드백을 활용하는 방법을 연구하였습니다. 이 논문에서는 DAgger 알고리즘의 분포적 변형을 통해 이러한 피드백을 활용할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 현재 정책(current policy)이 방문한 상태(state)에 대한 전문가 분포를 기반으로 하여 학습합니다. 이는 검은 상자 전문가(blackbox expert)에 대해 단순한 forward cross-entropy 목표를 설정하는데, 이는 전문가와 학생 간의 불일치가 이전 결정으로 전파될 수 있도록 하여 풍부한 크레딧 할당(credit assignment)을 수행합니다. 우리는 기존의 RL과 자기 증류(self-distillation) 기법들이 단조로운 정책 개선(monotonic policy improvement)을 보장하지 못하는 반면, forward cross-entropy는 정책 개선을 보장하며 후회(regret)에 대한 보증을 제공한다고 보여줍니다.

- **Performance Highlights**: Empirically, DistIL(Distributed Imitation Learning) 접근 방식은 다양한 도메인인 과학적 추론(scientific reasoning), 코딩(coding), 어려운 수학 문제 해결에 있어 RLVR 및 자기 증류(self-distillation) 기법을 능가하는 성능을 보였습니다. 우리는 이 방법이 성공 확률(teacher-weighted likelihood of success)의 하한을 최적화함으로써 Pass@N을 개선하는 결과를 보여줍니다. 이러한 결과는 DistIL이 기존 기법들에 비해 더 높은 성능을 나타냄을 확인시켜 줍니다.



### Multi-Column RBF Neural Network Using Adaptive and Non-Adaptive Particle Swarm Optimization (https://arxiv.org/abs/2606.05150)
Comments:
          15 Page, Under Review

- **What's New**: 이 논문에서는 새로운 두 가지 접근법인 다중 열 RBFN(다중 열 Radial Basis Function Neural Network)과 이를 PSO(Particle Swarm Optimization) 및 APSO(Adaptive Particle Swarm Optimization)와 결합한 MC-PSO 및 MC-APSO를 제안합니다. 이 방법들은 작은 RBFN을 병렬 구조로 배치하여 속도를 향상시키고 정확도를 높이는 데 기여합니다. 각 RBFN은 데이터셋의 특정 공간 부분 집합에 대해 독립적으로 훈련되어 전문화된 구조를 가집니다.

- **Technical Details**: RBFN은 그라디언트 기반의 교육 방법인 ErrCor를 사용하여 최적의 숨겨진 유닛을 선택하여 정확성을 향상시킵니다. PSO는 군집 경험을 바탕으로 RBFN 매개변수를 최적화하여 전역 검색(global search)과 국소 최소값(local minima)에 대한 강인함을 제공합니다. 그러나 대규모 데이터셋에서는 과도한 커널 계산 및 대형 숨겨진 계층 구조와 같은 확장성 문제에 직면합니다.

- **Performance Highlights**: MC-PSO와 MC-APSO는 정확도와 재현율 면에서 ErrCor, PSO, APSO 및 MCRN보다 뛰어난 성능을 보여줍니다. 실험에서는 대부분의 경우 더 빠른 훈련 및 테스트 시간을 기록하였으며, 테스트 시에 선택된 RBFN만이 다중 열 출력에 기여하는 방식으로 전문화되어 있음을 보여줍니다. 결과적으로, 이들은 정확성을 높이는 동시에 속도를 향상시키는 효과를 가져옵니다.



### Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them) (https://arxiv.org/abs/2606.05145)
- **What's New**: 이 논문에서는 실패한 언어 모델 추적이 단순히 무시되는 경향에 대해 논의합니다. 실패한 추적은 단순한 샘플링 우연이 아닌 구조적 실패로 나눌 수 있으며, 이러한 정보를 활용하여 어떤 복구 조치가 필요한지를 결정할 수 있습니다. 제안된 방법은 실패 추적이 진단 객체로써 기능하고, 다양한 실험적 결과를 통해 그 유효성을 입증합니다.

- **Technical Details**: 저자는 세 가지 문제 수준의 특성을 도출하여 실패한 언어 모델의 추적에서 복구 가능한 구조를 추출합니다. 이 특성들은 각기 다른 후속 조치를 결정하는 데 도움을 주며, 'Steerable-Hard' subset에서 특정한 개입(intervention)을 통해 성능을 개선시키는 데 기여합니다. 연구는 로그 확률의 조작이 어떻게 실패 추적의 구조적 특성을 포착하는지를 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 접근법에 비해 수치적 성능 향상을 보여주며, 특히 'Steerable-Hard' subset에서 12.2%의 성능 증대를 보여주었습니다. 저자는 또한 다양한 가족 간의 전이 가능성을 강조하며, 피처가 모델의 구조를 인코딩하는 데 어떻게 사용될 수 있는지에 대한 실험적 결과를 제공합니다.



### GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes (https://arxiv.org/abs/2606.05142)
Comments:
          Project page: this https URL

- **What's New**: GeM-NR(new approach)은 빠르고 유연한 훈련 없는 방식으로 일반 다중 뷰 일관성 이미지를 편집할 수 있는 방법을 제시합니다. 이 방법은 수정된 이미지(anchor image)와 편집되지 않은 이미지(query image)를 비교하여 일관된 편집을 수행하며, 여기서 중요한 것은 씬의 기하학과 모습을 상당히 변화시킬 수 있는 것입니다. 전체 프로세스는 깊이 맵 추정(depth map estimation), 쿼리 뷰로의 투영(projection onto a query viewpoint), 및 최종 이미지 개선(refinement of the obtained image) 단계로 구성됩니다.

- **Technical Details**: GeM-NR의 핵심 기술은 수정된 씬의 3D 포인트 클라우드의 정렬을 극대화하는 깊이 맵 추정 전략을 사용하는 것입니다. 이 과정에서 깊이 추정기가 필요하며, 이후에는 보정된 쿼리 이미지를 바탕으로 편집 프로세스를 진행합니다. 다중 뷰 편집을 통해 쿼리 이미지에 일관된 편집을 적용하고, 다수의 뷰에도 스케일링할 수 있도록 설계되어 있는 것이 이 방법의 특징입니다.

- **Performance Highlights**: 실험 결과 GeM-NR 방식은 다양한 편집 작업에서 일관성을 높일 수 있음을 보여주었습니다. 특히 기하학적 및 외관적 변화가 큰 편집을 수행하는 능력이 기존 방법보다 뛰어난 성능을 보였습니다. 정량적 및 정성적 성과 모두에서 우리의 방법이 최신 기술(state-of-the-art)과 비교하여 우수한 품질의 편집 결과를 생성하는 것을 나타내고 있습니다.



### Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agen (https://arxiv.org/abs/2606.05130)
- **What's New**: 이 논문은 도시 시뮬레이션, 교통 계획 및 정책 분석에 중요한 개인 수준의 이동 예측에 대한 새로운 접근 방식을 제안합니다. 제안된 방법론인 AgentMob은 기존의 수행 방식인 훈련 기반 모델의 한계를 극복하고, 훈련 없이도 강력한 성능을 보여주는 LLM 기반 에이전트 프레임워크를 제시합니다. 이 방법은 사용자의 다음 위치 예측을 적응적인 증거 통제(decision-making) 프로세스로 형식화하여 명확성을 높입니다.

- **Technical Details**: AgentMob은 LLM을 직접적인 예측자가 아닌, 증거의 양을 결정하는 컨트롤러로 활용합니다. 각각의 일상적인 사례에 대해 빠른 경로를 통해 해결하며, 모호한 사례는 최근의 이동 경로, 역사적 행동 및 지리적 정보를 바탕으로 반복적인 도구 사용을 촉발합니다. 이 구조는 LLM의 출력을 명시적으로 결합하여 예측 과정을 보다 투명하게 만듭니다.

- **Performance Highlights**: AgentMob은 세 가지 이동 데이터셋에서 훈련 없는 LLM 기반 방법 중 가장 우수한 성능을 나타냈습니다. BW 데이터셋에서 Acc@1 지표가 71.42%에 이르렀고, non-fast-path 케이스에서는 통계적 기준선보다 성능을 두 배 이상 개선하였습니다. 이는 주어진 증거를 토대로 불확실한 예측을 해결하는 데 큰 이점을 가지고 있음을 보여줍니다.



### Audio Interaction Mod (https://arxiv.org/abs/2606.05121)
Comments:
          Next generation of LALMs, work in progress

- **What's New**: 이번 연구는 오디오 상호작용 모델(Audio Interaction Model)을 제안하며, LALMs를 통합하여 실시간으로 음성 및 환경을 인식하고 반응하는 역동적인 온라인 모델로 발전시킵니다. Audio-Interaction이라는 통합 스트리밍 모델을 통해 오프라인 작업 수행과 온라인 일반 오디오 지침 이행을 동시에 가능하게 합니다. 이를 실현하기 위해 SoundFlow라는 프레임워크를 개발하여 데이터 생성부터 훈련 및 배포까지 모든 과정을 아우릅니다.

- **Technical Details**: Audio-Interaction 모델은 항상 작동하는 상호작용 알고리즘으로, 오디오를 청크 단위로 소비하며 각 청크에서 반응 여부를 결정합니다. SoundFlow 프레임워크는 세 가지 주요 구성요소로 이루어져 있으며, 먼저 계층적 이벤트 큐레이션을 통한 상호작용 데이터 합성을 제공합니다. 이러한 데이터는 고유한 TFJP(Time-Frequency Joint Preprocessing) 모듈을 통해 소음 억제 및 음향 신호의 경계를 매끄럽게 처리합니다.

- **Performance Highlights**: Audio-Interaction는 기존의 모델과 비교해 제한된 성능 기법을 고수하면서도 새로운 기능을 여는 데 성공하였습니다. 8개의 벤치마크에서 Audio-Interaction은 주요 오디오 작업에서 경쟁력 있는 성능을 유지하며, 실시간 ASR, 스트리밍 오디오 지침 따르기 및 선제적 도움을 비롯한 기능을 제공할 수 있음을 입증합니다. 특히 전체 음성을 포함하고 여러 턴에서의 성능 향상이 두드러지며, 모델의 변화 과정을 구체적으로 분석하였습니다.



### Continual Visual and Verbal Learning Through a Child's Egocentric Inpu (https://arxiv.org/abs/2606.05115)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 BabyCL이라는 연속(multimodal) 언어 학습 프레임워크를 소개합니다. 이 프레임워크는 SAYCam 데이터셋을 단일 시간 순서에 따라 처리하여 아동의 실제 경험에 근접한 방식으로 학습을 진행합니다. BabyCL은 시각적 표현 학습과 이미지-텍스트 대비(objective)를 결합하여 언어 학습을 효율적으로 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: BabyCL은 이벤트 세그먼트로 나누어진 연속적인 비디오 스트림에서 높은 차원의 시각적 표현 학습을 수행합니다. 각 세그먼트는 약 3분간의 텀을 두고 클러스터링 방법을 사용해 생성되며, 두 개의 재생 버퍼를 통해 시각적 및 다중 모달(history) 데이터를 독립적으로 관리합니다. 이 프레임워크는 공유된 백본(backbone)을 통해 동시에 세 가지 대조 손실(loss)을 최적화함으로써 비주얼과 언어 학습을 동시에 진행합니다.

- **Performance Highlights**: BabyCL은 SAYCam Labeled-S 4AFC 벤치마크에서 기존의 스트리밍 학습 기준과 비교해 개선된 성과를 보이며 오프라인 훈련의 상한선과의 격차를 크게 줄였습니다. 또한, 여러 실험 결과를 통해 온라인 시간 세그먼트 창의 길이나 재생 버퍼의 퇴거 규칙에 관계없이 얻은 성과가 견고하다는 것을 입증하였습니다. 이는 아동의 실제적인 경험에 더 가까운 훈련 조건 하에서도 의미 있는 단어-참조 매핑이 생성될 수 있다는 것을 보여줍니다.



### Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Hav (https://arxiv.org/abs/2606.05107)
- **What's New**: 이번 연구에서는 강력하지만 일반적인 비전 기초 모델을 전문적인 과학 분야에 적응시키기 위한 라벨이 없는 접근 방식을 제안합니다. 전통적인 감독된 파인 튜닝(supervised fine-tuning)은 라벨이 부족하고, 특정 작업을 위한 훈련이 모델의 일반성을 해치고 강건성을 감소시킬 수 있어 적합하지 않습니다. 대신 우리는 메타데이터(metadata)를 활용하여 새로운 도메인에 대한 표현을 자기 감독(self-supervised) 방식으로 조정합니다.

- **Technical Details**: 우리의 방법론인 FINO는 표준 자기 감독 목표와 유연한 메타데이터 안내를 결합하여 매우 세밀한 이산 메타데이터(discrete metadata)와 연속 메타데이터(continuous metadata) 모두를 처리합니다. 이 방법은 표현이 정보적 요소를 유지하도록 장려하면서 잡음 요소를 억제하도록 설계되었습니다. FINO는 세포 내 형광 현미경(subcellular fluorescence microscopy), 지구 관측(Earth observation), 야생 동물 모니터링(wildlife monitoring) 및 의료 이미징(medical imaging) 분야에서 전통적인 비지도 도메인 적응(unsupervised domain adaptation) 및 전적으로 감독된 적응(fully supervised adaptation)을 초월하는 성능을 보입니다.

- **Performance Highlights**: FINO는 작업 라벨(task labels) 없이 백본(adaptation) 적응을 수행하고, 슈퍼비전(supervision)을 위해 가벼운 프로브(probes)만을 사용하면서도 전문적인 도메인 특화(state of the art) 성능을 초과하는 결과를 보여주었습니다. 이는 다양한 과학적 도메인에서 FINO의 뛰어난 일반화 능력을 강조합니다. 이 연구는 어떻게 메타데이터를 활용하여 높은 성능을 유지하는지에 대한 중요한 통찰력을 제공합니다.



### Arithmetic Pedagogy for Language Models (https://arxiv.org/abs/2606.05106)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 연구는 인도네시아의 GASING 방식에 기반하여 언어 모델의 수학적 사고 방법을 훈련하는 데 사람이 사용하는 수학 교육 방법이 어떻게 적용될 수 있는지를 조사합니다. GASING 방법을 통해 기본적인 산술 문제를 해결하기 위한 절차적 접근 방식을 채택하고, 이러한 방법을 자연어 Chain-of-Thought (CoT) 감독과 결합합니다. 연구의 목표는 소규모 모델이 수학적으로 사고하는 능력을 어떻게 학습할 수 있는지를 탐구하며, 이는 인간의 수학 이해를 돕는 교육 방법과의 연계를 통해 이루어집니다.

- **Technical Details**: 연구에서는 GASING 방법을 사용하여 언어 모델의 훈련 및 추론 패턴을 수립합니다. 특정 수학적 관계를 내부적으로 형성하는 Transformer 모델의 훈련을 통해, 모델은 처녀 상태에서 초기 상태를 지나 최종 출력을 산출하는 일련의 과정을 통하여 발생하는 수학적 절차를 학습합니다. 모델 성능을 평가하기 위해, 동일한 문제를 푸는 다른 대규모 언어 모델과 비교하여 일반화 능력을 평가합니다.

- **Performance Highlights**: 훈련된 모델은 새로운 산술 문제에 대해 80% 이상의 정확도로 성과를 보이며, 더 큰 매개변수를 가진 모델과 비교하여 경쟁력 있는 성능을 달성했습니다. 특히, 이번 연구는 예측 목표 하에서 소규모 모델이 수학적 사고 패턴을 학습할 수 있음을 보여줍니다. 이러한 결과는 특정 교육적 방법이 적은 규모의 모델에서도 강력한 산술 능력을 지속적으로 발휘할 수 있다는 것을 입증합니다.



### Automatic Generation of Titles for Research Papers Using Language Models (https://arxiv.org/abs/2606.05085)
Comments:
          24 pages, 24 tables, 01 figure

- **What's New**: 이번 연구는 초록에서 연구 논문의 제목을 자동으로 생성하는 기술을 제안합니다. 특히, 사전 훈련된 언어 모델(PLMs)과 대형 언어 모델(LLMs)을 사용하여 제목 생성의 효과성을 평가합니다. 새로운 데이터셋 SpringerSSAT를 소개하고, 여러 모델의 성능을 비교하여 최적의 제목 생성 방법을 탐색합니다.

- **Technical Details**: 연구는 PLMs(예: T5, BART, PEGASUS)와 LLMs(예: LLaMA-3-8B, GPT-3.5-turbo)의 조합을 통해 이뤄지며, 모델 훈련은 CSPubSum, LREC-COLING-2024 및 새로 생성된 SpringerSSAT 데이터셋을 기반으로 합니다. ROUGE, METEOR와 같은 자동화된 평가지표를 사용하여 모델의 성능을 평가하며, ChatGPT의 창의적인 제목 생성 가능성도 탐구합니다.

- **Performance Highlights**: 실험 결과, PEGASUS-large 모델이 대부분의 지표에서 뛰어난 성능을 보인 것으로 나타났습니다. ChatGPT도 스타일 다양성을 인정받아 인간 작가의 제목과의 비교에서 긍정적인 평가를 받았습니다. 모든 훈련된 모델과 데이터셋은 Hugging Face에 공개되어 연구자들이 활용할 수 있도록 하고 있습니다.



### UniCAD: A Unified Benchmark and Universal Model for Multi-Modal Multi-Task CAD (https://arxiv.org/abs/2606.05058)
- **What's New**: 이번 논문은 CAD(Computer-Aided Design)의 다중 모드 및 다중 작업 학습을 위한 새로운 기준점인 UniCAD를 제안합니다. UniCAD는 다양한 입력 모드에서 CAD 재구성, CAD 생성 및 CAD 질문 응답 등의 작업을 포함하며, 이로 인해 다양한 CAD 작업을 통합적으로 평가할 수 있는 방법을 제공합니다. 또한, UniCAD-MLLM이라는 다중 모드 대형 언어 모델을 통해 텍스트, 이미지, 스케치 및 포인트 클라우드를 활용하여 CAD 작업을 통합적으로 수행합니다.

- **Technical Details**: UniCAD는 텍스트, 이미지, 스케치 및 포인트 클라우드의 다양한 입력 모드를 통합하여 CAD 모델을 생성하고 이해하는 데 필요한 평가 프로토콜과 작업별 메트릭스를 정의합니다. UniCAD-MLLM은 각기 다른 입력 모드를 처리하기 위해 모드별 인코더를 사용하고, 이를 공유된 기하학적 및 의미적 잠재 공간으로 투사하여 CAD 생성 및 이해 작업을 통합적으로 수행합니다. 특히, 최종 CAD 출력은 실행 가능하고 편집 가능한 Python 스크립트 형식으로 생성되어, 사용자가 쉽게 해석하고 수정할 수 있습니다.

- **Performance Highlights**: UniCAD-MLLM은 UniCAD 및 Fusion360 벤치마크에서 폭넓은 실험을 통해 기존의 작업별 및 다중 작업 기준선 모델보다 우수한 성능을 기록하였습니다. 대규모 다중 모드, 다중 작업 학습을 통해 특화된 시스템을 초월하는 성능을 보여줍니다. 이러한 결과는 통합 CAD 모델링의 효과성과 확장성을 검증하며, 향후 연구를 위한 데이터셋, 코드 및 사전 훈련 모델을 공개할 예정입니다.



### Self-Reflective APIs: Structure Beats Verbosity for AI Agent Recovery (https://arxiv.org/abs/2606.05037)
- **What's New**: 이 논문에서는 self-reflective API의 개념을 제안하고 이를 통해 AI 에이전트가 API 호출 시 발생하는 검증 오류를 처리하는 방식을 개선하고자 합니다. 기존의 API는 오류에 대한 일반적인 코드와 단순한 설명만 제공하는데, 이는 에이전트가 해결책을 스스로 추론하도록 남겨두는 문제를 가지고 있습니다. Self-reflective API는 에이전트가 직접 수행할 수 있는 체계적인 복구 지침을 제공하는 향상된 API 응답을 통해 이 문제를 해결하고자 합니다.

- **Technical Details**: Self-reflective API는 기계가 읽을 수 있는 구조화된 피드백을 API 계약의 일부로 취급합니다. 이는 검증 실패 시 복구 피드백 객체를 포함하는 응답 구조를 통해 이루어집니다. 이 API는 요청을 거부한 이유와 다음 LLM 호출이 성공하도록 하기 위한 파라미터 변경 사항을 명확히 전달하는 데 중점을 둡니다. 구조화된 피드백은 응답에 포함되어 있어 에이전트가 이를 별도로 요청하지 않고도 즉시 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, self-reflective API를 사용할 경우 Anthropic 모델에서 일반적인 오류 진단 대비 작업 완료율이 36.7~40.0pp 증가하며, 성공당 토큰 효율성도 1.8~2.2배 향상되었습니다. 그러나 gpt-4o-mini 모델에서는 이러한 성과가 보이지 않았으며, 이는 LLM의 부족한 문맥 이해 능력이 중요한 요소임을 시사합니다. 이를 통해 self-reflective API는 특정한 도메인 지식이 필요한 상황에서 특히 효과적임을 알 수 있습니다.



### Invariant Gradient Alignment for Robust Reasoning Distillation (https://arxiv.org/abs/2606.05025)
Comments:
          30 Pages

- **What's New**: 이 논문에서는 Invariant Gradient Alignment (IGA)라는 새로운 훈련 프레임워크를 제안합니다. 이 프레임워크는 훈련 데이터와는 다른 의미적 표면을 가진 OOD (out-of-distribution) 입력에 대한 모델의 성능 저하 문제를 해결합니다. IGA는 세 가지 혁신적인 방법을 통해 세멘틱적으로 다양한 예제 간의 그래디언트 업데이트를 정렬합니다.

- **Technical Details**: IGA의 구성 요소로는 (i) 논리 이성질체 집합 (Logical Isomer Sets) 생성, (ii) 고차원 그래디언트 변동을 억제하는 연속 그래디언트 충돌 마스크 (Continuous Gradient Conflict Mask), (iii) 마스킹된 그래디언트를 LoRA의 저차원 매니폴드로 재투사하는 절차가 포함됩니다. 이론적으로 IGA는 ERM보다 더 엄격한 OOD 일반화 한계를 제공합니다.

- **Performance Highlights**: 실험적으로 IGA는 네 가지 벤치마크에서 여덟 개의 기초 모델을 초과하여 ERM-SFT 대비 최대 14.3 포인트의 정확성 향상을 보였고, 논리적 일관성 점수(Logical Consistency Score)에서는 0.031을 기록하며 0.142에 비해 4배 향상된 결과를 나타냈습니다.



### DAR: Deontic Reasoning with Agentic Harnesses (https://arxiv.org/abs/2606.05009)
- **What's New**: 이 논문은 Deontic Agentic Reasoning (DAR)이라는 새로운 접근 방식을 소개합니다. DAR은 모델이 규정에 따라 요구에 맞게 법령을 동적으로 검색할 수 있도록 하는 에이전틱(Agentic) 추론 설정입니다. 기존의 추론 방식은 모델이 모든 규칙과 사례를 하나의 프롬프트에서 함께 제공받는 방식이지만, DAR은 규정 파일을 독립적으로 관리하여 보다 효율적인 추론을 가능하게 합니다.

- **Technical Details**: DAR에서는 법령이 프롬프트에 포함되지 않고, 별도의 파일(statute.txt)로 저장되어 요청할 때마다 필요한 정보를 조회할 수 있습니다. 모델은 사례와 질문을 받고, 도구 호출을 통해 법령의 관련 부분을 동적으로 읽어냅니다. 이러한 방식은 모델이 상태를 쌓아가며 규정을 효과적으로 활용할 수 있도록 돕습니다.

- **Performance Highlights**: DAR 평가를 위해 DeonticBench에서 다양한 모델을 테스트한 결과, 최상위(frontier) 모델은 DAR를 통해 성능 향상을 보였으나, 약한(open-source) 모델은 성능이 악화되기도 했습니다. 예를 들어, GPT-5.2는 SARA-Numeric에서 30%에서 60%로 개선된 반면, Qwen3.5는 34%에서 11%로 감소했습니다. 이러한 결과는 에이전틱 환경이 모델의 능력에 따라 다르게 작용함을 시사합니다.



### M$^3$Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks (https://arxiv.org/abs/2606.05008)
Comments:
          We present an evaluation designed for multi-modal memory in multi-modal models

- **What's New**: 본 논문은 다중 모달 모델의 기억 능력을 체계적으로 평가하기 위한 M$^3$Eval이라는 새로운 프레임워크를 소개합니다. 기존의 연구는 시각적 인식과 추론에 초점을 맞춰 기억 메커니즘을 명확히 측정하지 못했으며, 이로 인해 기억의 다양한 차원들이 잘 이해되지 않았습니다. M$^3$Eval은 인지 심리학의 원리를 기반으로 하여 특정 기억 메커니즘을 고립시키는 비디오 기반 질문-응답(task) 과제를 설계하였습니다.

- **Technical Details**: M$^3$Eval은 기억의 주요 차원을 네 가지로 구분합니다: (1) 동시 입력으로부터 정보를 유지하는 능력, (2) 유사한 내용의 방해에 대한 강인성, (3) 섞인 사건을 일관된 표현으로 통합하는 능력, (4) 비디오 세그먼트 간의 추상적 속성을 추적하는 능력입니다. 이 평가 프레임워크는 비디오 이해 과제를 통해 다양한 모델을 광범위하게 평가하며, 각각의 평가 방식은 명확한 질문과 특정 실패 모드를 수치화하는 메트릭으로 설계되어 있습니다.

- **Performance Highlights**: 결과적으로, 다중 모달 모델은 병렬 비디오 스트림을 처리할 때 독립적인 표현을 유지하지 못하며, 이는 주의 혼동으로 인한 것으로 추측됩니다. 모델의 기억 능력은 인간 보다 시간이 엇갈린 정보를 조직할 때 더 약하며, 복잡한 속성을 추상화할 때도 기호 기억(symbolic memory) 능력이 훨씬 떨어집니다. 이 연구는 다중 모달 모델의 기억 한계를 드러내고 향후 시스템 설계에 대한 새로운 통찰을 제공합니다.



### SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models (https://arxiv.org/abs/2606.05004)
Comments:
          accepted by ACL 2026 (main)

- **What's New**: 이 논문에서는 공공 대규모 언어 모델(LLM)의 사용자 프롬프트 프라이버시를 보호하기 위한 모델 독립적인 프레임워크인 SharedRequest를 제안합니다. 기존의 프라이버시 보호 방법들이 가지는 유용성과 효율성 유지의 한계를 극복할 수 있는 새로운 배치 레벨 접근 방식을 통해 사용자 정보를 숨기는 방법을 제시합니다. 이 방법은 각각의 프롬프트가 아닌 여러 프롬프트를 묶어서 처리하여 성능 저하를 최소화합니다.

- **Technical Details**: SharedRequest는 원래 프롬프트와 노이즈가 섞인 변형을 결합하여 민감한 정보를 모호화하는 메커니즘을 사용합니다. 이 과정에서 동의어 요청을 그룹화하여 배치 처리하고, 그에 따른 추론 비용을 전체 쿼리에서 분산시키는 방식입니다. 또한, 사용자의 익명성을 보장하는 경량 멀티파티 프로토콜을 설계해 LLM 아키텍처에 대한 수정 없이 작동합니다.

- **Performance Highlights**: 실험 결과, SharedRequest는 이전의 차별적 프라이버시 기준에 비해 20% 이상의 유용성을 달성하며, 비배치 추론에 비해 쿼리 비용을 최대 5배까지 줄이는 것으로 나타났습니다. 이러한 비율은 대규모 LLM 배치 처리에서 프라이버시 보호와 효율성을 동시에 만족시키는 새로운 접근 방식을 제시합니다.



### From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents (https://arxiv.org/abs/2606.04990)
- **What's New**: 이 논문은 대형 언어 모델(LLM) 기반의 에이전트가 복잡한 작업을 수행하기 위해 외부 도구 및 시스템과 상호작용하는 방식을 탐구합니다. LLM 에이전트는 멀티 스텝 과제를 해결하기 위해 다양한 외부 요소와 연결되며, 이러한 복잡성은 에이전트의 행동 검증과 디버깅을 더욱 어렵게 만듭니다. 따라서, 에이전트 운영의 맥락을 명확히 하기 위해 증거 추적(evidence tracing)과 실행 기원(execution provenance)의 필요성이 대두됩니다.

- **Technical Details**: 증거 추적은 에이전트의 주장을 지지하거나 반박하는 증거 단위를 식별하고 연결하는 메커니즘으로 정의됩니다. 실행 기원은 에이전트 실행의 전반적인 구조적 표현을 포함하며, 이는 검색된 문서, 도구 호출, 메모리 접근, 중간 주장 및 최종 출력을 아우릅니다.

- **Performance Highlights**: 논문은 또한 최근의 에이전트 연구에서 도구 사용의 안전성과 행동의 신뢰성 관련 문제점을 다루고 있으며, 에이전트 시스템의 신뢰성 확보를 위한 구조적 메커니즘이 필요함을 강조합니다. 이와 같은 접근은 LLM 에이전트가 신뢰할 수 있는 결정을 내리기 위해 필요한 기준들을 명확히 하는데 기여할 것입니다.



### DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving (https://arxiv.org/abs/2606.04987)
- **What's New**: 이번 논문에서는 여러 참가자들이 협력적으로 체스 퍼즐을 해결하는 대화 데이터셋인 DeliChess를 소개하고 있습니다. 이 데이터셋은 107개의 대화로 구성되어 있으며, 참가자들은 개별적으로 퍼즐을 해결한 후, 다자간 토론을 통해 집단 응답을 수정합니다. 기존의 데이터셋들은 구조화된 복합적 사고 과제에 중점을 두지 않았지만, DeliChess는 이 과제를 해결하기 위한 새로운 기회를 제공합니다.

- **Technical Details**: DeliChess 데이터셋은 3.4명의 참가자로 구성된 그룹이 평균 72회의 대화를 통해 세 가지 유형의 체스 퍼즐(포지셔널, 전투, 엔드게임)을 다룹니다. 참가자들은 개별적으로 퍼즐을 시도한 후,.live multi-party chat에서 토론을 하고 답변을 수정합니다. 이 데이터셋은 토론 전후의 선택과 퍼즐 난이도, 이동 품질에 대한 메타데이터를 포함하고 있습니다.

- **Performance Highlights**: 분석 결과, 토론을 통해 그룹의 퍼포먼스가 크게 향상되는 것으로 나타났습니다. 질문을 자극하는 발화(probing utterances)는 결과의 변동성을 증가시키지만, 반드시 모든 경우에서 성과를 개선하는 것은 아닙니다. 이 데이터셋은 단체 의사 결정 및 다자간 상호작용의 다이내믹스를 연구하는 데 중요한 자원이 될 것입니다.



### Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistanc (https://arxiv.org/abs/2606.04970)
Comments:
          53 pages, 14 figures

- **What's New**: 이번 논문에서는 실시간으로 단계별 절차적 작업을 지시하는 프로액티브 다중 모드 보조 시스템을 제안합니다. 이 시스템은 사용자에게 중단할 시점과 코칭 방법을 자율적으로 결정하며, 이를 실현하기 위해 새로운 대규모 Wearable-Egocentric 데이터셋인 EgoProactive를 출시합니다. 이 데이터셋은 명시적인 Out-of-Plan(OOP) 주석 및 회복 단계를 포함하고 있습니다.

- **Technical Details**: 제안된 시스템은 두 개의 모델 아키텍처로 구성되며, 하나는 장기 계획을 생성하고 업데이트하고, 다른 하나는 실시간 상호작용을 처리합니다. Duplex 상호작용 모델은 스트리밍된 egocentric 비디오를 소비하고 각 샘플링 시간에서 중단 여부를 결정하는 의사결정을 발행합니다. 이로 인해 단일 전진 패스에서 모든 작업을 수행하는 전통적인 방식에서 벗어나 모듈화된 구조로 기능을 분리합니다.

- **Performance Highlights**: 실험 결과, 훈련된 Llama-4 시스템은 강력한 상용 모델인 Claude Opus 4.6, Gemini 3.1 Pro, GPT 5.2에 비해 중재 품질을 크게 향상시켰습니다. OOP 회복에서 훈련된 모델이 높은 품질의 안내를 제공하며, 모든 6개 데이터셋에서 우수한 성과를 보여주었습니다. 또한, 계획 품질이 통제될 때, 훈련된 모델은 OOP 회복에 대한 큰 이익을 가져오는 것을 확인했습니다.



### From Prompt to Process: a Process Taxonomy and Comparative Assessment of Frameworks Supporting AI Software Development Agents (https://arxiv.org/abs/2606.04967)
- **What's New**: 이 논문은 AI 소프트웨어 개발을 지원하는 다양한 프레임워크를 탐구하며, 이전의 자동 완성 및 코드 생성 도구의 한계를 넘어서는 새로운 틀을 제시합니다. 이제 AI 도구는 단순한 보조 도구가 아닌 개발 프레임워크로 발전하였으며, 여기에서 프로세스, 역할, 아티팩트 및 검증의 개념이 포함됩니다. 저자들은 AI 프레임워크의 특성을 체계적으로 평가할 수 있는 6차원 프로세스 분류법을 개발하였고, 이를 6가지 선정된 프레임워크에 적용하였습니다.

- **Technical Details**: 연구 방법으로는 주로 구체적인 검색을 통해 AI 소프트웨어 개발 프레임워크에 대한 비교 연구를 수행하였습니다. 이 논문에서는 기존 문헌과 비공식 문헌을 흡수하며, 명확한 기준을 가지고 프레임워크를 평가하고 있습니다. 선정된 프레임워크는 프로세스 지원 기능과 사용자 중심의 설계를 기반으로 하며, GitHub의 트랙션을 기반으로 활동성과 인기도를 측정하여 최종 목록을 작성하였습니다.

- **Performance Highlights**: 그 결과, 선정된 프레임워크는 모두 6차원에서 완전하게 다루지 못하는 구조적 거래 관계가 있기 때문에, 프로세스의 깊이와 포터블리티(이식성) 사이에서 균형을 요구합니다. 반복적 위험 요소로는 사양과 코드 간의 괴리, 생성된 아티팩트에 대한 과도한 신뢰, 커뮤니티 확장의 취약성 등이 있습니다. 논문은 또한 실험적 평가와 지침을 위한 연구 의제를 제안하며, 중간 품질 기준, 맥락 거버넌스와 같은 영역에 집중하고 있습니다.



### AdaKoop: Efficient Modeling of Nonlinear Dynamics from Nonstationary Data Streams with Koopman Operator Regression (https://arxiv.org/abs/2606.04930)
Comments:
          Accepted by KDD'26

- **What's New**: 본 논문에서는 비정상 데이터 스트림에서 비선형 동역학을 모델링하기 위한 효율적인 스트리밍 알고리즘인 AdaKoop을 제안합니다. 이 알고리즘은 Koopman operator 이론에 기반하여 비선형 동역학을 선형 전환으로 표현하며, 이는 무한 차원 공간에서 작동합니다. AdaKoop은 원시 관측값과 재생 커널 힐베르트 공간(RKHS) 특징을 잠재 벡터의 방출로 취급하는 확률적 프레임워크를 사용합니다.

- **Technical Details**: 이 논문은 Koopman operator 이론을 기반으로 하는 확률적 정보 확장 상태 공간 모델을 소개합니다. 이 모델은 비선형 동역학을 관측값과 RKHS 특징으로부터 나오는 공통 잠재 선형 동역학 시스템으로 처리하여 수학적으로 다루기 쉽게 만듭니다. 이를 통해 AdaKoop은 반복적 비선형 최적화의 높은 계산 비용을 피하면서 비선형 동역학을 효율적으로 안정적으로 모델링할 수 있습니다.

- **Performance Highlights**: 71개의 다양한 도메인에서 실험한 결과, AdaKoop은 최신 방법들에 비해 실시간 예측 정확도 및 계산 효율성에서 우수한 성능을 보였습니다. 이 알고리즘은 통계적 가설 테스트를 통해 패턴의 급격한 변화 감지를 수행하고, 모델 파라미터를 점진적으로 업데이트하여 비정상 데이터 스트림을 능동적으로 처리합니다. AdaKoop의 설계는 반영구적인 데이터 스트림에서도 확장 가능성을 가지게 합니다.



### Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning (https://arxiv.org/abs/2606.04923)
Comments:
          23 pages, 7 figures

- **What's New**: 본 논문에서는 지침 기반 강화학습(Reward-based Reinforcement Learning)에서 보상 해킹(reward hacking)을 연구하는 새로운 환경인 CHERRL(Controllable Hacking Environment for Rubric-based RL)을 소개합니다. 기존의 LLM-as-a-Judge(LaaJ) 시스템은 심사 기준(bias)에 내재된 편향이 있고, 이에 따라 모델이 보상 신호를 잘못 최적화하는 경향이 있습니다. CHERRL은 이러한 편향을 명시적으로 주입하여 보상 해킹을 안정적으로 재현할 수 있도록 돕고, 이로 인해 연구자들이 보상 해킹의 발생을 정확하게 식별하고 분석할 수 있는 환경을 제공합니다.

- **Technical Details**: CHERRL은 LaaJ의 보상 신호를 청정한 gold 보상과 분리된 편향 보상으로 나누어 명시적으로 제어할 수 있게 설계되었습니다. 이중 심사자(dual-judge) 구조를 통해 CHERRL은 보상 해킹의 발달을 유도하고, Hacking의 정확한 발생 지점을 수치적으로 확인하는 방법을 제시합니다. 이 방법은 기계 학습(Machine Learning) 에이전트가 훈련 로그를 통해 보상 해킹의 신호를 분석하고 감지할 수 있도록 도와줍니다.

- **Performance Highlights**: CHERRL의 활용 가능성을 입증하기 위해, 우리는 다양한 심사자 편향이 해킹 경로에 미치는 영향을 분석했습니다. 또한, 훈련 로그에서 보상 해킹의 발생을 자동으로 탐지하기 위한 LLM 기반의 탐지 시스템인 Reward Hacking Detection Agent(RHDA)를 소개하였습니다. 이 시스템은 훈련 과정에서 해킹 발생 시점을 안정적으로 탐지할 수 있는 도구로, 향후 지침 기반 RL의 해킹 연구에 기여할 것입니다.



### Geometry-Aware Distillation for Prompt Tuning Biomedical Vision-Language Models (https://arxiv.org/abs/2606.04922)
Comments:
          Preprint. Code is available at this https URL

- **What's New**: 현재 시각-언어 모델(Vision-Language Models, VLMs)의 프롬프트 및 어댑터 기반 조정 방법은 의료 영상 분야에서 매력적입니다. 이는 임상 데이터의 민감성으로 인해 최적화된 백본 모델을 사용하고 제한된 주석을 사용해야 하기 때문입니다. 그러나 기존 방법들은 일반적으로 모든 비대상 클래스를 균등하게 잘못된 것으로 처리하여 임상적으로 중요한 클래스 관계를 무시하고, 한정된 감독 환경에서 불안정한 의사 결정을 초래합니다.

- **Technical Details**: 새로운 Omni-Geometry Knowledge Distillation (OGKD) 프레임워크는 교사 모델에 클래스 관계 구조를 주입하여 지향적인 목표를 생성합니다. 이러한 목표를 통해 두 가지 증류 손실을 개발하였으며, Global Geometry-Aware Distillation (GAD)은 글로벌 이미지 토큰에서 작동하고, Label-Guided Geometry Distillation (LGD)은 주목 패치 토큰에 동일한 구조를 적용합니다. 이는 섬세한 일치를 개선하는 데 도움을 줍니다.

- **Performance Highlights**: OGKD는 11개의 의료 데이터셋에서 포괄적인 실험을 통해 기존 최첨단 VLM 적응 방법에 비해 일관되게 1.7%-2.8%의 정확도 향상을 이끌어냈습니다. 또한, 여러 새로운 클래스로 일반화하는 데 강인성을 보이며, 다른 접근 방식보다 더 신뢰할 수 있는 예측 결과를 제공합니다. 이러한 개선 사항은 의학적 결정의 신뢰성과 정확성을 높이는 데 기여합니다.



### 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions (https://arxiv.org/abs/2606.04906)
- **What's New**: 이번 논문은 AI가 생성한 텍스트가 발생시키는 광범위한 사회적 위험성을 다루며, AI 생성 텍스트 탐지(AITD)에 대한 명확한 정의의 필요성을 강조합니다. 현재의 데이터셋과 접근법들이 각각의 고유한 기준을 정의하고 있지만, 이는 실제 적용 가능성과는 거리가 멀다는 점에 착안했습니다. 이 연구는 AI 생성 텍스트에 대한 다양한 개념과 그 특성을 체계적으로 정의하고, AITDNA라는 새로운 벤치마크 데이터셋을 구축했습니다.

- **Technical Details**: AITDNA는 99명의 인간 저자가 공동으로 작성한 350개 이상의 텍스트로 구성되어 있으며, 이 과정에서 발생하는 인간과 AI 간의 상호작용을 상세히 기록합니다. 이러한 데이터셋은 특정 환경에서 발생하는 인간-AI 공동 저작의 복잡성을 반영하며, 기존 문헌에 기반한 AITD 개념을 확장하여 내용 기반 및 저자 ID 기반의 새로운 개념을 추가했습니다. 이 연구는 다양한 AI 탐지기와 AITD 관련 데이터셋의 미세 조정을 통해 성능 평가의 일관성을 강화할 수 있는 방법을 제시합니다.

- **Performance Highlights**: AITDNA 데이터셋을 사용하여 기존 AITD 데이터셋의 숨겨진 가정과 인간-AI 공동 저작과의 정렬을 평가했습니다. 기존 AITD 데이터셋들은 자주 비현실적인 가정을 내포하고 있으며, 특정 개념의 탐지가 어렵다는 사실이 드러났습니다. 연구 결과, 각 개념별로 탐지기의 성능을 비교하고, 이러한 가정들이 성능 평가에 미치는 영향을 실증적으로 추정하여 AITD에 대한 더 정교한 접근법의 필요성을 제안합니다.



### Provably Auditable and Safe LLM Agents from Human-Authored Ontologies (https://arxiv.org/abs/2606.04903)
- **What's New**: 이번 논문에서는 선형 감사 가능성(linear auditability)이 필요한 비트리비얼 문제 도메인을 위해 LLM 에이전트 아키텍처인 Agentic Redux를 소개합니다. 우리는 적절한 도메인에서 실행할 때 Agentic Redux의 실행이 항상 의미론적으로 올바르다는 것을 증명합니다. 또한 의료 청구 규정 준수 및 보안 취약점 공개와 같은 두 가지 생산 준비 도메인을 제시합니다.

- **Technical Details**: Agentic Redux는 타입 이론(type theory) 기반의 설계로, 각 에이전트는 자신의 로컬 상태만 보고하며 글로벌 상태에 변경을 제안합니다. 메타 에이전트는 이러한 제안을 평가하여 글로벌 상태를 변경하는 유일한 주체로서, 모든 결정을 기록하는 감사 로그(audit log)를 유지합니다. 이는 시스템의 불변성을 보장하고, 잘못된 동작을 방지하는 데 기여합니다.

- **Performance Highlights**: 논문에서 제안한 에이전트 아키텍처는 규제 요구사항을 항상 준수할 수 있도록 최적화되어 있습니다. 모든 제안 및 결정은 감사 가능성이 확보된 로그에 기록되며, 이는 향후 외부 감사자가 시스템의 결정을 검토할 수 있게 합니다. Agentic Redux는 다양한 문제 도메인에 적용 가능성을 보여주며, 특히 금융 및 의료 문제 해결을 위한 효과적 방법을 제시하고 있습니다.



### DiverAge: Reliable Pluralistic Face Aging with Cross-Age Identity Relation Guidanc (https://arxiv.org/abs/2606.04881)
Comments:
          11 pages,10 figures, 5 tables

- **What's New**: 본 논문에서는 다층적 플루랄리스틱 얼굴 노화 프레임워크인 DiverAge를 제안합니다. DiverAge는 확산 오토인코딩(dispersion autoencoding) 기술에 기반하여 외형 수준의 다양성(appearance-level diversity)과 순서 수준의 신뢰성(sequence-level reliability)을 모두 유지합니다. 이 시스템은 각 연령군에서 다양한 후보를 생성할 뿐만 아니라, 서로 다른 연령대 간의 일관성을 보장합니다.

- **Technical Details**: DiverAge는 확산 디코딩(stochastic diffusion decoding) 기술을 통해 외형 수준의 다양성을 보존하고, 인퍼런스 타임 가이드인 Cross-age Identity Relation Regulator (CARR)를 사용하여 순서 수준의 신뢰성을 향상시킵니다. CARR는 실제 동일 신원의 교차 연령 쌍에서 추정한 Cross-age Identity Similarity (CIS) 우선 순위를 기반으로 하며, 이를 통해 세대 간의 과도한 정체성 이동을 억제합니다. 이 방식은 훈련 목표를 수정하지 않으며 추가적인 매개변수를 도입하지 않습니다.

- **Performance Highlights**: 실험 결과, DiverAge는 정체성 보존(identity preservation), 연령 정확도(age accuracy), 이미지 품질(image quality), 외형 수준의 다양성(appearance-level diversity)을 유지하면서도 순서 수준의 신뢰성을 개선하였습니다. 본 연구는 기존의 얼굴 노화 연구에서 간과되었던 순서적 정체성 유사성 패턴을 고려하여 신뢰성 있는 얼굴 노화의 생성 가능성을 높였습니다.



### Abduction Prover in Isabelle/HOL (https://arxiv.org/abs/2606.04877)
Comments:
          Accepted to Isabelle2026

- **What's New**: 이번 논문에서는 Isabelle/HOL을 위한 Abduction Prover를 소개합니다. 이 프로버는 복잡한 증명 목표에 대해 유용한 추측을 식별하여 증명 스크립트를 구축함으로써 증명 검색의 자동화를 향상시킵니다.

- **Technical Details**: Abduction Prover는 귀납적 추론(inductive reasoning)을 사용하여 증명 목표를 해결하는 방식으로, 기존의 증명 보조 도구들이 가지는 자동화 한계를 극복하고자 합니다. 이 방식은 특정한 논리적 구조를 기반으로 하여 보다 효과적인 증명 과정을 가능하게 합니다.

- **Performance Highlights**: 이 새로운 접근 방식은 기존의 증명 보조 도구들에 비해 더 높은 자동화 수준과 효율성을 제공합니다. 이를 통해 형식 검증(formal verification) 과정이 더 간소화되며, 사용자에게 더욱 편리한 도구를 제공합니다.



### Learning Empirically Admissible Neural Heuristics for Combinatorial Search (https://arxiv.org/abs/2606.04860)
Comments:
          13 pages, 3 figures, 2 tables, 1 algorithm

- **What's New**: 이 논문에서는 조합 최적화(combinatorial optimization) 및 경로 탐색(pathfinding) 문제 해결을 위한 새로운 방법론을 제안합니다. 특히, 기존의 심층 강화 학습(deep reinforcement learning) 방식이 가지는 한계를 극복하기 위해, 검증된(admissible) 신경 휴리스틱(neural heuristics) 함수를 학습하는 프레임워크를 개발하였습니다. 이 방법은 오버추정(overestimation)을 방지하는 비대칭 손실 함수(asymmetric loss function)를 활용하여 더욱 효과적인 경로 Optimality를 보장합니다.

- **Technical Details**: 제안된 방법론은 일반화 가능한 프레임워크로, Admissible Bellman Operator를 통해 하한을 정하고, 비대칭 손실 함수 및 검증 기반(calibration-based) 안전 오프셋을 적용하여 신경망의 예측을 조정합니다. 이를 통해 경량화된 검증이 가능해지고, 최적 경로가 현장에서 유지됩니다. 이 연구에서는 2x2 루빅스 큐브, 3x3 라이트 아웃 그리드 및 8-퍼즐을 플랫폼으로 사용하여 프레임워크의 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 신경 휴리스틱은 2x2 루빅스 큐브에서 검색 노드 확장을 83% 줄이고, 3x3 라이트 아웃 그리드에서 19.9%, 8-퍼즐에서는 1.9% 감소하는 성과를 보였습니다. 또한, 제안된 방법은 평가 프로토콜에서 관찰된 검증 위반(admissibility violations)이 없었으며, 경로의 최적성을 확보하였습니다. 이는 기존의 분석적 기준선에 비해 실질적인 성능 개선을 나타냅니다.



### Uncertainty-Aware End-to-End Co-Design of Neural Network Processors: From Training and Mapping to Fabrication (https://arxiv.org/abs/2606.04850)
Comments:
          14 pages

- **What's New**: 이 논문에서는 신경망 프로세서의 설계를 위한 통합 프레임워크를 제시합니다. 기존의 방법들이 특정 알고리즘에 tightly coupled 되어 있어, 개별 구성 요소를 개선하기 힘든 문제를 해결합니다. 이 프레임워크는 네트워크 훈련, 칩 매핑, 웨이퍼 수준 제작, 및 컴퓨트 리소스 할당을 아우르는 네 가지 상호 운용 가능한 설계 블록으로 구성되어 있습니다.

- **Technical Details**: 각 블록은 시스템의 나머지 부분과 기능-리소스 인터페이스만 노출하여, 다른 부분의 구조적 변화 없이도 개별 블록의 개선이 가능합니다. 특히, 불확실성(unity)의 처리를 중심으로 하며, 성공 확률의 역수인 Confidence를 비용, 시간 및 전력과 함께 최적화 가능한 리소스로 도입합니다. 이 접근 방식은 병렬 아키텍처와 다양한 설계 파라미터의 상호 작용을 더 잘 반영할 수 있게 해줍니다.

- **Performance Highlights**: 세 가지 사례 연구를 통해 접근 방식을 검증했습니다. 첫 번째 연구는 이종 응용 시나리오에서 Pareto-optimal(파레토 최적) 구현을 복원함을 보여주었습니다. 두 번째 연구는 Confidence가 사후 진단이 아니라 지속적으로 조정 가능한 설계 조정기 역할을 한다는 것을 확인했습니다. 마지막으로, 단일 블록의 구현을 개선하면 전역 Pareto 프론트에 자동으로 전파되며, 공동 설계 다이어그램을 수정할 필요가 없음을 입증했습니다.



### Signed Dual Attention: Capturing Signed Dependencies in Time Series Forecasting (https://arxiv.org/abs/2606.04833)
Comments:
          5 pages, 3 figures, accepted at AAAI 2026 AI4TS Workshop

- **What's New**: 이번 연구에서는 시간 연속성을 포함한 데이터에서의 긍정적 및 부정적 의존성을 모델링하는 새로운 방법, Signed Dual Attention을 제안합니다. 기존의 attention mechanism은 동질적(homophilic) 상호작용만을 가정하여 이러한 의존성을 다루는 데 한계가 있었습니다. Signed Dual Attention은 추가적인 파라미터 없이도 두 가지의 관계 패턴을 캡처할 수 있는 방법론입니다.

- **Technical Details**: Signed Dual Attention은 상관 구조(correlation structures)에 영감을 받은 이중 메시지 전송 메시지 패스(message-passing) 방식을 활용합니다. 이 모듈은 지지적(supportive) 및 대조적(contrastive) 정보를 하나의 공유 블록 내에서 전파합니다. 이를 통해 추가적인 파라미터 없이도 두개의 헤드 attention의 표현력을 효과적으로 달성할 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 기존 아키텍처에 쉽게 통합될 수 있으며, 서명된 관계 모델링이 필요한 특정 시나리오에서 성능 향상을 가져올 수 있습니다. 또한, 이 연구는 더 표현력이 뛰어나고 파라미터 효율적인 transformers로 나아갈 수 있는 길을 열어줍니다.



### OA-CutMix: Correcting the Label Bias of CutMix (https://arxiv.org/abs/2606.04820)
- **What's New**: 이번 논문에서는 CutMix의 레이블 할당 방식이 정당한 가정에 기반하지 않음이 밝혀졌습니다. 기존의 CutMix는 패치의 면적을 근거로 레이블을 할당하지만, 패치가 배경에 위치할 경우, 객체에 대한 가시적인 정보가 반영되지 않는 문제가 있습니다. 이를 해결하기 위해 Object-Aware CutMix (OA-CutMix)가 제안되었고, 이는 사전 계산된 분할 마스크를 사용하여 가시적인 객체 면적에 비례해 레이블을 재조정합니다.

- **Technical Details**: OA-CutMix는 CutMix의 이미지 믹싱 절차를 변경하지 않고, 레이블의 가중치만 수정합니다. SAM3을 활용하여 각 훈련 이미지에 대한 분할 마스크를 사전에 계산하고, 믹싱 시에는 각 이미지의 가시적 객체 픽셀 수를 계산하여 레이블을 비례적으로 할당합니다. 이를 통해, OA-CutMix는 동적 방법과 달리 훈련 중에 추가적인 비용이 발생하지 않고, 효율성을 유지합니다.

- **Performance Highlights**: OA-CutMix는 10개 이상의 정적 및 동적 믹싱 방법과 비교했을 때, 모든 작업에 걸쳐 일관되게 높은 정확도를 기록했습니다. 특히, 작은 객체에 대한 성능 개선이 두드러졌으며, 레이블 수정만으로도 기존의 이미지 믹싱 알고리즘을 수정한 방법의 성능을 초과할 수 있다는 것을 보여주었습니다.



### Learning While Acting: A Skill-Enhanced Test-Time Co-Evolution Framework for Online Lifelong Learning Agents (https://arxiv.org/abs/2606.04815)
- **What's New**: 이번 연구에서는 Lifelong Learning (평생 학습)을 위한 전이학습 프레임워크인 LifeSkill을 제안합니다. 이 프레임워크는 두 단계의 Reinforcement Learning (강화 학습) 접근 방식을 사용합니다. 이 시스템은 에이전트가 상호작용하는 환경 내에서 지속적으로 기술을 습득할 수 있도록 도와주며, 기존의 정적 매개변수 추론 방식에서 벗어나도록 설계되었습니다.

- **Technical Details**: LifeSkill의 첫 번째 단계는 Verifier-Guided Skill Learning (검증자 안내 기술 학습)을 통해 기술 추출 시 직접 지도 학습의 부족함을 해결합니다. 이 단계에서는 실패한 시도 후 기술 추출기가 후보 기술을 제안하며, 각 기술은 여러 기술 조건의 정책 롤아웃의 평균 검증자 보상에 의해 평가됩니다. 두 번째 단계인 Online Skill Internalization (온라인 기술 내재화)은 테스트 시간 상호작용 동안 정책 모델을 지속적으로 개선합니다.

- **Performance Highlights**: LifelongAgentBench에서의 실험 결과, LifeSkill은 기존의 강력한 기준선보다 평균 7점 향상된 성능을 보여주었습니다. 추가 분석을 통해 확인된 바에 따르면, 검증자 안내 기술 학습과 온라인 기술 내재화 모두 지속적인 평생 적응에 필수적인 요소로 작용함을 알 수 있었습니다.



### Scenario Generation for Risk-Aware Reinforcement Learning with Probably Approximately Safe Guarantees (https://arxiv.org/abs/2606.04812)
Comments:
          8 pages, preprint

- **What's New**: 본 연구에서는 강화 학습(RL) 에이전트의 실제 배치를 위해 안전 보장을 중요시합니다. 제안된 방법은 변리형 오토인코더(Variational Autoencoder, VAE)를 사용하여 상태 공간의 분포를 근사하고, 상한과 하한 경계 인증(barrier-certificates)을 통해 알려진 안전 행동과 비안전 행동을 구분하도록 설계되었습니다. 이 접근 방식은 정책의 상태 공간 내에서 안전한 영역을 최적화하는 쌍대 최적화 문제로 프레임하여, 안전 망을 제공하는 강력한 확률적 보장을 가능하게 합니다.

- **Technical Details**: 제안된 방법론은 샘플링된 정책 궤적을 사용하여 안전 제약 조건에서 발생하는 상한과 하한을 제공합니다. 이를 위해, 이 연구는 기존 안전 영역 내에서 날카로운 확률 보장을 제공하기 위해 비안전 영역에서 샘플 상태를 탐색합니다. VAE의 잠재적 특성을 사용하여 상태의 안전성 특성을 최적화하고, 경계 인증을 통해 안전 행동을 가진 정책의 강화에 기여합니다. 모델 프레임워크는 안전과 비안전 간의 경계를 설정하는 데 사용되며, 적절한 탐색을 통해 학습 프로세스에서 안전한 행동을 보장합니다.

- **Performance Highlights**: 이 연구는 Gymnasium 환경에서 실시된 실험을 통해 제안된 경계 인증 방법이 효과적으로 작동함을 보여주었습니다. 고급 안전 보장 전략을 도입하면서 발생하는 불확실성을 줄이고, 강화 학습의 에이전트가 탐색 중에 비안전 영역으로 들어갈 확률을 최소화하여 성능을 개선하는 점을 강조합니다. 기존의 방법들과 비교했을 때, 본 연구의 방법이 상한과 하한 경계의 수렴 속도가 더 빠르며, 안전 구역의 품질을 향상시키는 데 더 효과적임을 보였습니다.



### NoRA: Evaluating Grounded Reasonableness in Visual First-person Normative Action Reasoning (https://arxiv.org/abs/2606.04806)
- **What's New**: 본 논문에서는 AI와 agentic 시스템이 사회적 환경에서의 규범적 능력(normative competence)을 갖추어야 함을 강조합니다. 기존의 접근 방식들은 단순히 텍스트에서 규범적 판단을 평가하거나 제한된 후보 행동 중 선택하는 방식이었습니다. 그러나 이는 실질적으로 충분하지 않으며, 에이전트는 주어진 상황에서 적절한 행동을 스스로 식별해야 합니다. 이를 위해 NoRA라는 새로운 비주얼 벤치마크를 소개하고, 에이전트가 다음 행동을 생성하고 그 정당성을 평가하도록 요구합니다.

- **Technical Details**: NoRA는 각기 다른 1,420개의 주석이 달린 비디오 클립을 포함하며, 후보 행동을 생성하고 이를 사실-이유-행동 지원 그래프를 통해 정당화해야 합니다. 연구는 12개의 멀티모달 시스템을 순서대로 평가하며, 행동 정렬, 사실적 기반 및 지원 결합을 통해 평가합니다. 이러한 프레임워크는 AI 시스템의 규범적 결정력을 촉진시키기 위해 철학적 기준에 근거하고 있으며, 기존의 MCQ 형식에서 벗어나 행동의 적절성을 평가할 수 있게 합니다.

- **Performance Highlights**: 우리의 연구에서, 현재의 비전 언어 모델(VLM)은 자주 그럴듯한 다음 행동을 생성하고 관련 장면 사실을 회복하지만, 적절한 지역적 지원에 선택된 행동을 결합하는 데는 어려움을 겪고 있습니다. 구조화된 프롬프트를 사용할 경우, GPT-5.2는 GPT-5.4의 grounded reasonableness 점수의 68.6%에 도달합니다. Gemini-3-Flash는 Gemini-3.1-Pro 점수의 75.6%에 도달하여 NoRA가 실제적인 규범적 결정력을 개선하는 데 기여하고 있음을 보여줍니다.



### Activation Steering of Video Generation Models via Reduced-Order Linear Optimal Contro (https://arxiv.org/abs/2606.04775)
- **What's New**: 본 논문에서는 텍스트-비디오(T2V) 모델의 안전성을 개선하기 위해 새로운 방법을 제안합니다. 이전 방법론들은 모델의 가중치를 변경하거나 재훈련을 요구했지만, Latent Activation Linear-Quadratic Regulator (LA-LQR)는 더욱 최소 침습적으로 모델을 조정할 수 있는 체계적 기법을 제공합니다. 이 접근법은 T2V 추론을 동적 시스템으로 모델링하여 비디오 품질과 프롬프트 충실도를 유지하면서도 원하지 않는 결과를 줄이도록 설계되었습니다.

- **Technical Details**: LA-LQR는 T2V 모델의 활성화를 저차원 잠재 공간으로 투영하여, 불필요한 개입을 최소화하면서 목표 설정점으로 활성화를 유도하는 피드백 신호를 계산합니다. 이론적으로, LA-LQR는 잠재 벡터의 운동 방정식을 활용하여 시스템의 동적 특성을 설명하고, 개입의 정확성을 향상시키는 방법론적 토대를 제공합니다. 이러한 최적 제어 문제를 작업 관련 하위 공간에서 해결하여, 모델의 복잡성을 효과적으로 감소시킵니다.

- **Performance Highlights**: 제안된 LA-LQR은 개념 제어 및 비디오 안전 기준에서 기존 방법보다 안전 생성물의 비율을 감소시켰으며, 프롬프트에 대한 충실도와 비주얼 품질을 유지합니다. 실험 결과는 LA-LQR의 효과성이 기존의 T2V 조정 및 안전 기준을 초월함을 입증합니다. 이는 T2V 모델이 학습한 불안전한 개념을 효과적으로 조정할 수 있는 가능성을 제시합니다.



### Coarse-to-fine Hierarchical Architecture with Sequential Mamba for Brain Reconstruction (https://arxiv.org/abs/2606.04772)
- **What's New**: 이 연구에서는 CHASMBrain이라는 새로운 위계적 2단계 프레임워크를 제안하여 이미지에서 fMRI로의 인코딩을 수행합니다. 이 프레임워크는 Mamba 디자인을 활용하여 글로벌 의미 토큰과 로컬 공간 패치를 명확히 구분하여 처리합니다. 제안된 모델은 자연 장면 데이터셋에서 Pearson 상관관계 0.429 및 MSE 0.261을 달성하며, 기존의 여러 모델을 초월한 성능을 보여줍니다.

- **Technical Details**: CHASMBrain은 1단계에서 ROI 수준의 노이즈 제거된 활성화를 예측하고, 2단계에서 이러한 활성화를 보완하여 복잡한 복셀 수준 예측을 만듭니다. 이 과정에서 Mamba-VAE를 사용하여 세부적인 신호 회복을 지원합니다. 또한, 두 개의 스트림을 통해 글로벌 및 로컬 정보를 통합하여, 더 정교한 시각적 인식이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다수의 기준선 모델들보다 뛰어난 성능을 보였으며, 패치 스트림과 CLS 스트림의 비대칭 전문화가 인과 관계를 통해 구체화되었습니다. 교차 주제 전이 실험을 통해, 학습된 베이스 모델이 개인간에 일반화되는 방식도 확인되었으며, 이는 모형이 개인에게 독립적인 시각적 표현을 포착한다는 것을 암시합니다.



### Description-Code Inconsistency in Real-world MCP Servers: Measurement, Detection, and Security Implications (https://arxiv.org/abs/2606.04769)
Comments:
          Preprint

- **What's New**: 이 논문에서는 모델 컨텍스트 프로토콜(Model Context Protocol, MCP) 내에서 발생하는 설명-코드 불일치(Description-Code Inconsistency, DCI)의 중요성을 다룹니다. MCP는 대규모 언어 모델(Large Language Models, LLM)이 외부 도구를 사용할 수 있게 하는 표준으로 자리잡고 있습니다. 그러나 도구의 설명과 실제 코드 간의 불일치가 존재하여 오해를 일으킬 수 있으며, 이는 LLM의 잘못된 결정으로 이어질 수 있다는 점에 경각심을 줍니다.

- **Technical Details**: 연구진은 DCI의 문제를 정의하고 기능 불일치와 미보고된 부작용을 포함하는 포괄적인 분류체계를 제안합니다. 이를 기반으로, DCIChecker라는 자동화된 프레임워크를 구현하여 설명과 코드 간의 일관성을 검증합니다. DCIChecker는 LLM 기반의 의미론적 추론을 사용하여 설명의 자연어 표현과 코드 간의 불일치를 평가하며, 두 가지 방식의 프롬프트를 활용하여 결과를 비교합니다.

- **Performance Highlights**: 대규모 데이터 세트를 기반으로 진행된 실험에서 2,214개의 실제 MCP 서버에서 추출한 19,200개의 설명-코드 쌍 중 9.93%에서 DCI가 발생한 것으로 나타났습니다. 이러한 불일치는 특히 기능적 과장이 두드러지며, LLM이 도구를 잘못 선택하거나 의도하지 않은 시스템 동작을 유발하는 등 다양한 위험을 초래합니다. 논문은 DCI를 완화하기 위한 전략을 제안하여, 개발자와 사용자, 플랫폼의 역할에 따라 다양한 접근법을 제공합니다.



### Archi: Agentic Operations at the CMS Experimen (https://arxiv.org/abs/2606.04755)
- **What's New**: 아키(Archi)는 과학 협업을 위한 오픈소스(end-to-end) 프레임워크로, 이질적인 데이터 소스의 체계적인 수집과 조직을 결합합니다. 또한, 이를 통해 구성 가능한(private), 개인적인, 확장 가능한 에이전트를 배포하여 데이터를 검색하고 추론할 수 있습니다. 2026년 2월 이후로 CERN의 LHC 실험의 컴퓨팅 작업 팀에 배포되어 기술 운영자를 지원하는 에이전트로 기능하고 있습니다.

- **Technical Details**: Archi는 문서, 역사적 데이터 및 실시간 모니터링 시스템을 결합하여 검색 및 분석 기능을 제공합니다. 이 시스템은 운영자의 피드백과 실제 사용에서 수집된 질의 세트에 대해 평가되었습니다. 그 평가 기준은 인간 및 자동 패널에 의해 등급이 매겨졌습니다.

- **Performance Highlights**: 이 시스템은 운영 업무에 효과적이며 CMS 운영자가 제기한 실제 질문들을 해결하는 데 성공적입니다. 또한, 로컬에서 호스팅되는 오픈 웨이트 모델이 경쟁력 있게 작동하여, 민감한 데이터의 완전한 개인 관리를 가능하게 한다는 점이 관찰되었습니다.



### An Empirical Audit of Input Encoders for Multi-Channel Signal Transformers (https://arxiv.org/abs/2606.04752)
Comments:
          21 pages, 1 figure, 8 tables. Code: this https URL

- **What's New**: 이 논문에서는 Transformer 모델이 다채널 스칼라 신호를 처리하기 위해 다양한 입력 인코더를 비교합니다. 주요 방법으로는 공유 스칼라 투영, 채널별 선형 투영, 비선형 MLP, 채널 독립 및 채널-토큰 아키텍처 등이 포함되어 있습니다. 실험 결과, 표준 채널별 선형 투영이 다른 아키텍처와 비슷한 성능을 보여주며, 종합적으로 데이터에 따라 복잡한 구조를 사용할 필요성에 대해 실용적인 권고를 제공합니다.

- **Technical Details**: 연구에서 사용된 구조는 기본적으로 작은 인과적 Transformer 모델로, 입력 인코더는 8개의 서로 다른 버전으로 다양화됩니다. 각 인코더는 채널별로 값들을 어떻게 처리할지를 결정하는 다양한 방식을 채택하고 있으며, 이러한 방식들은 정보 손실을 최소화하도록 설계되었습니다. 특히, 채널 독립성과 같은 복잡한 설정은 성능에 어떤 영향을 미치는지를 분석합니다.

- **Performance Highlights**: 실험 결과, 채널별 선형 투영이 다른 입력 인코더와 비교했을 때 작은 차이를 제외하고는 비슷한 수렴 성능을 보여줍니다. 특정 아키텍처는 샘플 수가 많아질수록 성능 차이가 줄어드는 경향을 보이며, sinusoidal positional encoding을 과정을 통해 여전히 중요한 기여를 한다고 밝혔습니다. 마지막으로, 공유 스칼라 합산 방식은 정보 이론적으로 한계에 도달함을 확인할 수 있었습니다.



### TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration (https://arxiv.org/abs/2606.04743)
- **What's New**: 본 논문은 TIDE라는 새로운 프레임워크를 소개하며, 이는 사용자의 명시적 요청에 의해 드러나지 않은 숨겨진 문제를 발견하는 것을 목표로 합니다. TIDE는 두 가지 보완적 메커니즘으로 구성되어 있으며, 반복적인 발견(iterative discovery)과 사고 템플릿(thought templates)을 통해 효과적인 문제 해결을 지원합니다. 이 프레임워크는 개인 작업 공간과 소프트웨어 저장소 등 두 가지 현실적 설정에서 검증되었으며, 단일 예측(single-shot) 및 병렬 다중 에이전트(parallel multi-agent) 기준보다 현저한 성과를 보였습니다.

- **Technical Details**: TIDE 프레임워크는 반복적인 발견과 사고 템플릿을 결합하여 작동합니다. 반복적인 발견은 매 라운드마다 소규모 후보군을 제시하며, 이전에 발견한 내용을 기준으로 후속 라운드가 진행되어 문제 범위를 확대합니다. 사고 템플릿은 이전에 해결된 사례에서 추출된 재사용 가능한 형식(schema)으로, 문제 유형에 대한 구체적인 지침을 제공합니다.

- **Performance Highlights**: TIDE는 네 가지 LLM 모델 백본을 사용하여 개인 작업 공간과 소프트웨어 저장소에서 검증되었으며, 작업 범위, 문제 식별, 문제 해결에서 일관되게 더 나은 성과를 나타냈습니다. 이 결과들은 TIDE가 기존의 단일 요청 기반 접근 방식보다 더 효과적인 발견 과정을 통해 사용자에게 보다 유용한 지원을 제공할 수 있음을 시사합니다.



### Revisiting Vul-RAG: Reproducibility and Replicability of RAG-based Vulnerability Detection with Open-Weight Models (https://arxiv.org/abs/2606.04739)
Comments:
          Accepted at AI&CCPS 2026 workshop, co-located with the 21st International Conference on Availability, Reliability and Security (ARES 2026). This is the authors' preprint version

- **What's New**: 본 연구에서는 소프트웨어 취약성 탐지를 위한 Retrieval-Augmented Generation (RAG) 기반 프레임워크인 Vul-RAG의 재현성(reproducibility) 연구를 진행했습니다. 기존의 모델이나 API에 의존하지 않고, 오픈 가중치(open-weight) 모델과 로컬 추론(local inference) 환경에서 Vul-RAG의 결과를 재현하였습니다. 이를 통해 최신 모델들에 대한 성능 검증 및 상호 모델 재현 가능성(cross-model replicability)을 평가하며, 취약성 탐지 기법에서의 실용적 의미를 논의합니다.

- **Technical Details**: Vul-RAG는 소스 코드 취약성 탐지를 위해 LLM을 고급 취약성 지식으로 강화하는 방법론입니다. 이 프레임워크는 기존의 Common Vulnerabilities and Exposures (CVE) 데이터를 기반으로 취약성의 기능적 의미와 원인, 수정 방법 등을 다차원적으로 구성합니다. 취약성을 검출하는 과정은 세 가지 단계로 나뉘며, 각 단계에서 LLM이 절차적으로 지식을 활용하여 취약성을 판단하도록 유도합니다.

- **Performance Highlights**: Vul-RAG의 연구 결과는 로컬 환경에서도 재현 가능하다는 것을 보여주었으나, 성능은 대체로 0.30의 페어와이즈 정확도에서 정체되었습니다. 이 정체 현상은 최신 모델에서도 지속적으로 나타났으며, 이는 모델의 용량 향상이 반드시 성능 향상으로 이어지지 않음을 의미합니다. 연구 결과에 따른 실용적인 함의와 모델 선택과 배치에 대한 고려사항도 함께 논의되었습니다.



### Curvature-aware dynamic precision approach for physics-informed neural networks (https://arxiv.org/abs/2606.04736)
- **What's New**: 이 논문에서는 Physics-informed neural networks (PINNs)의 최적화 과정에서 수치적 정밀도(numerical precision)에 대한 민감성을 해결하기 위한 새로운 방안을 제안합니다. 저자들은 고정된 정밀도 대신 학습 중에 정밀도를 동적으로 조정하는 곡률 인식 정밀도 컨트롤러(curvature-aware precision controller)를 도입하여 계산 효율성을 높이고 예측 정확도를 유지하려고 합니다. 또한 이 방법은 제한된 메모리 BFGS (L-BFGS) 최적화로부터 유래된 곡률 정보를 재사용하여 적절한 정밀도를 지속적으로 유지합니다.

- **Technical Details**: 저자들은 PINNs에서 곡률 정보가 어떻게 수치적 민감도(numerical sensitivity)에 영향을 미치는지를 기반으로, 정밀도 조정의 단계를 동적으로 구분합니다. FP32가 충분한 경우에는 이를 유지하고, 훈련 동역학에 따라 수치적 민감도가 필요한 경우에는 FP64로 변환합니다. 이러한 과정은 기존의 고정된 정밀도 방식과는 다른 접근 방식으로, 동적 정밀도 조정이 PINNs 훈련에서 성능을 개선하는 데 어떤 역할을 하는지를 탐구합니다.

- **Performance Highlights**: 제안한 방법은 네 가지 PINN 실패 모드 벤치마크 및 특정 예제를 테스트한 결과, 모든 벤치마크 방정식에서 FP64의 전체 솔루션 정확도를 일관되게 달성하거나 약간 초과하면서도 훈련 시간을 줄일 수 있음을 보여줍니다. 이러한 결과는 PINN 최적화에서 정밀도 민감도가 단계에 따라 달라지며, 수치적으로 중요한 단계에서만 높은 정밀도를 선택적으로 적용할 때 계산 비용을 낮추면서도 예측 정확도를 유지할 수 있다는 것을 시사합니다.



### Trace-Mediated Peak Bias: Bridging Temporal Credit Assignment and Cognitive Heuristics in Deep Reinforcement Learning (https://arxiv.org/abs/2606.04735)
- **What's New**: 본 논문에서는 깊은 강화 학습에서 'Trace-Mediated Peak Bias' (TMPB)라는 체계적인 실패 모드를 밝혀냈습니다. TMPB는 에이전트가 현재 보상보다 높은 크기의 보상을 우선시하여 가치 추정에 오류를 발생시키는 현상입니다. 이는 인지 심리학의 피크-엔드 규칙(Peak-End Rule)과 유사한 인간의 기억 편향을 메커니즘적으로 설명합니다. 적응형 최적화가 이러한 문제를 완화할 수 있음을 보여줍니다.

- **Technical Details**: TMPB 현상은 비선형 함수 근사와 자격 추적(eligibility trace)이 결합되어 발생하는 불안정성을 통해 나타납니다. 본 연구에서는 마르코프 의사결정 과정(MDP)을 통해 두 가지 경로(τs​t​e​a​d​y와 τp​e​a​k)를 비교하여, 높은 강도의 보상이 보다 빈번하지만 낮은 보상보다 우선시되는 경향을 조사합니다. 이 과정에서 TD(λ) 알고리즘을 활용하면서 자격 추적 값을 세밀하게 조정하여 가치 추정 오류를 도출했습니다.

- **Performance Highlights**: 실험 결과, TMPB는 중간 자격 추적 깊이에서 가장 두드러지게 나타났으며, 이 경향이 비효율적인 경로를 선택하도록 이끕니다. 강화 학습 에이전트는 피크 경로에 대해 비합리적인 선호도를 보이며, 이는 인간의 경험 평가와 유사하게 나타났습니다. 이러한 결과는 적응형 최적화가 공평한 가치 추정을 위해 필수적임을 시사합니다.



### CoRe-MoE: Contrastive Reweighted Mixture of Experts for Multi-Terrain Humanoid Locomotion with Gait Adaptation (https://arxiv.org/abs/2606.04718)
Comments:
          Kailun Huang, Zikang Xie, Yanzhe Xie and Panpan Liao contributed equally to this work. Corresponding authors: Renjing Xu and Haohui Huang

- **What's New**: 이번 연구에서는 CoRe-MoE라는 새로운 두 단계 강화 학습 프레임워크를 제안합니다. 이 프레임워크는 보행과 달리기 사이의 매끄러운 전환을 달성하면서 자연스럽고 안정적인 이동을 유지할 수 있도록 설계되었습니다. Terrain-adaptive 전략을 포함하여 사용자는 복잡한 지형에서도 안정적인 보행을 유지할 수 있게 됩니다.

- **Technical Details**: CoRe-MoE는 격자 모델을 통해 보행 생성과 지형 적응을 분리하여 동작합니다. 첫 번째 단계에서는 안정적인 보행 정책을 학습하여 자연스럽고 부드러운 전환 능력을 확보합니다. 이후, 지형 인식 MoE 브랜치가 추가되고, 대비 학습(Objective)을 통해 지형 표현을 구성하여 전환 전문화를 지속적으로 촉진합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면 CoRe-MoE는 성공률, 이동 안정성, 다중 지형 적응성 측면에서 기존 방법들을 능가하는 것으로 나타났습니다. Unitree G1 로봇에서 제로샷 배치를 통해 복잡한 지형에서도 안정적인 보행 성능을 달성하며, 외부 간섭 아래에서도 정확한 발 위치와 동적 안정성을 유지하는 효과를 입증했습니다.



### VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training (https://arxiv.org/abs/2606.04708)
- **What's New**: 이번 연구에서는 Universal Manipulation Interface (UMI) 데이터와 대규모 Vision-Language-Action (VLA) 모델 훈련 간의 두 가지 주요 불일치를 식별하였습니다. 이를 해결하기 위해 VISTA라는 프레임워크를 제안하며, 이는 UMI 데이터를 시각적 및 물리적 요구 사항에 맞추어 조정합니다. VISTA는 UMI-VQA라는 대규모 비전-언어 데이터셋을 구축하고, 물리적 타당성을 확보하기 위해 체계적인 검증 파이프라인을 도입하였습니다.

- **Technical Details**: VISTA는 세 가지 상호 작용하는 구성 요소로 이루어져 있습니다. 첫째, UMI-VQA는 손목 장착 어안 카메라에 최적화된 최초의 대규모 VQA 데이터셋으로, 비주얼 지식과 언어 기반 질문-응답 쌍을 포함합니다. 둘째, 물리적 유효성을 보장하기 위해, 모든 궤적에 대해 데이터를 사전 검증하고 궤적의 연속성, 자기 충돌 위험 및 실행 신뢰도를 평가합니다. 셋째, 두 단계의 공동 훈련 과정을 통해 VLA 모델의 성능을 향상시킵니다.

- **Performance Highlights**: VISTA는 다양한 시뮬레이션 및 실제 조작 작업에서 기존 강력한 기준선인 π₀.₅, LingBot-VLA 및 Wall-X를 눈에 띄게 초월하는 성능을 보여주었습니다. 특히, UMI-VQA를 포함할 경우 정책 성능이 일관되게 향상되었으며, 물리적 검증 점수가 배치 성공을 강하게 예측하는 것으로 나타났습니다. 이를 통해 VISTA는 유망한 로봇 정책 학습을 위한 필수적인 요소로 자리 잡았습니다.



### Enhancing MedSAM with a Lightweight Box Predictor for Medical Image Segmentation (https://arxiv.org/abs/2606.04705)
- **What's New**: 이 논문은 MedSAM 아키텍처에 경량의 Box Predictor 모듈을 통합하여 강화된 세분화 프레임워크를 제안합니다. 단일 사용자 클릭으로부터 대략적인 경계 상자를 추정하여 포인트 프롬프트의 모호함을 줄이는 방식으로 작동합니다. 또한, Box Predictor는 별도의 훈련 과정을 거쳐 MedSAM에 통합되어 효율성을 유지하면서도 정확성을 높입니다.

- **Technical Details**: 제안된 프레임워크는 Box Predictor라는 경량 모듈을 포함하여 사용자가 제공하는 단일 포인트 프롬프트를 경계 상자로 변환합니다. 이 모듈은 MedSAM의 Prompt Encoder와 Mask Decoder를 향상하기 위한 공간적 선행정보를 제공합니다. 박스 예측기는 처음에 독립적으로 훈련된 후 MedSAM에 통합되어 훈련 효율성과 성능을 동시에 개선하도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 다양한 해부학적 구조와 이미징 도메인에 걸쳐 분할 정확도와 강건성을 향상시킵니다. FLARE22, BRISC, BUSI, LungSegDB의 네 가지 데이터세트에서 0.89(BUSI), 0.93(FLARE22), 0.88(BRISC), 0.98(LungSegDB)의 Dice 점수를 기록했습니다. 본 연구는 제한된 라벨 데이터 상황에서도 MedSAM의 강력한 성능을 유지하는 방법을 입증했습니다.



### Graph-Guided Universum Learning in Generalized Eigenvalue Proximal SVMs for Alzheimer's Disease Classification (https://arxiv.org/abs/2606.04699)
- **What's New**: 이 논문에서는 치매 예방을 위한 알츠하이머병(AD)의 조기 진단의 중요성을 강조하고 있습니다. 제안된 UG-GEPSVM 및 IUG-GEPSVM 모델은 기존의 Universum 샘플을 보다 효과적으로 활용하는 방법을 제시합니다. 특히, 이러한 모델은 MCI(경도 인지 장애) 샘플을 포함해 AD와 CN(인지 정상) 간의 미묘한 관계를 고려하여 성능을 극대화합니다.

- **Technical Details**: UG-GEPSVM과 IUG-GEPSVM는 각기 다른 방법으로 GEPSVM 모델을 개선합니다. UG-GEPSVM은 그래프 기반의 Laplacian 정규화를 일반화된 고유값 문제로 통합하여 AD와 CN 간의 구분을 수행합니다. 반면 IUG-GEPSVM은 안정적인 수치 해석을 위해 IGEPSVM 확장 버전을 기반으로 하여 문제를 단순화합니다.

- **Performance Highlights**: 실험 결과, 제안된 두 모델 모두 기존의 GEPSVM 및 Universum 기반 방법보다 일관되게 우수한 성능을 보여줍니다. 특히 UG-GEPSVM은 88.07%의 높은 평균 AUC를 기록하며, 잡음이 증가하는 조건에서도 안정적인 성능을 유지합니다. 이는 ADNI MRI 데이터 세트를 활용한 여러 실험을 통해 통계적으로 유의미한 결과로 입증되었습니다.



### Real-Time Automatic License Plate Recognition Using YOLOv8, SORT Tracking, and Temporal Data Interpolation (https://arxiv.org/abs/2606.04684)
Comments:
          7 Pages, For Accessing code:this https URL mobeen-pmo/Automatic-License-Plate-Recognition

- **What's New**: 본 연구는 도로 교통 감시 환경에서 자동 번호판 인식(Automatic License Plate Recognition, ALPR)의 실시간 처리 한계를 극복하기 위해 5단계의 end-to-end 알고리즘 파이프라인을 제안합니다. 이 방법은 깊이 학습 기반 객체 탐지와 운동학적 멀티 객체 트래킹, 기하학적 시간 데이터 보간을 원활하게 연결합니다. YOLOv8 nano 모델을 활용하여 차량을 로컬라이즈하고, SORT 알고리즘을 통해 프레임 간의 공간-시간적 링크를 구축합니다.

- **Technical Details**: 제안된 방법론에서는 원시 비디오 프레임을 처리하기 위한 다섯 단계의 파이프라인이 존재합니다. 첫 번째 단계에서는 노드 없는 YOLOv8 nano 아키텍처(3.2 million parameters)를 사용하여 물체의 중심을 직접 예측합니다. SORT 알고리즘은 운동학을 모델링하여 차량 탐지를 연결하고, 칼만 필터와 헝가리안 알고리즘을 통해 데이터 연관을 최적화합니다.

- **Performance Highlights**: 제안된 ALPR 프레임워크는 101.9%의 위치 정보의 연속성을 향상시키기 위한 시간적 경계 상자 보간 알고리즘을 통해 성능을 극대화합니다. EasyOCR을 사용하여 영국 번호판 형식에 대한 강력하고 구문 인식이 가능한 후처리 유닛이 개발되었습니다. 실험적 분석에서는 결과와 탐지 확인, 경계 상자의 기하학, 트래킹 분할 간의 복잡한 연관성을 정량적으로 분석하였습니다.



### Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models (https://arxiv.org/abs/2606.04672)
Comments:
          Accepted at ICML 2026

- **What's New**: 본 논문에서는 Continuous-Time Dynamic Graphs (CTDGs)을 위한 새로운 상태 공간 모델, CTDG-SSM을 제안합니다. 이 모델은 기존 방법들의 장기 시간 의존성과 공간 의존성을 동시에 해결하는 데 중점을 두고 있습니다. 특히, 새로운 메모리 기반 기법인 CTT-HiPPO를 도입하여 시공간 정보를 효과적으로 압축하면서도 학습합니다.

- **Technical Details**: CTDG-SSM은 동시적이고 세밀한 시간적 출력을 위해 시간 다항 기저를 이용한 메모리 압축을 통합합니다. 이 방법은 라플라시안(Laplacian) 행렬의 다항식으로 포물형 메모리 업데이트를 이루어냅니다. 제안된 모델은 파라미터 수가 상대적으로 적으면서도 효과적인 성능을 보여줍니다.

- **Performance Highlights**: CTDG-SSM은 동적 링크 예측(dynamic link prediction), 노드 분류(dynamic node classification), 순서 분류(sequence classification) 등 다양한 벤치마크에서 최첨단 성능을 달성했습니다. 특히, LRT와 LRS가 요구되는 데이터셋에서 significant 성능 향상을 이루어내며, AUC-ROC와 파라미터 수를 기준으로 하여 경쟁 방법들에 비해 우수한 성능을 입증합니다.



### Why Muon Outperforms Adam: A Curvature Perspectiv (https://arxiv.org/abs/2606.04662)
- **What's New**: Muon은 Adam에 비해 대규모 언어 모델 학습에서 약 두 배의 훈련 효율성을 개선하지만, 이러한 이점의 로컬 기하학적 출처는 여전히 불분명합니다. 본 연구는 Muon의 우수성을 Adam과의 비교에서 기울기 관점으로 설명하려고 하는 첫 번째 단계입니다. 특히, Muon은 손실 감소가 더 크며, 이는 주로 Muon이 더 낮은 두 번째 차원 곡률 비용(curvature cost)을 발생시키기 때문입니다.

- **Technical Details**: 본 연구에서는 2차 테일러 전개(second-order Taylor expansion)를 활용하여 Muon과 Adam의 손실 감소를 분석합니다. Muon은 작은 NDS(Normalized Directional Sharpness)를 통해 더 낮은 곡률 패널티(curvature penalty)를 발생시키며, 이는 업데이트 노드(update norm)와는 무관합니다. 훈련 데이터와 모델 구조가 Muon의 NDS 이점에 기여하는 방식을 조사하며, 데이터 불균형(data imbalance)이 Muon의 NDS 이점을 증가시키는 것을 보여줍니다.

- **Performance Highlights**: 연구 결과는 Muon이 GD에 비해 더 낮은 평균 NDS를 달성하며, 이는 고-곡률(high-curvature) 방향 간의 업데이트 에너지를 균형 있게 조정함으로써 달성됩니다. 데이터 세트의 불균형이 심할수록 Muon의 NDS 이점은 더욱 커지며, 훈련 초기와 중간, 후반에서는 주로 내부 레이어 곡률이 Muon의 작은 NDS를 유지하는 핵심 요인임을 발견했습니다. 이를 통해 Muon의 성능 우수성의 구체적 메커니즘을 밝혀냈습니다.



### Instance-Level Post Hoc Uncertainty Quantification in Object Detection (https://arxiv.org/abs/2606.04656)
Comments:
          7 pages, 2 figures

- **What's New**: 이번 논문에서는 자율주행의 안전성을 위한 객체 탐지에서의 불확실성 정량화 문제를 다룹니다. 기존의 모델을 변경하지 않고도 사후적으로 불확실성을 계량하는 방법을 제안하는데, 이는 Laplace approximation을 기반으로 하고 있습니다. 새로운 방법으로 Monte-Carlo generalized linearized model (MC-GLM)을 소개, 각 인스턴스의 불확실성을 효과적으로 정량화합니다.

- **Technical Details**: MC-GLM은 각 바운딩 박스에 대한 불확실성을 산출하며, Monte Carlo 샘플링을 통해 원래의 Jacobian 계산을 대체하는 방식으로 구현됩니다. 이 방법은 다차원에서 수치적으로 효율적인 계산을 가능하게 하며, 사후적으로 불확실성을 평가할 수 있도록 설계되었습니다. 이를 통해 인스턴스 수준의 정확한 불확실성을 제공할 수 있으며, 계산 자원 소모가 제한적입니다.

- **Performance Highlights**: nuScenes 데이터셋을 사용한 실험 결과, 제안된 MC-GLM 방법의 효과성을 검증하였습니다. 이 방법은 바운딩 박스에 대한 불확실성을 고품질로 나타내며, 자율주행 시스템에서의 실시간 사용에도 적합한 것으로 나타났습니다. 또한, 예측 분포의 신뢰성을 높이며, 기존의 방법보다 더 나은 성능을 보여주고 있습니다.



### QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples (https://arxiv.org/abs/2606.04646)
Comments:
          14 pages

- **What's New**: 이 논문에서는 데이터베이스 스타일 쿼리와 유사한 자연어 질문에 대한 응답을 평가하기 위한 진단 벤치마크인 QO-Bench를 소개합니다. 기존의 검색 보강 생성(Retrieval-Augmented Generation, RAG) 시스템은 주로 의미론적 관련성에 최적화되어 있었으나, 실제 쿼리 실행의 정확성을 보장하지 않는다는 문제점을 지적합니다. QO-Bench는 특정 이벤트의 쿼리 연산자를 보존하는 검색을 목표로 하며, 22,984개의 뉴스 기사와 614개의 기업 사건을 바탕으로 785개의 질문을 평가합니다.

- **Technical Details**: QO-Bench는 이벤트 튜플에 대한 설정에서 쿼리-연산자 질문 응답(QO-QA)을 재정립합니다. 각 gold answer는 이벤트 튜플에서 결정적으로 계산되며, 템플릿 특정 리콜에 따라 점수를 매깁니다. 이 디자인은 각 실패를 특정 연산자로 귀속시킬 수 있도록 하여, RAG, ReAct RAG, GraphRAG, 정보 추출-SQL 시스템을 비교 분석합니다. 또한, 두 축 프레임워크인 인덱스-시간 보존과 쿼리-시간 실행을 통해 각 패러다임의 실패 요인을 파악합니다.

- **Performance Highlights**: 실험 결과 각 패러다임은 특정 연산자에서 우월성을 보이지 않으며, 유사성 검색은 필터/프로젝트에서 강점을 보이는 반면, 정보 추출-SQL 시스템은 교차 사건 조인에 약점을 보입니다. QO-Bench는 연산자 실행이 핵심 병목임을 시사하며, 이는 단순한 검색 대답 생성만으로는 개선되지 않습니다. 또한, Gold evidence를 이용한 긴 컨텍스트 오라클조차도 한계에 도달하지 못하였고, 이는 연산자 실행의 중요성을 부각시킵니다.



### QuBLAST: A Framework for Quantizing Large Language Models with Block-Level Compression Approach and Activation Scaling Strategy (https://arxiv.org/abs/2606.04620)
Comments:
          10 pages, 9 figures, 5 tables

- **What's New**: QuBLAST는 대형 언어 모델(LLM)의 포스트 훈련 양자화를 위한 새로운 방법론으로, 블록 수준의 압축 접근 방식을 활용하고 활성화 스케일링 전략을 도입합니다. 기존의 방법들과 달리 QuBLAST는 모델의 주의 블록별로 서로 다른 양자화 수준을 적용하여 메모리 효율성을 극대화합니다. 이를 통해 LLM의 성능 저하를 최소화하면서도 모델 크기를 40%-45.2% 줄일 수 있다는 점이 특징입니다.

- **Technical Details**: QuBLAST의 핵심 단계는 세 가지로 구성됩니다: 네트워크 모델 분석, 블록 수준 압축, 양자화 설정 선택입니다. 먼저 pretrained LLM의 구조를 조사하여 각 주의 블록의 민감도 분석을 통해 적절한 양자화 접근 방식을 결정합니다. 블록 수준의 압축 단계에서는 클로스 엔트로피 손실 분석을 통해 각 주의 블록에 대한 양자화 설정을 확정하며, 활성화 스케일링을 통해 활성화 값의 범위를 조정하여 양자화의 부정적인 영향을 완화합니다.

- **Performance Highlights**: QuBLAST를 적용한 실험 결과, 다양한 모델 아키텍처에서 LLM의 메모리 사용량을 40%-45.2% 감소시키면서 WikiText-2 및 WikiText-103 데이터셋에서 성능이 5% 이내로 유지됨을 보였습니다. 특히 활성화 양자화를 활용한 경우, 메모리 절약이 42.4%-48.20%에 달하며 성능 저하는 2% 이내로 유지됩니다. 이러한 결과는 QuBLAST가 효율적인 양자화 방식임을 입증합니다.



### Ekka: Automated Diagnosis of Silent Errors in LLM Inferenc (https://arxiv.org/abs/2606.04594)
Comments:
          ICML 2026

- **What's New**: 이 논문에서는 LLM (Large Language Model) 서빙 프레임워크에서 발생하는 'silent error'를 자동으로 진단할 수 있는 Ekka라는 시스템을 제안합니다. 기존의 수동 진단 프로세스의 비효율성을 해결하기 위해, Ekka는 참조 구현(reference implementation)과의 차별적 디버깅을 통해 문제의 근본 원인을 파악합니다. 이 시스템은 실제 LLM 서빙 프레임워크에서 발생한 90개의 silent error 데이터를 이용하여 효과성을 입증하며, 기존 방법에 비해 평균 진단 정확도를 24%에서 34% 향상시켰습니다.

- **Technical Details**: Ekka는 첫 번째 단계에서 코드베이스 및 모델 아키텍처를 분석하고 실행 추적을 수집합니다. 이후 에이전트를 기반으로 한 버그 진단 단계에서 구성 요소 매핑(component mapping), 활성화 정렬(activation alignment), 그리고 에러 분석(error analysis) 등의 과정을 통해 silent error를 정확히 식별합니다. 이 시스템은 다양한 내부 구성 요소 및 메모리 레이아웃을 가진 서빙 프레임워크 간의 중간 실행 상태를 정렬하여 비교합니다. Ekka는 최종적으로 오류 비율에 관한 change-point 분석을 수행하여 문제의 근본 원인을 pinpoint합니다.

- **Performance Highlights**: Ekka는 벤치마크 테스트를 통해, 17개의 LLM에서 발생한 문제를 효과적으로 진단하였으며, 평균 진단 비용은 약 $30로 매우 경제적입니다. 진단 정확도는 80%의 pass@1 및 88%의 pass@5로, 최신 시스템들보다 우수한 성과를 달성하였습니다. 또한 Ekka는 새로운 silent errors를 진단하였으며, 이들 역시 개발자들에 의해 확인되었습니다.



### Synthetic Personalities: How Well Can LLMs Mimic Individual Respondents Using Socio-Economic Microdata? (https://arxiv.org/abs/2606.04592)
- **What's New**: LLM 기반의 디지털 트윈은 시장 조사에서 스케일과 속도를 혁신적으로 변화시킬 잠재력을 가지고 있으나, 기존의 트윈은 제한된 인구 통계 질문에 의존하거나 목적성 설문 및 인터뷰에서 수집된 데이터에 기반한 것들이 많습니다. 본 연구는 기업들이 보유한 비균질 패널 데이터로부터 개인별 트윈을 구축하는 방법을 제안하고 있습니다. 이러한 접근법은 연구에서 수집한 데이터가 아닌 기업의 기존 데이터를 활용하여 상세한 개별 트윈을 설계하였습니다.

- **Technical Details**: 본 연구는 독일의 사회경제 패널(SOEP) 데이터를 기반으로 개인 수준의 디지털 트윈을 구축하였으며, 이는 28,000명 이상의 응답자를 인터뷰한 41년간의 종단적 패널입니다. 연구는 3×5×2×2 방식의 구축 방법 그리드에서 평가되었으며, 총 210만 개 트윈 응답을 사용하여 정확도와 순위 상관관계를 измер하였습니다. 매개 모델의 구조 및 정보 깊이에 따라 트윈의 질이 어떻게 변화하는지를 체계적으로 분석하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 정보 깊이가 증가함에 따라 트윈의 정확도와 순위 상관관계가 증가하지만, 75% 엔트로피에서 비용 효율적인 최대치를 이룹니다. 최적의 매개 방식으로의 전환이 모든 모델에서 정확도를 향상시켰으며, 최고 정확도는 78.8%에 달했습니다. 이 연구는 데이터 설계가 아닌 항목 볼륨과 모델 선택이 트윈 기반 시장 조사의 주요 제한 요소임을 제시합니다.



### Multi-SPIN: Multi-Access Speculative Inference for Cooperative Token Generation at the Edg (https://arxiv.org/abs/2606.04581)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 효율을 높이기 위해 Speculative Inference (SPIN)을 다중 사용자 에지 시스템에서의 협력적 토큰 생성이 가능하도록 분산 배포하는 Multi-access SPIN (Multi-SPIN) 아키텍처를 제안합니다. 이를 통해 자원 제약이 있는 장치와 서버 간의 계산 부하를 효과적으로 분산시키는 이점이 있습니다. 기존의 SPIN에서 발전된 이 새로운 프레임워크는 여러 사용자들이 사용 가능한 드래프트 길이를 최적화하고, 업링크 대역폭을 조정하여 총 토큰 생산량을 극대화하는 문제를 해결합니다.

- **Technical Details**: Multi-SPIN은 각 사용자 장치에서 소규모 언어 모델(Small Language Models, SLMs)을 활용하여 후보 토큰 시퀀스를 생성하고, 이를 서버에 업로드하여 동시에 검증합니다. 이 과정에서 드래프트 길이는 사용자 노드의 계산 부화 및 다중 접근 지연을 조절하는 중요한 변수로 사용됩니다. 논문에서는 두 가지 경우, 즉 사용자가 균일한 드래프트 길이를 사용할 경우와 이질적인 드래프트 길이를 사용할 경우에 대해 최적의 대역폭 할당 및 드래프트 길이 제어 방안을 제시합니다.

- **Performance Highlights**: Multi-SPIN은 Llama-2와 Qwen3.5 모델을 사용한 다양한 실험을 통해 이질성을 무시한 경우에 비해 최대 88%의 생산량 개선을 보여줍니다. 특히, 드래프트 길이의 최적화는 각 사용자 노드의 수용 능력에 따라 조정되어 총 토큰 생산량을 극대화하며, 서버의 검증 효율성을 더욱 향상시킵니다. 이러한 성과는 Multi-SPIN이 자원 제약이 있는 에지 서버 환경에서 LLM을 효과적으로 사용할 수 있도록 돕는다는 점에서 큰 의미가 있습니다.



### Rollout-Level Advantage-Prioritized Experience Replay for GRPO (https://arxiv.org/abs/2606.04560)
- **What's New**: 이 논문은 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 위한 새로운 롤아웃 수준의 리플레이 버퍼를 제안합니다. 이 방식은 과거의 롤아웃을 전부 저장하고 샘플링하는 대신 개별 롤아웃을 저장하고 샘플링합니다. 또한 최대 τmax 단계보다 오래된 롤아웃을 제거하여 낡음(staleness)을 관리합니다.

- **Technical Details**: 제안된 리플레이 버퍼는 나이 퇴출(age eviction)과 신선하게 고정된 조합(fresh-anchored composition)을 통해 정책 지연(policy lag)을 제어합니다. 각 롤아웃의 나이는 생성된 이후의 단계 수로 정의되며, 특정 기준을 초과하는 롤아웃은 제거됩니다. 이 접근 방식은 훈련 롤아웃을 유지하며, 이를 통해 성능 향상을 이루는 방법을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 5개의 수학 벤치마크에서 GRPO 및 단순 리플레이 기법보다 뛰어난 성능을 보입니다. 가장 큰 모델인 4B에서 5개 벤치마크 평균에서 +4.35 pp의 개선을 기록했습니다. AES 메트릭에서 GRPO보다 효율성 차이가 가장 컸으며, 이는 +0.579로 나타났습니다.



### Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents (https://arxiv.org/abs/2606.04555)
- **What's New**: 이번 연구에서는 'Segment Tree Memory'(SegTreeMem)라는 새로운 기억 아키텍처를 제안합니다. SegTreeMem은 대화의 역사(history)를 주어진 시간순으로 분류하여 관리하며, 실시간으로 새로운 발언(utterance)을 삽입할 수 있도록 설계되었습니다. 기존의 기억 시스템이 주제적 유사성에 중점을 두는 것과 달리, 이 시스템은 발언의 시간 순서를 유지합니다.

- **Technical Details**: SegTreeMem은 발언을 메모리 구조에 통합하기 위해 오른쪽 단말 노드(rightmost frontier)에서 소규모의 노드를 업데이트합니다. 이 구조는 시간적 순서를 보존하고 계층적인 기억 세그먼트를 형성할 수 있도록 하며, 정보 검색(retrieval) 시에도 계층적 시간적 문맥(hierarchical temporal context)을 활용합니다. 이 메모리 아키텍처는 세 개의 장기 기억 벤치마크 및 두 개의 대형 언어 모델(LLM) 백본을 통해 검증되었습니다.

- **Performance Highlights**: SegTreeMem은 기존의 평면적 검색(flat retrieval), 그래프 구조 메모리(graph-structured memory), 트리 구조 메모리(tree-structured memory) 벤치마크에서 응답 품질을 향상시키는 성과를 보여주었습니다. 추가적인 시간적 순서(permutation analysis) 분석에서는 기억의 구조를 설정할 때 시간적인 순서가 성능 향상에 결정적인 요소임을 지적하였습니다. 대화형 에이전트의 효과를 증진시키기 위해 매우 중요한 기여를 하는 메모리 구조로 평가받고 있습니다.



### Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models (https://arxiv.org/abs/2606.04535)
Comments:
          Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)

- **What's New**: 이번 논문에서는 Dynamic Infilling Anchors (DIA)라는 혁신적인 방법을 제안합니다. 이 방법은 생성 길이를 조정하는데 도움을 주며, 고정된 앵커(anchor) 사용의 비효율성을 극복합니다. DIA는 훈련 없이 동적으로 앵커 포지션을 추정하여 형식 제약이 있는 작업에서의 품질과 신뢰성을 높입니다.

- **Technical Details**: DIA는 두 단계로 구성됩니다. 첫 번째 단계에서는 끝 앵커의 포지션을 추정하여 생성 길이를 조정합니다. 두 번째 단계에서는 고정된 앵커 간의 콘텐츠를 반복적으로 생성하는 과정을 통해 구조적 일관성을 확보합니다. 이 접근법은 모델이 충분한 생성 공간을 확보하도록 하여 중복 방지 및 불필요한 컴퓨팅을 최소화합니다.

- **Performance Highlights**: 실험 결과, DIA는 GSM8K 및 MATH 데이터셋에서 0-샷(0-shot) 접근 방식으로 형식 정확도를 각각 58.83%에서 72.63%, 29.10%에서 76.82%로 크게 향상시켰습니다. 답변 정답률도 GSM8K에서 14.86%에서 46.78%로 개선되었습니다. 이러한 성과로 DIA는 신뢰성 있고 구조적으로 인식할 수 있는 생성으로 나아가는 확고한 경로가 됩니다.



### Optical-Guided Neural Collapse for SAR Few-Shot Class Incremental Learning (https://arxiv.org/abs/2606.04528)
Comments:
          16 pages, 6 figures

- **What's New**: 본 논문은 합성 개구 레이더(SAR) 이미징에서의 Few-shot class-incremental learning (FSCIL) 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 광학 ATR 데이터 세트를 활용하여 SAR 특징 학습을 위한 기하학적 우선 정보를 제공하는 방식으로, 강한 방위 각도 민감도를 극복하고자 합니다. 특히, 이 연구는 데이터 부족과 SAR의 변동성 문제를 해결하기 위해 렌즈의 도움을 받습니다.

- **Technical Details**: 제안된 방법에서는 데이터가 풍부한 광학 ATR 데이터로부터 직교(feature subspaces) 특징 서브 공간을 파생시키고, 이를 SAR 특징 학습에 활용합니다. 두 개의 손실 함수, 즉 프로젝션 손실과 분류자 손실을 조정하여, 주요 각도 제약을 통해 SAR 특징을 직교 서브 공간에 투영합니다. 이러한 접근 방식은 클래스 평균에 집중하고, 클래스 간 각도를 유지하면서 신경 붕괴(neural collapse)를 유도합니다.

- **Performance Highlights**: 실험 결과, 본 연구는 24개 대상 클래스를 포함하는 벤치마크 데이터 세트에서 최고의 최종 정확도를 달성하며, 기존의 FSCIL 방법들보다 성능 저하가 적은 우수한 성능을 보였습니다. 또한, 신경 붕괴 지표는 클래스 간 분리성과 클래스 내 밀집도를 개선하여, 학습된 특징이 이상적인 단순엣지-ETF 기하학에 더 가깝게 근사함을 보여줍니다.



### ANN Search: Recall What Matters (https://arxiv.org/abs/2606.04522)
- **What's New**: 이번 논문에서는 Approximate Nearest Neighbor (ANN) 검색의 질을 평가하는 새로운 메트릭인 1/Ratio@k를 탐구합니다. 기존의 Recall@k는 정확하지 않은 기준으로 작업의 난이도를 과장하고 있으며, 이로 인해 불필요한 계산 비용이 증가하고 있습니다. 반면, 1/Ratio@k는 검색된 결과의 질을 더 정확히 반영하며, 컴퓨팅 자원 소모가 적고 적용이 용이합니다.

- **Technical Details**: ANN 검색은 정보 검색, 추천 시스템, 검색 강화 생성(retrieval-augmented generation) 등 다양한 분야에서 핵심적으로 사용됩니다. 기존의 Recall@k 기준 대신, 1/Ratio@k는 검색된 이웃과 진짜 이웃의 거리 차이를 측정하여 더 효과적인 품질 측정을 제공합니다. 두 메트릭 모두 0과 1 사이의 값을 가지지만, Recall@k는 식별자 일치를 세는 반면, 1/Ratio@k는 검색된 결과의 질을 측정합니다.

- **Performance Highlights**: 논문에서 제안하는 1/Ratio@k는 다양한 실험에서 Recall@k에 비해 더 효율적인 성능을 보였습니다. 여러 데이터셋에 대한 벤치마킹 결과, 1/Ratio@k 기준으로 작업 품질 기준에 도달하는 것이 훨씬 쉬운 것으로 나타났습니다. 또한, 구조적 성능 측면에서 Recall@k가 감소하더라도 1/Ratio@k는 안정적인 성능 지표를 유지하는 것을 확인하였습니다.



### Treat Traffic Like Trees: A Semantic-Preserving Hierarchical Graph-Based Expert Framework for Encrypted Traffic Analysis (https://arxiv.org/abs/2606.04517)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 논문에서는 데이터 흐름과 관련된 복잡한 패킷 구조를 효과적으로 분석하기 위해 Protocol Tree Graph Attention with Mixture of Experts (PTGAMoE)라는 새로운 암호화된 트래픽 분석 프레임워크를 제안합니다. PTGAMoE는 프로토콜의 계층적 특성을 보존하며, 각 필드에 대한 전문가(Experts) 네트워크를 구성하여 고유한 데이터 구조에 적합한 분석을 가능하게 합니다. 또한, 엄격한 데이터 유출 방지 설정에서 뛰어난 성능을 보여주며, 모델 해석 가능성을 높이는 중요한 인사이트를 제공합니다.

- **Technical Details**: PTGAMoE는 패킷 필드를 실세계 인캡슐레이션 시맨틱스를 반영하는 프로토콜 트리 그래프로 표현하여, 프로토콜 필드 간의 계층적 종속성을 명시적으로 모델링합니다. 이 구조는 전통적인 고정 길이 벡터로의 변환 없이 프로토콜의 본질적인 의미를 유지하도록 설계되었습니다. 각 레이어에 특화된 그래프 주의 전문가를 사용하여 다양한 프로토콜 특성을 포착하고, MoE 융합 모듈이 다층 표현을 선택적으로 결합합니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해, PTGAMoE는 최신 상태 모델(SOTA)인 ET-BERT, YaTC, RBLJAN보다 훨씬 뛰어난 성능을 보였습니다. 특히, PTGAMoE는 패킷 수준의 표현을 통합하여 흐름 설명자로 변환하는 강력한 집계 메커니즘을 포함하고 있으며, 모델의 결정 과정을 직관적으로 이해할 수 있게 해줍니다. 제안된 NGI 및 GCR 메트릭스를 통한 해석 가능성 역시 모델의 신뢰성을 높여줍니다.



### GeoMin: Data-Efficient Semi-Supervised RLVR via Geometric Distribution Modeling (https://arxiv.org/abs/2606.04516)
- **What's New**: 이번 논문에서는 RLVR(Reinforcement Learning with Verifiable Rewards)과 관련하여 새로운 GeoMin 프레임워크를 제안합니다. GeoMin은 전체 특성 분포를 모델링하여 정확한 롤아웃과 잘못된 롤아웃 간의 구조적 차이를 해독하며 신뢰할 수 있는 자기 보상 신호를 평가하는 데 사용됩니다. 이 접근 방식은 레이블이 있는 데이터의 큰 잠재력을 활용하여 신뢰할 수 없는 레이블 없는 데이터의 품질을 극대화합니다.

- **Technical Details**: GeoMin은 두 단계의 프레임워크로 구성되어 있습니다. 첫 번째 단계에서는 레이블이 있는 데이터를 통해 강력한 분포 식별 가능성을 수립하고, 두 번째 단계에서는 레이블이 없는 데이터를 포함한 반지도 학습을 수행합니다. 여기서 각 레이블 없는 샘플은 정확한 vs 잘못된 vMF(von Mises-Fisher) 분포에 대한 상대적 친화성을 기반으로 기하학적 신뢰 점수를 계산하여 신뢰할 수 있는 샘플을 선택합니다.

- **Performance Highlights**: GeoMin은 89.0%의 F1 점수를 달성하며, 가장 강력한 기준선보다 +4.1% 향상된 성능을 보입니다. 특히, 전체 감독 방식의 기준선을 초과하며 오직 10%의 레이블만으로도 더 나은 성능을 보여 데이터 효율성 측면에서 놀라운 결과를 나타냅니다.



### Self-Evolving Deep Research via Joint Generation and Evaluation (https://arxiv.org/abs/2606.04507)
- **What's New**: 이 연구에서는 기존의 고정된 평가자가 성능 향상에 따라 평가 기준을 동적으로 조정할 수 없는 한계를 극복하기 위해, 평가자와 해결자가 동시에 진화하는 자기 발전 코 진화(training framework for co-evolution) 방식인 SCORE를 제안합니다. 이 방식은 생성 및 평가를 독립된 모듈로 취급하는 대신, 두 가지의 내재적 연결성을 활용하여 성능을 공동으로 개선하도록 합니다. 이론적인 분석을 통해 공유 매개변수(shared-parameter) 하에서의 SCORE의 역할과 평가자의 일관성의 중요성을 규명합니다.

- **Technical Details**: SCORE는 주어진 쿼리(q)와 환경(ℰq)에 대해 후보 보고서(r)의 정책(π)을 학습하는 프레임워크로 구성됩니다. 이 방법은 쿼리에 특정한 증거와 평가 환경에 기반하여 긴 보고 내용을 생성하는 데 필요한 질적 특성을 평가합니다. 또한, 고정된 외부 메타 하네스를 통해 쿼리 특이적인 평가 환경을 형성하며, 평가자는 쿼리 기반의 루브릭을 구성하고 구조화된 보고서를 평가하는 역할을 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 SCORE는 기존의 연구 벤치마크에서 보고서 생성을 향상시키는 데 있어 일관된 성과 개선을 보여주었습니다. 우리의 방법은 평가 차원에 걸쳐 에이전트 성능을 향상시키고, 다양한 생성 작업에서 안정적인 학습 압력을 유지합니다. 코 진화 평가와 생성이 열린 연구 에이전트 훈련의 유망한 방향임을 입증하였습니다.



### Smart Picks in the Dark: Towards Efficient RLVR for Reasoning via Tracing Metacognitive Pivots (https://arxiv.org/abs/2606.04503)
- **What's New**: 본 논문에서는 'pick in the dark' 설정을 통해 RLVR(Reinforcement Learning with Verifiable Rewards)의 훈련에 필요한 무표식 샘플을 전략적으로 선택하는 방법을 제안합니다. 이를 위해 고도로 조정된 불확실성 추정기(unreliable uncertainty estimator)를 사용하여 데이터의 효율적인 분할을 통해 학습을 최적화합니다. 특히, PivotTrace라는 새로운 데이터 트리아지(data triage) 프레임워크를 도입하여 주의(attention) 역학을 활용해 사고 중 메타인지의 전환을 추적합니다.

- **Technical Details**: PivotTrace는 메타인지 피벗(metacognitive pivots)을 기반으로 모형의 불확실성을 정량화하고, 이를 통해 적응적인 데이터 라우팅을 수행하는 3단계 데이터 트리아지 프레임워크입니다. 이를 통해 피벗密度(pivot density)를 정밀하게 측정하여, 자동화된 데이터 라우팅을 가능하게 하여 주석(annotation) 및 훈련 효율성을 극대화합니다. PivotTrace는 두 가지 핵심 메커니즘을 통해 이루어지며, 주의 역학에서 피크 감지를 활용해 피벗 수를 불확실성의 강력한 대리 지표로 삼고, 자동 임계값 조정 모듈을 도입해 최적의 파트셔닝 경계를 동적으로 결정합니다.

- **Performance Highlights**: PivotTrace는 제한된 샘플만으로도 뛰어난 성능을 보이며, 29.3%의 주석이 달린 샘플로 전체 데이터 세트를 대상으로 하는 완전 감독 훈련보다 2.75배 더 빠른 수렴 속도를 자랑합니다. 실험 결과, PivotTrace는 평균 정확도에서 가장 강력한 기준선보다 인도내(ID)에서 +1.6%, 외부 데이터셋(OOD)에서 +2.4% 성능을 향상시켰습니다. 이는 RLVR의 효율성을 크게 개선함을 보여줍니다.



### SFMambaNet: Spectral-Frequency Enhanced Selective State Space Model for Correspondence Pruning (https://arxiv.org/abs/2606.04493)
- **What's New**: 본 논문에서는 최초로 주파수(domain) 영역 인식을 결합하여 대응(인라이어) 프루닝(Correspondence Pruning) 작업을 수행하는 SFMambaNet을 제안합니다. 기존의 Graph Neural Network(GNN) 기반 방법들이 기하학적 특징에 의존하는 반면, SFMambaNet은 지역적 스펙트럴-기하학적 주의(Local Spectral-Geometric Attention)와 스펙트럴 통합 글로벌 맘바(Spectral-Integrated Global Mamba) 블록을 통해 이러한 한계를 극복합니다. 이를 통해 인라이어와 아웃라이어를 보다 효과적으로 구분할 수 있습니다.

- **Technical Details**: SFMambaNet은 두 개의 주요 구성 요소로 나뉩니다. 첫 번째는 LSGA 블록으로, 지역 그래프 상호작용에 스펙트럴 위치 인코딩을 통합하고 다중 스케일 맘바 처리를 도입하여 미세한 기하학적 일관성을 캡처하고 지역 특징의 구별 가능성을 향상시킵니다. 두 번째는 SIGM 블록으로, 주파수 정보(frequency information)를 활용하여 상태 공간 내에서 고주파 잡음(high-frequency noise)의 축적을 억제합니다.

- **Performance Highlights**: SFMambaNet은 여러 도전적인 작업에서 현재의 최첨단 방법들보다 뛰어난 성능을 보입니다. 실험 결과, 이 방법은 인라이어와 아웃라이어 간의 분리 가능성을 향상시키며, 거의 선형 복잡도로 강력한 글로벌 컨텍스트 모델링 능력을 제공합니다. 코드 및 자료는 제공된 링크를 통해 이용할 수 있습니다.



### Evaluating Reasoning Fidelity in Visual Text Generation (https://arxiv.org/abs/2606.04479)
Comments:
          Peer reviewed and accepted at CVPR 2026 at the GRAIL-V (Grounded Retrieval and Agentic Intelligence for Vision-Language) workshop (non-archival track)

- **What's New**: 최근의 텍스트-이미지(T2I) 모델들은 이미지 내에 잘 구조화된 텍스트를 렌더링할 수 있는 능력을 보여주어 문서 생성 및 슬라이드 제작과 같은 다양한 응용 프로그램에 기여하고 있습니다. 그러나 이러한 시스템이 복잡한 솔루션을 텍스트로 직접 표현할 때 추론 능력을 신뢰성 있게 유지하는지 여부는 여전히 불확실합니다. 본 연구에서는 시각적 텍스트 생성을 통한 추론 충실도를 평가하여 이 문제를 조사하고자 하였습니다.

- **Technical Details**: 우리는 여러 과제를 설계하여 현대의 대형 언어 모델(LLM)이 쉽게 해결할 수 있지만 T2I 모델에게는 어려운 다단계 텍스트 추론 문제를 평가했습니다. 주어진 프롬프트에 대해 T2I 모델이 이미지를 생성하고 텍스트를 추출하여 그 정확성을 검증하는 방법으로, 렌더링 오류와 추론 오류를 분리하여 평가를 수행했습니다. 이는 명확한 텍스트 렌더링이 시각적 텍스트에서 추론을 평가하는 데 필수적임을 인식하는 것에서 출발했습니다.

- **Performance Highlights**: 실험 결과, 현재의 T2I 모델들이 논리적으로 일관된 시각 텍스트를 생성하는 데 있어 신뢰할 수 없다는 것이 밝혀졌습니다. 특히, 렌더링된 텍스트가 시각적으로 명확할지라도 의미적 오류와 논리적 불일치, 그리고 잘못된 중간 단계가 빈번하게 발생했습니다. 이 결과는 텍스트 전용 모델이 동일한 작업에서 보여준 강력한 추론 성능과 대조되며, 시각적 텍스트 추론에 있어 더욱 신뢰할 수 있는 해결책이 필요함을 시사합니다.



### ChessMimic: Per-Rating Transformer Models for Human Move, Clock, and Outcome Prediction in Online Blitz Chess (https://arxiv.org/abs/2606.04473)
- **What's New**: ChessMimic은 체스에서 사람의 움직임을 모방하는 세 가지 소형 encoder-only transformer 시스템을 발표합니다. 이 시스템은 움직임, 사고 시간 및 결과 예측을 위해 설계되었으며, 포지션 및 최근 이동 이력, 플레이어 등급, 시계 상태에 따라 조정됩니다. 각 모델은 100-Elo 등급 범위별로 별도의 인스턴스를 맞추어 매개변수 효율성과 기술별 보정의 선명도를 교환합니다.

- **Technical Details**: 체스 시스템 ChessMimic은 256 차원 임베딩, 8개 층, 900만 개의 매개변수로 이루어진 세 가지 독립적인 transformer 모델로 구성됩니다. 각각의 모델은 독립적으로 훈련되며, Lichess 블리츠 게임의 100 포인트 등급 범위에 대해 교육을 받아옵니다. ChessMimic은 운동 예측, 사고 시간 모델 및 결과 예측 모델을 포함하여 총 42개의 체크포인트를 보유합니다.

- **Performance Highlights**: ChessMimic의 인간 이동 예측 정확도는 모든 Elo 등급에서 Maia-2를 초과합니다. 결과 예측 모델은 시계 상태를 포함한 포지션, 플레이어 등급 및 남은 시간을 고려하여 AUC 0.78을 달성했으며, 이는 Maia-2 및 자원 기반 로지스틱 회귀 분석보다 우수한 성능을 보여줍니다. 사고 시간 모델은 ChessMimic이 인간과 유사한 사고 시간 사용을 구현할 수 있게 하며, 비알고리즘적 필터를 사용하여 정확한 신호를 제공합니다.



### Adaptive Calibration for Fair and Performant Facial Recognition (https://arxiv.org/abs/2606.04469)
- **What's New**: Adaptive Calibration (AC)라는 새로운 캘리브레이션(Calibration) 전략을 도입하여 얼굴 인식 시스템의 성능을 향상시킵니다. AC는 정규화된 엠베딩(Embeddings) 간의 코사인 유사도(Cosine Similarity)를 잘 조정된 확률로 매핑합니다. 이는 원래의 거리가 다른 엠베딩 영역에서 서로 다른 일치 확률을 나타낼 수 있는 기본 불일치를 수정합니다.

- **Technical Details**: Adaptive Calibration은 모든 미리 훈련된 얼굴 인식 시스템에 적용할 수 있는 사후(Post-hoc) 접근 방식으로, 비교적 일관된 지역 특화 캘리브레이션(Local Calibration)을 제공합니다. AC는 교차 검증이 필요 없는 공정성과 정확성을 동시에 고려하여, 기존 방법들보다 더 나은 성능을 보여줍니다. 이는 AUC(Area Under the Curve), Brier Score와 같은 다양한 메트릭을 사용하여 평가됩니다.

- **Performance Highlights**: AC는 FairCal 및 FRAPPÉ와 같은 기존의 방법을 초월하여 얼굴 인식 시스템의 공정성과 정확성을 모두 개선했습니다. 이 방법은 인구 통계 데이터 없이도 공정한 얼굴 인식을 가능하게 하며, 다양한 데이터 세트에서 최고의 성능을 기록했습니다. 실험 결과, AC는 다수의 표준 벤치마크에서 기존의 방법들을 지속적으로 능가하는 성과를 보여줍니다.



### ParetoPilot: Zero-Surrogate Offline Multi-Objective Optimization via Infer-Perturb-Guide Diffusion (https://arxiv.org/abs/2606.04468)
- **What's New**: 이번 논문에서는 오프라인 다목적 최적화(Offline MOO)를 위한 새로운 프레임워크인 ParetoPilot를 제안합니다. ParetoPilot는 외부 대리 모델 없이도 작동할 수 있는 제로-서로게이트(zero-surrogate) 확산(diffusion) 프레임워크로, 기존 모델의 조건부 priors를 최대한 활용합니다. 이는 데이터 프라이버시를 유지하면서도 효율성을 높여 줄 수 있는 접근입니다.

- **Technical Details**: ParetoPilot의 핵심 구성 요소는 IPG 엔진(Infer-Perturb-Guide engine)입니다. 이 엔진은 조건부 확산 모델의 비조건부 디노이징 단계에 직접 연계되어 있으며, 이를 통해 최적 방향을 추론하고 제어합니다. IPG 엔진은 근본적으로 수학적 원리를 활용하여 두 가지 힘을 결합해 적절한 다각화(control of diversity)와 수렴(convergence)을 동시에 달성합니다. 

- **Performance Highlights**: 51가지 작업에 대한 광범위한 실험 결과, ParetoPilot는 14개의 최신 대리 모델 및 역 생성 방법론에 비해 뛰어난 성능을 보여주었습니다. 우리의 접근법은 데이터 프라이버시를 존중하면서도 하이퍼볼륨 개선과 파레토 전선(Pareto front)의 강건한 커버리지를 달성하였습니다. 이를 통해 오프라인 MOO에서의 새로운 가능성을 제시합니다.



### SePO: Self-Evolving Prompt Agent for System Prompt Optimization (https://arxiv.org/abs/2606.04465)
Comments:
          26 pages. Code: this https URL

- **What's New**: 이 논문에서는 Self-Evolving Prompt Optimization (SePO)이라는 새로운 방법을 제안합니다. 이는 기존의 시스템 프롬프트 최적화 방식을 넘어서, 프롬프트 에이전트의 시스템 프롬프트도 최적화 대상으로 삼습니다. 한 개의 프롬프트 에이전트가 태스크 에이전트의 시스템 프롬프트와 자신을 모두 향상시키며, 이는 오픈 엔디드(evolutionary search) 진화 검색을 통해 가능합니다.

- **Technical Details**: SePO의 구조는 자기참조(self-referential) 디자인을 따릅니다. 이 방법은 두 단계로 나뉘며, 첫 번째 단계는 여러 태스크에 대한 프리트레이닝(pre-training)이고, 두 번째 단계는 특정 태스크에 대한 파인튜닝(fine-tuning)입니다. 이를 통해 프롬프트 최적화 기술이 특정 태스크에 한정되지 않고, 다양한 태스크에 대해 일반화됩니다.

- **Performance Highlights**: SePO는 수학(AIME'25), 추상적 추론(ARC-AGI-1), 과학(GPQA), 코드 생성(MBPP), 논리 퍼즐(Sudoku) 등 다양한 벤치마크에서 검증되었습니다. Manual-CoT에 비해 평균 정확도가 4.49 포인트 개선되었으며, 모든 태스크에서 최상의 정확도를 달성했습니다. 프리트레이닝과 파인튜닝 단계로 나누는 접근은 향후 여러 태스크에 대해 반복적으로 활용될 수 있는 프롬프트 에이전트의 효율적인 개선을 가능하게 합니다.



### CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities (https://arxiv.org/abs/2606.04460)
Comments:
          ICML 2026

- **What's New**: AI는 사이버 보안을 혁신할 잠재력을 가지고 있으며, 소프트웨어 취약점의 자율 탐지, 분석 및 수정 기능을 제공합니다. 현재의 사이버 보안 평가가 규모나 범위에서 제한적이며 실제 소프트웨어 취약점 발견 및 수정의 전체 생애 주기를 포괄하지 못하는 문제를 해결하기 위해, CyberGym-E2E를 제안합니다. 이 벤치마크는 취약점 발견, PoC 생성 및 패치 생성의 전 과정에서 AI 에이전트의 능력을 종합적으로 평가합니다.

- **Technical Details**: CyberGym-E2E는 139개의 다양한 오픈 소스 프로젝트에 걸쳐 920개의 실제 취약점을 포함하고 있습니다. 우리는 오픈 소스 취약성 데이터를 현실적인 평가 환경으로 변환하기 위해 자동화된 에이전트 강화 파이프라인을 구축했습니다. 이를 통해 현실적인 환경에서 AI 에이전트를 평가하고, 취약점 패치의 정확성을 확인하기 위해 개발자가 작성한 단위 테스트를 자동으로 분석합니다.

- **Performance Highlights**: AI 에이전트의 엔드투엔드 성능은 아직 최첨단 시스템에서도 어려움이 많습니다. 결과적으로 에이전트는 보안 패치 생성에서 높은 성공률을 보였지만, 취약점 탐지 및 PoC 생성은 여전히 도전 과제가 되어 있습니다. CyberGym-E2E는 엔드투엔드 보안 벤치마크를 구성하기 위한 첫 번째 확장 가능한 방법론을 소개하며, 취약점 생애 주기 전반에 걸쳐 AI의 능력을 대규모로 평가하는 데 기여하고 있습니다.



### Token Rankings are Unforgeable Language Model Signatures (https://arxiv.org/abs/2606.04459)
- **What's New**: 본 연구에서는 언어 모델의 파라미터가 로그잇 출력(logit outputs)에 고유한 기하학적 제약을 부여하며, 이로 인해 모델 식별이 가능하다는 점을 강조합니다. 기존의 API에서 제공되는 토큰 순위(token rankings)가 각 모델의 독특한 서명을 형성한다는 것을 발견했습니다. 이는 상응하는 확률 값은 제공되지 않지만, 확률에 따라 순서가 매겨진 토큰 목록이라는 점에서 중요합니다.

- **Technical Details**: 토큰 순위는 각 모델마다 고유한 top-$k$ 순위를 가지며, 이들은 NP-hard 문제로 인해 쉽게 복제할 수 없는 서명으로 기능합니다. 연구 결과, 이러한 순위 기반 서명은 (polynomially) 위조할 수 없는 첫 번째 서명임을 보여줍니다. 또한, API가 허용하는 top-$k$의 크기를 적절히 조절하여 모델 파라미터를 유출하지 않고도 위조 불가능한 서명을 생성할 수 있음을 입증합니다.

- **Performance Highlights**: 본 논문에서는 top-$k$가 충분히 작을 때 API가 모델의 마지막 레이어를 효과적으로 '도용'할 수 있는 가능성을 보여줍니다. 그러나 이러한 도용은 대략적일 뿐이며 서명을 위조하기에는 부족합니다. 최종적으로, 필요한 top-$k$ 값이 도용을 방지하기 위한 k보다 일반적으로 작기 때문에, API는 모델의 파라미터를 유출하지 않으면서도 안전한 서명을 제공할 수 있는 가능성을 지니고 있습니다.



### RowNet: A Memory Transformer for Tabular Regression (https://arxiv.org/abs/2606.04445)
Comments:
          Retrieval-based neural architecture for real estate valuation. Related to TabR (arXiv:2307.14338) and retrieval-augmented tabular learning

- **What's New**: 이번 논문에서는 RowNet를 소개합니다. RowNet는 부동산 가격 예측을 위한 검색 기반(neural architecture) 신경망 아키텍처로, 유사한 기록을 바탕으로 가격 예측을 수행합니다. 이 모델은 쿼리 속성을 메모리 뱅크에 저장된 라벨이 부여된 속성과의 쌍 유사성(features)으로 표현하며, 다단계 검색 프로세스를 통해 예측을 수행합니다.

- **Technical Details**: RowNet는 데이터를 처리하는 데 있어 전통적인 다층 퍼셉트론(MLP)과 다릅니다. 이 모델은 훈련 데이터를 독립적인 입력-출력 쌍으로 취급하지 않고, 전체 훈련 세트를 메모리 뱅크로 활용합니다. 또한, 주목(attention) 점수를 단순한 내적이 아닌 정확한 범주 일치 및 스케일에 민감한 수치 유사성을 포함하여 계산합니다.

- **Performance Highlights**: 논문에서 RowNet의 성능을 평가하고, 이의 학습된 유사 성질이 gradient boosting이나 tabular transformer와 비교할 때 어떻게 가치가 있는지를 설명합니다. 훈련 추적 및 예측 결과를 통해 이 아키텍처의 실질적인 영향을 보여주며, 부동산 시장에서의 지역적 유사성을 효과적으로 모델링하는 방법으로써의 가능성을 제시합니다.



### MemoryDocDataSet: A Benchmark for Joint Conversational Memory and Long Document Reasoning (https://arxiv.org/abs/2606.04442)
Comments:
          17 pages, 2 figures, 8 tables. Submitted for peer review

- **What's New**: MemoryDocDataSet는 AI 시스템의 두 가지 중요한 기능, 즉 긴 대화 기록 탐색과 긴 문서에 대한 깊은 독해 능력을 동시에 평가할 수 있는 새로운 벤치마크 데이터셋입니다. 이 데이터셋은 50개의 마이크로 월드(micro-worlds)와 1,000개의 QA 쌍으로 구성되어 있으며, 각 인스턴스는 3-5명의 페르소나(personas), 다수의 시간적 사건 그래프(temporal event graph), 그리고 20,000-50,000 토큰으로 구성된 긴 문서를 포함합니다. 특히, 기계가 대화 기록을 탐색하여 관련 문서를 찾고 그 문서에서 답변을 추출해야 하는 'Hybrid' 질문이 75.1%를 차지하는 것이 특징입니다.

- **Technical Details**: MemoryDocDataSet은 서로 다른 6개의 기본 구성(베이스라인 설정)을 평가하며, 이들은 잘라낸 컨텍스트, 긴 컨텍스트 LLM(long-context LLM), 회수 증강 생성(retrieval-augmented generation, RAG), 메모리 시스템을 포함합니다. 데이터셋의 품질은 LLM을 사용한 자기 일관성 분석을 통해 측정되며, 50개의 마이크로 월드에서 중위값 Cohen's κ는 0.634입니다. 본 연구의 목표는 메모리 기반 시스템과 긴 문서 내비게이션을 통합하는 새로운 시스템 구조를 동기화하는 것입니다.

- **Performance Highlights**: Baseline 설정에서 RAG-Both 모델이 0.358의 전체 F1 스코어를 달성하였으나, Hybrid 질문에서는 0.342에 불과하여 여전히 인간 성능에 비해 낮은 성과를 보였습니다. Document-only retrieval 방식인 RAG-Doc은 오히려 Hybrid 질문에서 0.267로 저조한 성과를 기록하며 이러한 조합 접근법에서의 명확한 성능 격차를 강조합니다. 이러한 격차는 대화 메모리와 긴 문서 탐색을 통합하는 아키텍처에 대한 필요성을 더욱 부각시킵니다.



### LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling (https://arxiv.org/abs/2606.04438)
- **What's New**: LoopMoE는 Mixture-of-Experts (MoE) 아키텍처와 반복적인(weight-shared) 계산을 결합한 새로운 언어 모델입니다. 이 모델은 IterAdaLN이라는 새로운 모듈레이션 신호를 도입하여 반복 과정에서 발생하는 비대칭 문제를 해결하고, 최적의 attention-to-FFN 비율을 유지하는데 기여합니다. LoopMoE는 이전 구조의 문제를 극복하고, 고유한 효율성을 보장하면서 언어 모델의 가능성을 최대한 활용하는 방향성을 제시합니다.

- **Technical Details**: LoopMoE는 Multi-head Latent Attention (MLA)와 MoE 서브레이어를 결합하여 각 레이어를 구성합니다. 이 모델은 특별히 설계된 샌드위치 레이아웃을 사용하여 반복적으로 구조화된 블록을 통해 깊이를 증가시키면서도 새로운 매개변수를 도입하지 않습니다. IterAdaLN은 반복 인덱스와 각 토큰의 숨겨진 상태를 기반으로 동적으로 affine 파라미터를 생성함으로써 구조적 비대칭을 해결합니다.

- **Performance Highlights**: 3B 규모에서 LoopMoE는 Vanilla MoE에 비해 9개의 벤치마크 중 8개에서 평균 1.0 이상의 성능 향상을 보였습니다. 더욱이, 9B 규모에서도 지속적인 성능 개선을 보여주어, 이 아키텍처의 이점이 큰 규모에서도 유지됨을 입증합니다. 이 연구는 MoE와 반복적인 계산의 접목을 통해 언어 모델의 새로운 가능성을 제시합니다.



### What If Prompt Injection Never Left? Exploring Cross-Session Stored Prompt Injection in Agentic Systems (https://arxiv.org/abs/2606.04425)
Comments:
          position paper

- **What's New**: 본 논문은 LLM(대형 언어 모델) 기반의 에이전트 시스템이 세션에 제한된 보조기능에서 상태ful 시스템으로 변모하는 과정을 설명합니다. 새로운 유형의 공격인 '교차 세션 저장 프롬프트 주입(cross-session stored prompt injection)'을 제안하며, 이는 공격자가 한 번의 상호작용 후에도 시스템 상태에 영향을 미칠 수 있음을 강조합니다. 이를 통해 프롬프트 주입 공격의 새로운 차원을 제시하며, 이러한 시스템에서의 보안 위험을 탐구합니다.

- **Technical Details**: 이 연구는 교차 세션 저장 프롬프트 주입(SPI)이라는 새로운 보안 취약성을 정의하고 체계적으로 분석합니다. 특히, 에이전트 시스템의 저장된 상태에 악의적 내용을 포함시키는 위협을 공식화하고, 이러한 내용이 이후의 상호작용에서 에이전트의 행동에 어떤 영향을 미칠 수 있는지를 설명합니다. 추가로, 벤치마크 및 샌드박스 도구킷을 개발하여 공격 위험성을 평가하고 다양한 모델 및 공격 목표에 대한 정량적 분석을 수행합니다.

- **Performance Highlights**: SPI 공격은 에이전트 시스템의 지속적인 상태 관리 방식에 따라 달라진다는 점이 확인되었습니다. 시스템의 운영 및 상태 저장 방식이 안전성을 저해할 수 있으며, 이는 이후의 실행에서 에이전트의 응답을 조작하거나 도구 기반 행동에 영향을 미칠 수 있습니다. 이러한 발견은 지속적인 에이전트 시스템의 설계 원칙에서 안전한 상태 관리가 우선시되어야 함을 강조합니다.



### L-TGVN: Leveraging Longitudinal Priors for Personalized Rapid MRI (https://arxiv.org/abs/2606.04419)
Comments:
          Accepted to MICCAI 2026

- **What's New**: 본 연구에서는 L-TGVN(긴급 신뢰 기반 가변 네트워크)을 소개합니다. 이 네트워크는 이전 스캔을 부가 정보로 활용하여, 샘플 수가 크게 줄어든 측정값으로부터 현재 스캔을 재구성합니다. 기존의 방법들과 달리, L-TGVN은 이전 스캔과 현재 스캔 사이의 사전 정합(pre-registration)을 필요로 하지 않습니다.

- **Technical Details**: L-TGVN은 측정된 데이터와 이전 스캔 간의 일관성을 유지하면서 이전 스캔의 영향을 제어합니다. 이는 영상 진단 품질을 위해 필수적인 정보와 컨텍스트를 제공하고, 경과에 따른 변화를 고려합니다. 또한 이전 방문 시의 프로토콜 차이에도 유연하게 대처할 수 있는 점이 특징입니다.

- **Performance Highlights**: L-TGVN은 이전 정보 기반 방법 및 장기적 사전 정보를 사용하지 않는 방법들과 비교하여 우수한 성능을 보여주었습니다. 특히, 도전적인 가속 상황에서도 미세 구조(fine structures)를 잘 보존하면서 정량 지표(quantitative metrics)에서 지속적인 개선이 관찰되었습니다. 이 알고리즘의 소스 코드는 링크를 통해 제공됩니다.



### An Empirical Study of Data Scale, Model Complexity, and Input Modalities in Visual Generalization (https://arxiv.org/abs/2606.04409)
Comments:
          12 pages, 9 figures, 4 tables

- **What's New**: 이 연구는 최신 딥 뉴럴 네트워크의 일반화 성능에 영향을 미치는 요인들, 즉 데이터 규모, 모델 복잡성 및 입력 양식에 대한 실증 분석을 제공합니다. 이를 위해 초기 실험에서는 1차원 비선형 함수를 구성하고, 데이터 샘플의 수와 다항식의 차수를 변화시켜 모델 성능에 미치는 영향을 관찰합니다. 주 실험에서는 CIFAR-10 및 CIFAR-100 데이터셋을 이용하여 다양한 교육 데이터 규모와 모델 아키텍처에서의 성능을 비교합니다.

- **Technical Details**: 이 연구는 모델의 일반화 성능에 대해 데이터 규모와 모델 복잡성, 입력 양식이 주는 영향을 살펴보며, 훈련 손실(training loss), 테스트 손실(test loss), 테스트 정확도(test accuracy)를 주요 지표로 사용합니다. 초기 실험에서는 저차원 합성 데이터에서 통제된 환경을 통해 비선형 함수 피팅을 시도하고, 이후 CIFAR-10 및 CIFAR-100 데이터셋에서 MLP, AlexNet, ResNet 모델의 성능을 비교하는 방법론을 채택하였습니다.

- **Performance Highlights**: 실험 결과, 교육 데이터 규모의 증가가 일반화 성능을 지속적으로 향상시키는 것으로 나타났습니다. 반면, 모델 복잡성의 변화는 안정적인 성과 향상을 제공하지 않았습니다. 또한, 색상 정보가 제거될 경우 모델 성능이 저하되며, 기울기(gradients), 에지(edges), 웨이브렛(wavelets)과 같은 명시적 사전 특징이 다양한 모델 아키텍처에서 불균형적인 영향을 미친다는 것을 발견했습니다.



### An Ensembled Latent Factor Model via Differential Evolution and Gradient Descent Optimization (https://arxiv.org/abs/2606.04408)
- **What's New**: 본 연구에서는 고차원 불완전 데이터(HDI)에 적합한 새로운 앙상블 잠재 요인 모델(ELFM-DEGDO)을 제안합니다. 이 모델은 차별 진화(differential evolution) 및 경량 하강법(gradient descent)과 같은 두 가지 최적화 기법을 통합하여 각기 다른 모델에서 생성된 잠재 요인들을 효과적으로 융합합니다. 이를 통해 HDI 데이터에 대한 보다 포괄적이고 편향이 적은 표현을 생성할 수 있습니다.

- **Technical Details**: 제안된 ELFM-DEGDO 모델은 기본적으로 HDI 행렬 R의 저차원 근사로서 잠재 요인 행렬 X 및 Y를 학습하는 구조입니다. 이 모델은 손실 함수와 정규화 항을 결합하여 최적화를 수행하며, 경량 하강법과 차별 진화를 통해 각각의 잠재 요인 행렬을 개선합니다. 특히, 차별 진화는 잠재 요인 행렬이 더 나은 성능을 보일 수 있도록 특별히 설계된 변형 과정을 포함합니다.

- **Performance Highlights**: 실험 결과, ELFM-DEGDO는 세 가지 HDI 데이터셋에서 기존의 여러 잠재 요인 모델에 비해 일관되게 더 나은 성능을 보였습니다. 이를 통해 제안된 모델이 HDI 데이터의 잠재 표현 학습에 있어 더 효과적임을 입증하였습니다. 이 연구는 불완전한 관측치에서 숨겨진 지식을 발견하기 위한 과정에서의 중요한 기여를 나타냅니다.



### Low-Rank Decay for Grokking in Scale-Invariant Transformers: A Spectral-Geometric View (https://arxiv.org/abs/2606.04405)
- **What's New**: 이 논문에서는 Transformer 구조에서 표준 Frobenius-norm weight decay의 한계를 조명하고, 이를 보완하기 위한 새로운 접근법인 Low-Rank Decay (LRD)를 제안합니다. LRD는 핵 노름(nuclear norm) 기반의 스펙트럴 규제기이며, 이는 모델의 성능 향상에 기여합니다. 특히, 이 방법은 훈련 세트를 기억한 이후에도 weight spectrum을 효과적으로 압축할 수 있는 이점을 가집니다.

- **Technical Details**: 논문은 LRD가 기존의 L2 decay가 가진 한계를 극복할 수 있도록 설계되었음을 언급합니다. LRD는 특정 weight matrices에 대해 핵 노름 유사한 감쇠를 적용하여 스펙트럴 희소성을 촉진합니다. Newton-Schulz 반복 방법을 사용해 polar factor를 근사하여 계산 비용을 줄이며, 각 단계에서 단일값 분해(SVD)의 필요성을 제거합니다.

- **Performance Highlights**: 실험 결과, LRD는 Query/Key 행렬에서 효과적인 랭크 붕괴를 유도하고, 적은 데이터 비율에서도 grokking을 촉진하여 효율적인 일반화를 가능하게 합니다. LRD는 데이터 비율 경계를 확장하여 기존의 L2 개선을 보여주고, 이를 측정하는 단계 다이어그램도 포함되어 있습니다. 이로 인해 LRD는 비선형 솔루션으로의 안정적인 전환을 돕는 역할을 합니다.



### TITAN-FedAnil+: Trust-Based Adaptive Blockchain Federated Learning for Resource-Constrained Intelligent Enterprises (https://arxiv.org/abs/2606.04388)
Comments:
          8 pages, 5 figures; code available at this https URL

- **What's New**: TITAN-FedAnil+는 데이터 프라이버시를 보존하면서 협동 지능을 위한 새로운 프레임워크입니다. 이 프레임워크는 블록체인 기반의 연합 학습을 위한 신뢰 기반 적응 네트워크를 제시하며, 악의적인 업데이트를 필터링하기 위해 적응형 클러스터 집계를 도입합니다. 또한, 계산 효율성을 높이기 위해 GPU 가속 벡터화 기법이 적용되었습니다.

- **Technical Details**: TITAN-FedAnil+는 비잔틴 모델을 가정하고, 네트워크 내의 악성 공격자를 탐지하기 위해 Affinity Propagation 기법을 사용합니다. 이 접근 방식은 클러스터 수를 미리 정할 필요 없이 유동적인 유사도 기반 클러스터링을 가능하게 합니다. 아울러, "상태 서명 블록체인 합의" 메커니즘을 통해 균형 잡힌 블록체인 재동기화를 실현합니다.

- **Performance Highlights**: 실험 결과, TITAN-FedAnil+는 50회의 통신 라운드에서 최대 81%의 메모리 오버헤드를 절감하여 8GB 엣지 장치에서의 실행 효율성을 크게 개선했습니다. 이 결과는 안전한 연합 학습 배포를 위한 강건성, 확장성 및 자원 효율성을 효과적으로 향상시킴을 보여줍니다.



### Rethinking Sales Lead Scoring with LLM-based Hierarchical Preference Ranking (https://arxiv.org/abs/2606.04387)
- **What's New**: 이번 연구에서는 고위험 도메인에서의 영업 리드 점수를 매기는 혁신적인 프레임워크인 HPRO(Hierarchical Preference Ranking Optimization)를 소개합니다. 기존의 LLM(대형 언어 모델)을 기반으로 하여 구조적 CRM(고객 관계 관리) 기능과 비구조적 고객 상호작용을 결합한 모델링을 지원합니다. 이 프레임워크는 리드 점수를 매기는 과정에서 적은 양의 데이터로도 보다 효율적인 평가를 가능하게 합니다.

- **Technical Details**: 저자들은 영업 리드 점수를 매기는 문제를 차별화된 구조의 LLM 아키텍처로 재구성합니다. 여기서 HPRO는 희소한 이진 감독 신호를 계층적 선호 신호로 변환하며, 판매 퍼널의 단계 간 계층적 우선순위 를 활용하여 리드의 매력을 높입니다. 제안된 모델은 LLM의 사전 훈련된 구조에 비즈니스 우선순위를 반영한 점수 예측 헤드를 추가함으로써 점수를 매길 수 있는 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, HPRO는 대규모 NEV 브랜드의 데이터를 활용하여 AUC(Area Under Curve) 0.8161을 달성하며, 상위 리드의 정밀도를 39.7% 향상시켰습니다. 또한, 132일에 걸친 온라인 A/B 테스트 결과, 9.5%의 매출 증가를 입증하여 실질적인 상업적 효과를 확인했습니다. 이러한 성과는 판매 리드 스코어링의 새로운 방향을 제시합니다.



### LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignmen (https://arxiv.org/abs/2606.04382)
- **What's New**: 이번 연구에서는 LCSH(Library of Congress Subject Headings)의 공인된 벤치마크가 없다는 문제를 해결하기 위해 LCSHBench를 소개합니다. LCSHBench는 하버드, 컬럼비아, 프린스턴 기록에서 수집된 22,346권의 도서로 구성되어 있으며, 15개 언어로 제공됩니다. 이 데이터셋은 두 개 이상의 독립적인 카탈로그 에이전시가 LCSH를 할당한 도서만 포함되고, 다양한 언어와 주제 유형에 대한 정확도 측정을 지원합니다.

- **Technical Details**: 연구에서 제안한 LCSHBench는 하나의 공통적인 기준으로 카탈로그의 합의를 기반으로 하고 있습니다. 465,187개의 도서에 대한 카탈로그링을 분석하여, 주제에 대한 카탈로그자 간의 동의 정도를 정량화하고, 이는 객관적 개념 수준과 주관적 표현 수준으로 나뉘어 분석됩니다. 이를 통해 두 가지 특정 작업인 open-vocabulary generation과 전체 어휘 검색 파이프라인에 대한 평가 기준을 마련합니다.

- **Performance Highlights**: LCSHBench를 활용한 첫 번째 실험에서는 300M 온디바이스 임베더를 저랭크로 파인튜닝하여 언어 간 검색 성능을 개선했습니다. 개발 세트의 정확한 회수율에서 3,072차원으로 호스팅된 임베더(0.659)보다 더 나은 성과를 보였습니다(0.623). 하지만 이 연구에서는 언어 패널에 따라 성과가 균일하지 않음을 보여주며, 향후 연구로는 보류된 테스트 및 엔드 투 엔드 확인을 계획하고 있습니다.



### From Symbolic to Geometric: Enabling Spatial Reasoning in Large Language Models (https://arxiv.org/abs/2606.04381)
- **What's New**: 이번 논문에서는 새로운 모델인 Spatial Language Model (SLM)을 소개합니다. SLM은 위치 정보를 1순위 모달리티로 다루며 모델의 추론 과정에서 기하학적 공간 추론을 가능하게 합니다. 기존의 대형 언어 모델(LLM)이 과제로 삼았던 기하학적 계산과 구조적 공간 연산자에 대한 네이티브 지원이 없었기 때문에 이에 대한 대안을 제시합니다.

- **Technical Details**: SLM은 학습된 공간 표현을 직접 사용하여 텍스트 설명이 아닌 공간 관계를 처리합니다. 이를 위해, 공간 표현, 원자적 기하학 연산, 자연어 지침을 정렬한 Spatial Instruction Dataset을 구축하였습니다. 새로운 벤치마크인 SpatialEval은 속성, 거리, 토폴로지, 상대 위치 작업을 통한 공간 추론을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: SLM은 기존의 기호적 추론에 의존하는 LLM 기반 접근 방식을 뛰어넘어 성능을 크게 향상시켰습니다. 실험 결과, SLM은 기하학적 공간 표현을 통합함으로써 강력한 공간 추론의 이점을 입증하였습니다. 이 연구에서 제안된 데이터셋, 평가 벤치마크, 모델 학습 코드 및 체크포인트는 제공된 링크에서 확인할 수 있습니다.



### DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling (https://arxiv.org/abs/2606.04374)
Comments:
          Jing Wang (Corresponding Author)

- **What's New**: 이 논문은 전자 상거래 검색의 관련성 문제를 다루고 있으며, 디스크리트 시맨틱 아이디 (SID)를 통해 쿼리 의존적인 순위를 모델링하는 새로운 방법인 DSIRM을 제안합니다. 기존에 사용되던 비지도 학습 기반의 SID 생성 방식의 한계를 극복하고자 하며, 쿼리-아이템 상호작용을 활용하여 정확한 시맨틱 패턴을 학습하는 방식을 채택합니다. 또한, 생성적인 대형 언어 모델을 이용하여 텍스트에서 아이템 SID를 예측하는 방법을 제시합니다.

- **Technical Details**: DSIRM은 쿼리-브리징 대조적 양자화 접근법을 통해 아이템 사이드에서 쿼리-아이템 상호작용을 모델링하며, 잔여 양자화(Residual Quantization) 과정에서 InfoNCE loss를 활용하여 유사한 쿼리와 함께 발생하는 아이템에 비슷한 SID를 부여하도록 합니다. 텍스트로부터 아이템 SID를 예측하기 위해 자가 회귀(autoregressive) LLM을 조정하여 의도 모호성을 해결하고, 쿼리 및 아이템 SID 간의 계층적 접두사 매칭을 통해 연속 임베딩을 보완하는 차별적인 특성을 생성합니다.

- **Performance Highlights**: Tmall의 대규모 생산 데이터를 활용한 실험 결과, 제안된 방법이 현재의 최첨단 모델보다 더 나은 성능을 나타내며, 오프라인 AUC가 +1.54% 향상되었습니다. 효율적인 하이브리드 아키텍처를 통해 배포된 DSIRM은 온라인에서 UCTR과 UCTCVR을 각각 +0.13%, +0.25% 향상시키며 상당한 산업적 가치를 입증하고 있습니다.



### Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers (https://arxiv.org/abs/2606.04373)
Comments:
          Accepted to appear at ICML 2026, Seoul, Korea

- **What's New**: 이번 논문에서는 Vision Transformers (ViTs)에 대한 새로운 Data-Free Quantization (DFQ) 접근법인 MaskAQ를 제안합니다. 기존의 DFQ 기법들이 합성 샘플과 양자화 모델의 입력 분포 간의 차이로 인해 성능 저하를 겪는 문제를 해결하기 위해, MaskAQ는 매우 중요한 이미지 패치인 Informative Region (유용한 영역)을 통해 성능을 극대화합니다. 이 기법은 저품질 샘플의 문제를 해결하기 위해 패치 유사성에 대한 차이 엔트로피를 최대화하여 노이즈 배경에서 유용한 영역을 분리하는 방법을 사용합니다.

- **Technical Details**: MaskAQ는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 차분 엔트로피 최대화를 통해 합성 샘플의 유용한 영역을 노이즈 배경에서 분리합니다. 둘째, 정보 지역을 고정시키기 위한 적응형 마스킹 메커니즘이 있으며, 이를 통해 양자화 모델(Q)와의 Masked Attention Alignment를 수행합니다. 마지막으로, 주기적인 샘플 갱신 전략을 통해 합성 샘플과 Q의 출력을 지속적으로 조화롭게 유지하여, 훈련 과정의 진화에 적응할 수 있는 능력을 부여합니다.

- **Performance Highlights**: MaskAQ는 다양한 백본 및 다운스트림 작업에서 기존의 최첨단 기법들을 초월하는 성능을 보여줍니다. 특히 3비트 양자화 상황에서 MaskAQ는 ImageNet 데이터셋의 DeiT-T 모델에서 기존 기법 대비 최대 3.1%의 Top-1 정확도 향상을 이끌어냈습니다. 이러한 결과는 MaskAQ가 VIts를 위한 DFQ에서 경쟁력 있는 솔루션임을 잘 보여줍니다.



### Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes (https://arxiv.org/abs/2606.04365)
- **What's New**: 이 논문에서는 신장( kidney ) CT( 컴퓨터 단층촬영 )에서 병변( lesion ) 특성을 단일 모델로 예측하는 새로운 접근법을 제안합니다. 기존의 3D 방법들은 환자나 장기 단위에서 예측하였으나, 우리는 각 신장 당 병변을 개별적으로 예측할 수 있는 방법으로 문제를 재구성하였습니다. 이를 통해 2,619개의 CT 볼륨을 사용하여 다층적 레이블링을 수행하였으며, KiTS23 데이터셋을 통해 외부 유효성을 검증합니다.

- **Technical Details**: 제안된 모델, LesionDETR은 DETR 스타일 아키텍처를 채택하여 크기 및 거리의 헝가리안 매칭을 사용하며, 각 슬롯 출력값을 종합하여 측면 수준의 목표로 삼는 계층적 손실을 사용합니다. 네 가지 입력 표현법과 여섯 가지 인코더 초기화 방식을 통해, 분할 마스크를 입력 채널로 사용하는 것과 같은 도메인의 복부(SuPreM) 사전 훈련이 주요 설계 요소로 나타났습니다. 일반 대규모 데이터로의 사전 훈련은 무작위 초기화보다 성능이 떨어지는 것으로 확인되었습니다.

- **Performance Highlights**: LesionDETR는 UF-Health에서 양측 측면 수준의 이상 AUC(곡선 아래 면적) 0.799 ± 0.009, KiTS23에서는 0.817 ± 0.072를 달성하였습니다. 또한, 카운트 조건부 변형 모델은 낭종 병변에 대해 per-lesion mAP(평균 정밀도) 0.190 ± 0.083을 기록했으며, 드문 고형 병변에서의 AP는 여전히 낮은 수준을 유지하고 있어 데이터 수집의 필요성을 강조합니다. 이 프레임워크는 하류의 구조화된 보고서 생성을 위한 개별 병변 예측을 검증할 수 있는 기능도 제공합니다.



### MorphoQuant: Modality-Aware Quantization for Omni-modal Large Language Models (https://arxiv.org/abs/2606.04349)
- **What's New**: 본 논문에서는 기존의 4-bit Omni-modal Large Language Models (OLLMs)에서 발생하는 포스트 훈련 양자화(Post-Training Quantization, PTQ)의 과제를 해결하기 위해 MorphoQuant라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 교차 모드 형태학(cross-modal morphology)을 보존하고, 아웃라이어 손실을 완화하기 위해 Distribution-Aware Bias Compensation (DABC) 메커니즘을 도입합니다. 또한, Morphology-Directed Quantization Function Optimization (MDQFO) 전략을 통해 양자화 그리드와 편향 마스크를 공동 최적화하여 다양한 모드 분포 간의 미세 조정을 보장합니다.

- **Technical Details**: MorphoQuant는 다양한 감각 입력이 포함된 OLLMs의 극한의 메모리 요구와 컴퓨팅 비용 문제를 해결하기 위해 설계되었습니다. DABC 메커니즘은 다채로운 아웃라이어 패턴의 영향을 중립화하여 채널별 편향을 통해 트렁케이팅 잔여물을 흡수합니다. MDQFO에 의해 최적화된 양자화 기능은 분산 스코어를 통해 아웃라이어가 길게 늘어진 경향을 수량화하여, 다양한 모드의 분포 정렬을 향상시키는 데 기여합니다.

- **Performance Highlights**: 본 연구에서 제안한 W4A4 모델은 ScienceQA에서 76.63%의 성능을 기록하며, 기존의 SOTA W4A4 방법을 뛰어넘고, 심지어 W4A16 기준선도 초과하는 놀라운 결과를 보여줍니다. 이러한 성과는 MorphoQuant 프레임워크의 정확도-효율성 간의 탁월한 균형을 입증하며, 다양한 모드의 분포 간의 전환을 더욱 원활하게 할 수 있는 잠재력을 보여줍니다.



### HYolo: An Intelligent IoT-Based Object Detection System Using Hypergraph Learning (https://arxiv.org/abs/2606.04345)
Comments:
          8 pages, multiple figures;

- **What's New**: 이번 논문에서는 하이퍼그래프 학습을 YOLO 아키텍처에 통합한 HYolo라는 인공지능 IoT 기반 객체 탐지 프레임워크를 제안합니다. 전통적인 YOLO 모델은 객체 간의 복잡한 고차원 관계를 모델링하지 못하는 경향이 있어, HYolo는 이러한 한계를 극복하기 위해 하이퍼그래프 학습을 통해 맥락 의존성을 더욱 풍부하게 캡처합니다. 실험 결과, COCO 데이터셋에서 기존 YOLO 모델 대비 약 12% 향상된 mAP@50 점수를 기록하며 전반적인 탐지 정확도와 강인성을 개선했습니다.

- **Technical Details**: HYolo 접근법은 하이퍼그래프를 사용하여 더 높은 차원의 상호작용을 촉진하고 맥락화를 개선합니다. 이 모델은 HyperC2Net 아키텍처를 통해 여러 피쳐 노드 간의 상호작용을 이끌어내며, 하이퍼그래프 컨볼루션(HyperConv) 계층을 도입하여 맥락 정보를 더 잘 추출할 수 있습니다. 기존 YOLO 기반 모델에서 단순한 피쳐 융합 방식으로 인해 기능 상호작용이 제한되었던 문제를 해결하고, 다중 차원 특성의 관계를 학습할 수 있도록 합니다.

- **Performance Highlights**: HYolo는 COCO 데이터셋에서 테스트되어 pAP@50, 손실, 정밀도-재현율 등의 성능 지표를 통해 평가되었습니다. 이 연구에서 도입한 거리 기반 하이퍼그래프 구축 방식 및 하이퍼그래프 컨볼루션은 모델이 복잡한 패턴을 학습할 수 있도록 도와주며, 특히 가벼운 탐지 시나리오에서 성능을 향상시킵니다. 결과적으로 HYolo는 평범한 YOLO 모델에 비해 탐지 효율성을 크게 개선한 것으로 나타났습니다.



### Expectations vs. Realities: The Cost of MSE-Optimal Forecasting Under Conditional Uncertainty (https://arxiv.org/abs/2606.04342)
Comments:
          12 pages, Accepted for KDD 2026 Research track

- **What's New**: 이번 연구에서는 다단계 시계열 예측(MSF)의 정확성과 현실성을 동시에 최적화하려는 데 어려움을 설명합니다. 우리는 평균 제곱 오차(MSE)가 증가하는 상황에서 조건부 분포가 어떻게 중요해지는지를 formalize합니다. 이를 통해 조건부 불확실성이 큰 경우 평균 제곱 오차의 최소화를 진행할 수 없는 상황을 규명했습니다.

- **Technical Details**: 연구는 조건부 불확실성의 분포를 다루며, 불확실성이 시간에 따라 축적되는 방식을 수학적으로 formalize했습니다. 이를 통해 예측 결과의 다양성을 설명하는 매개변수인 '조건부 불확실성 갭'을 도입하였고, 이는 사실상 어떤 결정론적 예측기도 MSE를 최소화하면서 미래 값의 분포를 일치시킬 수 없음을 보여줍니다.

- **Performance Highlights**: 실험 결과, MSE를 약간 완화하여도(5% 이하) 현실적 변동성을 극대화할 수 있음을 발견했습니다. 이러한 변화를 통해 평균 17.3%의 현실성 개선이 가능하며, 일부 데이터 세트에서는 30% 이상의 이득을 보였습니다. 이 연구는 MSE 기반 모델 선택의 구조적 결함을 밝혀내며 결국 정확성과 현실성 간의 트레이드오프를 탐색하는 것을 강조합니다.



### From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents (https://arxiv.org/abs/2606.04329)
- **What's New**: 이 논문에서는 AI 에이전트의 메모리 시스템에서의 메모리 중독(memory poisoning) 공격을 다룬 체계적인 연구 결과를 제시합니다. 저자들은 네 가지 메모리 쓰기 채널과 아홉 가지 구조적 취약점을 식별하고, 이를 바탕으로 메모리 중독 공격의 세 가지 분류 체계를 개발합니다. 또한, 새로운 벤치마크인 MPBench를 설계하여 메모리 중독 공격 평가를 위한 기초를 마련하였습니다.

- **Technical Details**: AI 에이전트의 메모리는 크게 단기 메모리와 장기 메모리로 나눌 수 있으며, 장기 메모리는 세션 간 지속적으로 정보를 저장합니다. 장기 메모리는 바람직한 사용자의 선호 및 과거 작업 결과를 기록하여 AI 에이전트의 향후 동작에 직접적인 영향을 미칩니다. 저자들은 네 가지 메모리 쓰기 채널을 정의하며, 각 채널에서 발생할 수 있는 공격의 구조적 취약점을 분석합니다.

- **Performance Highlights**: 연구 결과, 메모리 중독 공격의 평균 성공률은 50.46%로 나타났으며, 기존의 프롬프트 삽입 방어책들이 메모리 중독 공격에 대해 효과적이지 않음을 보여주었습니다. 메모리 기록 및 읽기 동작이 활발할수록 공격의 성공률이 증가하며, 이로 인해 AI 에이전트의 취약성이 드러나고 있습니다. 이러한 결과는 메모리 중독 공격의 예방 및 방어 디자인을 위한 기초 자료로 활용될 수 있습니다.



### Generalizable Multi-Task Learning for Wireless Networks Using Prompt Decision Transformers (https://arxiv.org/abs/2606.04328)
Comments:
          Accepted paper at IEEE International Mediterranean Conference on Communications and Networking (MeditCom) 2026

- **What's New**: 이 논문에서는 인공지능 기반 라디오 자원 관리(RRM) 시스템을 제안합니다. 이 시스템은 행동 정책을 학습하기 위해 Prompt Decision Transformer(PromptDT)를 사용하여 다중 셀 선택 문제를 시퀀스 모델링 문제로 변환합니다. 이를 통해 다양한 네트워크 구성에서 빠르게 적응하고, 다양한 베이스 스테이션(BS)과 사용자 장비(UE) 수를 처리할 수 있는 능력을 갖추었습니다.

- **Technical Details**: 시스템 모델은 다운링크(dl) CoMP 전송 시나리오를 고려합니다. 여기서 UE와 BS의 수는 조정 가능하며, UE의 이동은 랜덤 웨이포인트 모델을 따릅니다. 이는 Okumura-Hata 경로 손실 모델을 사용해 도시 매크로 셀 환경에서 통신 채널 전파를 모델링하며, BS 자원은 리소스 공정(ReF) 또는 비례 공정(PF) 스케줄링을 통해 할당됩니다.

- **Performance Highlights**: 실험 결과, PromptDT는 다중 작업 설정에서 기준선 대비 최대 49% QoE 개선을 보여주었습니다. 다양한 작업 환경에서의 성능 수준을 유지하면서 모델 용량이 증가함에 따라 긍정적인 확장성을 보여줍니다. 또한 새로운 작업에 대해 강인하게 몇 번의 샷으로 적응할 수 있는 능력을 검증하였습니다.



### A Geometric Characterization of the Stationary Plateau for Two-Layer Neural Networks (https://arxiv.org/abs/2606.04327)
Comments:
          47 pages

- **What's New**: 이번 연구는 부드러운 활성화 함수를 가진 두 계층 신경망의 손실 경관에서 나타나는 고정된 평탄면(stationary plateaus)의 기하학적 구조를 조사합니다. 특히, 숨겨진 뉴런을 복제함으로써 생기는 '뉴런 분할(neuron splitting)' 현상에 초점을 맞추고, 이러한 평탄면 상의 모든 고정 점(stationary points)에 대한 포괄적인 분류를 제공합니다.

- **Technical Details**: 우리는 고정된 평탄면의 구조를 이해하기 위해 '내부 헤시안(inner Hessian)' 행렬을 도입하고, 이를 통해 각 뉴런의 곡률을 평가하여 해당 고정 점이 지역 최소값(local minima)인지, 안장 점(saddle point)인지를 결정합니다. 연구 결과에 따르면, 뉴런 분할이 지역 최소값을 분할할 경우, 지역 최소값과 안장 점이 혼합된 평탄면이나 전부 안장 점으로 이루어진 평탄면이 생성될 수 있습니다.

- **Performance Highlights**: 이번 논문은 신경망의 폭 확장(width expansion)이 고정 점의 성격을 어떻게 보존하거나 변화시키는지를 설명하며, 이는 신경망 구조의 재파라미터화(reparameterization)와 관련된 새로운 기하학적 통찰을 제공합니다. 연구 결과는 신경망에서 넓은 구조가 더 유리한 최적화 행동을 보이는 이유를 풀어내는 데 기여할 수 있습니다.



### Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models (https://arxiv.org/abs/2606.04326)
Comments:
          Benchmarks available at this https URL

- **What's New**: 본 논문에서는 concept bottleneck models (CBMs)을 위한 새로운 합성 벤치마크(synthetic benchmarks)를 개발하여 연구진의 연구 및 개발을 지원합니다. 이러한 벤치마크는 데이터 모달리티(data modality), 개념 선택(concept choice), 주석 품질(annotation quality), 완전성(completeness) 등을 제어하여 레이블이 있는 데이터셋을 생성합니다. CBMs의 주요 사용 사례인 결정 지원(decision support)과 자동화(automation)를 중점적으로 다룹니다.

- **Technical Details**: CBM은 데이터를 기반으로 개념을 예측한 후, 해당 개념을 최종 레이블로 매핑하는 기계학습 시스템입니다. 본 논문은 CBMs의 성능 한계를 식별하고 그들이 작동하는 범위를 결정하는 방법을 포함하여 다양한 조건에서 성능 변화 평가를 가능하게 합니다. 이를 위해 새로운 벤치마크 과제와 메트릭(metrics)을 설계하여 CBM이 제공하는 독특한 가치를 반영합니다.

- **Performance Highlights**: 논문에서는 자동화된 작업과 사람이 의사 결정을 내릴 때의 성과 향상을 비교 분석하여 CBMs의 성능을 평가합니다. 사전 설정된 조건에서 CBMs의 실험 결과를 통해 모델의 실패 모드를 진단하고 후속 실험을 안내하는 방법을 보여줍니다. 이러한 접근 방식은 CBMs의 적용 가능성과 다양한 문제 해결을 위한 기여를 강조합니다.



### OpenRFM: Dissecting Relational In-Context Learning (https://arxiv.org/abs/2606.04320)
Comments:
          25 pages, including appendix

- **What's New**: Relational Foundation Models (RFMs)는 이제 단일의 사전 훈련된 예측기를 제공하여, 관계형 데이터베이스에 대해 하나의 전방 패스를 통해 예측을 수행할 수 있는 가능성을 보여줍니다. 그러나 오픈 RFMs와 상업용 RFMs 간에는 현저한 성능 차이가 존재하고, 그 원인은 체계적으로 이해되지 않았습니다. 이 연구에서는 Relational Transformer (RT)를 대상으로 하여 모델과 데이터 측면에서 이 격차의 원인을 분석하고, 두 가지 진단을 통해 모델의 성능 향상 방안을 제시합니다.

- **Technical Details**: 모델 측면에서는, RT가 관계 수준의 in-context learning (ICL)을 수행하기 때문에, 라벨이 있는 셀의 커버리지 부족으로 인해 퇴화된 회귀가 발생함을 보여줍니다. 데이터 측면에서는, RT의 사전 훈련 출처를 분석하여, 기존의 합성 데이터만으로의 사전 훈련은 충분하지 않음을 입증하며, 다양성이 확보된 실 데이터와 합성 데이터를 혼합한 새로운 사전 훈련 전략을 제안합니다. 이를 통해 OpenRFM 모델이 개발되었으며, 이는 기존 RT 모델을 기반으로 하여 더 낮은 라벨 커버리지에 대응할 수 있도록 두 단계의 ICL 구조를 채택합니다.

- **Performance Highlights**: OpenRFM 모델은 RT 백본에 비해 평균적으로 약 30% 성능이 향상되며, KumoRFMv1 상업 모델에 비해 더욱 효과적인 결과를 보여줍니다. 이는 다양한 평가테스트에서 입증되며, 단순하면서도 효과적인 접근 방식을 통해 parametric RFMs의 효율성을 증대시키는 데 기여합니다. 따라서 OpenRFM은 관계형 데이터베이스 활용의 새로운 가능성을 제시하고 있습니다.



### Anycast Performance in Contex (https://arxiv.org/abs/2606.04298)
- **What's New**: 본 논문은 IP anycast 기술이 다양한 응용 프로그램에서 어떻게 다르게 작용하는지를 탐색합니다. 특히 root DNS와 콘텐츠 배급 네트워크(CDN)에서의 anycast 지연(latency)을 비교합니다. 이를 통해 anycast의 영향이 각 응용 분야에 따라 상이하게 나타나며, 서로 다른 최적화 전략이 필요하다는 것을 제시하고 있습니다.

- **Technical Details**: 논문은 root DNS에서의 recursive caching과 CDN에서의 추가 왕복 시간(round trip)이 지연에 미치는 영향을 분석합니다. 연구는 각 설정에 대한 비교 지연(latency) 모델을 제시하고, 반복 가능한 측정(design) 방법론과 최적화 프레임워크를 개발합니다. 이를 통해 resilience-driven objectives와 latency-driven objectives의 분리가 가능해집니다.

- **Performance Highlights**: 연구 결과에 따르면, root DNS anycast는 경로 팽창(path inflation)이 나타날 수 있지만, 사용자가 인식하는 지연은 제한적입니다. 반면, CDN anycast는 경로 정책(route policy), 페어링(peering) 및 측정 피드백을 적극적으로 관리해야 하며, 이는 지연을 최소화하기 위한 노력으로 여겨집니다. 따라서 각 서비스에 대한 최적화 목표가 다름을 강조하며, root DNS에 대한 강건성은 유지관리, 그리고 CDN 서비스는 꼬리 지연(tail latency) 제어가 중심이 되어야 함을 제안합니다.



### Scaling Novel Graph Generation via Lightweight Structure-Guided Autoregressive Models (https://arxiv.org/abs/2606.04287)
- **What's New**: 이 논문은 새로운 경량 자회귀 (autoregressive) 그래프 생성 프레임워크를 제안하여 확장성과 참신성 문제를 해결합니다. 이 프레임워크는 구조 기반의 토폴로지 정렬을 사용하여 그래프를 정규 엣지 시퀀스로 직렬화하며, 이에 따라 near log-linear 생성이 가능해졌습니다. 또한, 탐색 지향적 증강과 반복적인 정제를 결합한 두 단계의 훈련 전략을 통해 과적합을 줄이고 통제된 참신성을 촉진합니다.

- **Technical Details**: 제안된 프레임워크는 SIR-GN에서 유도된 무감독 (unsupervised) 구조적 노드 표현을 기반으로 하는 엄격한 구조 가이드 직렬화 전략을 도입합니다. 이렇게 생성된 정규 이진 엣지 시퀀스는 경량 자회귀 모델이 학습하기 용이하도록 합니다. 이 프레임워크는 두 단계 훈련 전략을 통해 탐색 및 타당성 균형을 맞추며, 엠베딩 공간에서 GMM을 사용하여 발생하는 그래프를 필터링합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 높은 타당성과 독특성을 유지하면서 참조 데이터 세트에 없는 참신한 그래프를 다수 생성하는 데 성공했으며, QM9, ZINC250K, MOSES 등의 다양한 벤치마크에서 성능 향상을 보였습니다. 제안된 프레임워크는 경량의 원인 시퀀스 백본을 통해 통제된 참신성과 실용적인 확장성을 실현하며, LSTM 및 Mamba 스타일의 디코더를 지원합니다.



### Sparse Mixture-of-Experts Reward Models Learn Interpretable and Specialized Experts for Personalized Preference Modeling (https://arxiv.org/abs/2606.04284)
- **What's New**: 이번 연구에서는 희소 Mixture-of-Experts (MoE) 보상 모델을 제안하여 인간의 다양한 선호도를 보다 명확히 반영하고자 합니다. 기존의 접근 방법들이 보편적인 보상 함수를 가정하는 경향이 있는 반면, 이 모델은 이진 선호 데이터에서 여러 선호 구성 요소를 학습해 개별 선호를 모델링합니다. 훈련 과정에서 희소한 경로 선택(sparse routing)과 전문가 다양성(expert diversity)을 촉진하여 해석 가능성과 개인화를 개선합니다.

- **Technical Details**: 희소 MoE 보상 모델은 이진 선호 데이터를 기반으로 훈련됩니다. 이 모델은 Bradley-Terry 모델을 활용하여 두 가지 반응에 대한 선호 확률을 정의하며, 주어진 프롬프트에 대한 인간의 선호도를 추정합니다. 모델의 훈련은 관찰된 선호의 음의 로그 가능성(minimizing negative log-likelihood)을 최소화하는 방식으로 진행되어, 해석 가능성과 전문성을 갖춘 전문가들이 생성됩니다.

- **Performance Highlights**: 희소 MoE 모델은 실험을 통해 해석 가능한 경로 선택 패턴과 전문화된 전문가를 학습하여 개인화에 있어 상당한 개선을 이끌어냅니다. 오직 50개의 적응 예제만으로도 개인화의 정확성이 25.81 포인트 향상되었으며, 이는 기존의 방법들보다 크게 발전된 성과입니다. 전문가의 가중치 변화는 목표 선호와의 의미론적 연관성을 잘 드러내 보여 모드 적응 과정을 검토하는 유용한 수단이 됩니다.



### The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning (https://arxiv.org/abs/2606.04280)
- **What's New**: 이 논문은 대비 학습(contrastive learning, CL)에 관한 새로운 이론적 기초를 제시합니다. 기존의 CL 메커니즘이 어떻게 의미 있는 잠재 구조(latent geometry)를 회복하는지에 대한 이해를 심화시키고자 합니다. 특히, 데이터 샘플링의 다름(diversity condition)이 긴급히 필요함을 강조하며 이 조건이 위배될 경우 발생하는 기하학적 왜곡을 규명합니다.

- **Technical Details**: 저자들은 잠재 공간(코드) 샘플링에 대한 측정 이론(measure-theoretic) 기반의 다양성 조건(diversity condition)을 공식화했습니다. 이 조건은 아이소메트릭(거리 보존) 회복을 위해 필수적인 것으로, 샘플링이 제한된 경우의 고전적인 전지원(von Mises-Fisher) 설정에서 적합한 결과를 도출합니다. 이를 통해 적절한 InfoNCE 목표를 제안하고, 기하학적 보존(latent space recovery)을 가능하게 하는 방법론을 살펴봅니다.

- **Performance Highlights**: CIFAR-10 데이터셋을 통해 이론적 예측을 실험적으로 검증하며, 건축 인덕티브 바이어스(architectural inductive bias)가 샘플링 다양성이 제한될 때 표현 품질에 어떤 영향을 미치는지 분석합니다. 결과적으로, 다양한 샘플링 전략과 인코더 구조가 대조적 표현 학습에서 어떻게 상호 작용하는지를 명확히 하여, 향후 연구 및 방법론의 발전 방향을 제시합니다.



### From Ticks to Flows: Dynamics of Neural Reinforcement Learning in Continuous Environments (https://arxiv.org/abs/2606.04275)
Comments:
          Presented at ICLR 2026: this https URL

- **What's New**: 본 논문에서는 연속 환경에서의 심층 강화 학습(Deep Reinforcement Learning)을 위한 새로운 이론적 틀을 제시합니다. 이 접근법은 문제를 연속 시간 확률 프로세스(Continuous-time Stochastic Process)로 모델링하며, 확률적 제어(Stochastic Control)의 통찰을 활용합니다. 우리는 탐색(Exploration)과 확률적 전이(Stochastic Transitions)를 모두 포함하는 유효한 액터-크리틱 알고리즘 모델을 소개합니다.

- **Technical Details**: 우리는 단일 은닉층 신경망을 기반으로 환경의 상태를 두 개의 시간 척도 프로세스(Two Time Scale Process)로 모형화하는 방법을 보여줍니다. 여기서 환경 시간(Environment Time)과 경량 시간(Gradient Time)이라는 두 가지 시간 축이 활용됩니다. 특히, 무한한 너비의 두 개 층 네트워크에서 환경 상태를 나타내는 시간 의존적 확률 변수(Random Variables)의 진화를 경량 단계(Gradient Steps)에서 어떻게 변화하는지를 분석합니다.

- **Performance Highlights**: 우리는 확률적 미분 방정식(Stochastic Differential Equations)의 이론을 이용하여 연속 강화 학습에서 처음으로 경량 단계를 거치면서 상태 분포(State Distribution)의 미세한 변화에 대한 방정식을 도출합니다. 이 연구 결과는 과제의 학습 속도(Learning Rate)가 매우 작을 때의 변화에 주목합니다. 마지막으로, 우리는 장난감 연속 제어 과제(Toy Continuous Control Task)를 사용하여 이론적 결과를 실증적으로 뒷받침합니다.



### StandardE2E: A Unified Framework for End-to-End Autonomous Driving Datasets (https://arxiv.org/abs/2606.04271)
- **What's New**: 본 논문에서는 autonomous driving (자율 주행) 분야에 있어 E2E (end-to-end) 모델의 발전을 다룹니다. 연구자들이 다양한 sensor-rich driving datasets를 사용할 때, 각 데이터셋의 파일 형식과 API가 서로 다르기 때문에 적절한 preprocessing을 수행하는 데 많은 시간을 소요해야 했습니다. 이를 해결하기 위해 StandardE2E라는 통합 프레임워크를 제안하여, 데이터셋 간의 차이를 줄이고 사용성을 높입니다.

- **Technical Details**: StandardE2E는 하나의 통합된 데이터 스키마 아래에서 각 데이터셋별 preprocessing을 표준화하며, 여러 데이터셋을 하나의 PyTorch DataLoader로 조합할 수 있도록 합니다. 사용자는 새 데이터셋을 추가할 때 raw frames를 canonical schema로 매핑하는 것만 신경 쓰면 되며, 나머지 파이프라인은 변하지 않습니다. 또한, HD-map, 3D detections, driving command 등 다양한 modality를 지원하여 자율 주행 연구를 더욱 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: 현재 StandardE2E는 Waymo End-to-End, Waymo Perception, Argoverse 2 Sensor 등 6개의 데이터셋을 기본적으로 지원합니다. cross-dataset pretraining과 auxiliary-task supervision을 통해 데이터 처리의 효율성을 크게 향상시킬 수 있으며, 통합된 구조 덕분에 새로운 데이터셋 추가 시 소요되는 비용을 최소화합니다. 이로 인해 연구자들은 더 간편하게 다양한 데이터셋을 활용하여 실험할 수 있게 되었습니다.



### Instant-Fold: In-Context Imitation Learning for Deformable Object Manipulation (https://arxiv.org/abs/2606.04269)
- **What's New**: 이번 논문에서는 Deformable Object Manipulation (DOM)을 위한 새로운 프레임워크인 Instant-Fold를 제안합니다. Instant-Fold는 단일 인간의 시연을 통해 다양한 조작 모드를 추론하고 실행할 수 있는 imitation learning의 한 형태입니다. 특히, 이 방법은 gradient 업데이트 없이도 다양한 조작 모드를 직접적으로 인식하고 적용할 수 있는 장점이 있습니다.

- **Technical Details**: Instant-Fold는 시간 대비 대조적 사전학습(temporal contrastive pretraining)을 통해 변형에 대한 인지 시각 표현을 학습합니다. 이후, 시연을 조건으로 하는 flow-matching transformer 정책을 이용해 조작 모드에 따라 동작을 예측합니다. 이 프레임워크는 의류 접기와 같은 시뮬레이션 작업에 적용되며, 3D 위치와 사전 훈련된 변형 인식 의미적 특징을 결합한 geo-semantic token으로 물체를 표현합니다.

- **Performance Highlights**: Instant-Fold는 실제 데이터를 추가 수집하거나 미세 조정(finetuning) 없이도 다양한 folding 모드에 일반화하며 실세계 환경에서도 성능을 발휘합니다. 이는 기존의 복잡한 조작 절차를 단순화하고, 하나의 시연만으로도 새로운 동작을 효과적으로 학습할 수 있도록 합니다. 이 연구는 수많은 유사한 연구와 차별화되는 방향으로 DOM 분야에서의 새로운 가능성을 제시하고 있습니다.



### Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA (https://arxiv.org/abs/2606.04262)
Comments:
          16 pages, 7 figures

- **What's New**: 이 연구에서는 DOSEBENCH라는 새로운 벤치마크를 도입하여 OTC 약물인 아세트아미노펜과 이부프로펜에 대한 81개 시나리오를 평가합니다. 이 벤치마크는 사용자가 안전하게 또 다른 복용을 할 수 있는지에 대한 건강 질문을 다루며, 최근 LLM(대형 언어 모델)의 성능을 평가할 수 있도록 설계되었습니다. 기존의 의료 질문-응답 벤치마크는 임상 지식이나 일반적인 의료 추론에 중점을 두었지만, DOSEBENCH는 OTC 복용 결정을 중심으로 하여 보다 안전 관련한 문제에 초점을 맞춥니다.

- **Technical Details**: DOSEBENCH는 아세트아미노펜과 이부프로펜 사용에 관한 실제 성인 소비자 질문을 기반으로 한 OTC 복용 시나리오로 구성되어 있습니다. 연구에서는 네 가지 LLM을 평가하였고, 의사 결정의 정확성, 일관성, 설명 가능성, 실패 패턴 및 신뢰도 관련 행동을 측정했습니다. 평가 결과, 모델이 종종 시간적 추론(temporal reasoning) 및 애매모호한 경우(ambiguity-sensitive cases)에서 어려움을 겪으며, 안정적이거나 신뢰감 있는 답변이 복용 제한을 위반할 수 있음을 발견했습니다.

- **Performance Highlights**: DOSEBENCH의 결과는 현재의 LLM이 소비자 지향적인 OTC 복용 결정을 내리는 데 있어 신뢰감을 주지만, 여전히 문제를 야기할 수 있음을 보여줍니다. 안전 관련 복용 규제를 준수하지 않는 잘못된 사례가 많아, 이 벤치마크는 LLM의 안전 지향적 신뢰성(safety-oriented reliability)을 평가하는 데 유용한 테스트베드(testbed)를 제공함을 시사합니다. 이러한 발견은 OTC 복용 QA가 시간적 추론, 제약 준수 및 안전 관련 불확실성 처리와 같은 중요한 요소를 평가하는 데 효과적임을 보여줍니다.



### Overview of the EReL@MIR 2025 Multimodal Document Retrieval Challenge (Track 1) (https://arxiv.org/abs/2606.04240)
Comments:
          MDR Challenge Report at WWW2025

- **What's New**: 이번 논문은 2025년 Web Conference와 공동으로 개최된 첫 EReL@MIR 워크샵에서 진행된 Multimodal Document Retrieval Challenge의 트랙 1에 대해 소개합니다. 이 챌린지는 하나의 시스템을 통해 긴 문서 내에서의 closed-set retrieval과 이미지 또는 이미지-텍스트 쿼리를 통한 open-domain retrieval을 처리해야 하는 두 가지 작업으로 구성됩니다. 455명의 참가자와 586개의 제출물이 있었으며, 시스템들은 평균 Recall@{1,3,5}에 따라 순위가 매겨졌습니다.

- **Technical Details**: 챌린지는 두 가지 작업을 위해 설계되었으며, Task 1인 MMDocIR은 텍스트 쿼리에 따라 단일 긴 문서의 관련 페이지를 순위 매기는 것이고, Task 2인 M2KR은 이미지나 이미지-텍스트 쿼리를 가진 단일 global corpus에서 관련 위키피디아 스타일의 구문을 검색하는 것입니다. 모든 시스템은 Qwen2-VL 계열의 decoder 기반 멀티모달 LLM embedder를 사용하였으며, 우리가 알아본 세 가지 우승 시스템은 서로 다른 방식으로 성능을 높였습니다.

- **Performance Highlights**: 이번 챌린지에서 우승한 세 팀의 시스템은 fine-tuned ensemble, 강력한 vision-language re-ranker를 이용한 training-free multi-route fusion, 또는 zero-shot late interaction을 통해 최고의 성과를 올렸습니다. 특히, training-free 시스템은 fine-tuned 승자와 0.1점 차이로 마무리되었습니다. 이 결과들은 멀티모달 정보 검색의 발전을 위한 중요한 교훈을 제공합니다.



### Recover-LoRA for Aggressive Quantization: Reclaiming Accuracy in 2-Bit Language Models via Low-Rank Adaptation with Knowledge Distillation on Synthetic Data (https://arxiv.org/abs/2606.04238)
- **What's New**: 이 연구는 2비트 정밀도의 가중치 양자화(weight quantization)에 따른 정확도 저하 문제를 Recover-LoRA라는 경량화된 데이터 없는 정확도 회복 방법을 통해 해결하는 방안을 제시합니다. 특히, MLP의 게이트와 업 프로젝션 층만을 2비트로 양자화하고 나머지 층을 높은 정밀도로 유지하는 혼합 정밀도(mixed-precision) 전략을 통해 대규모 언어 모델의 처리 속도를 크게 개선합니다. 이러한 설정은 메모리 용량과 대역폭 제약이 있는 엣지 및 장치 배포에 특히 중요합니다.

- **Technical Details**: 연구에서는 게이트 및 업 프로젝션 층의 2비트 양자화를 통해 처리량을 7.5%에서 23.3%까지 개선하며, 양자화 오류는 예측 가능한 특정 층으로 국한됩니다. Recover-LoRA는 2비트 양자화로 인해 손실된 정확도를 회복하기 위해 로짓 증류(logit distillation) 기술을 적용하여 훈련 중인 저순위 적응기(low-rank adapter)를 사용합니다. 이 과정은 레이블 데이터 없이도 가능하고, 10,000개의 합성 훈련 샘플만으로도 높은 정확도 회복을 달성합니다.

- **Performance Highlights**: Qwen3-4B를 사례로 들어, Recover-LoRA는 12개 벤치마크 중 9개에서 80%에서 95%의 정확도 회복률을 기록했습니다. 이 연구 결과는 Recover-LoRA가 가중치 압축을 동시에 달성할 수 있는 실용적인 포스트 양자화 정확도 회복 도구로서의 가능성을 보여줍니다. 또한 합성 데이터는 수집된 레이블 데이터에 비해 비슷한 성능을 보였으며, 이는 자기 훈련이나 QAT가 꼭 필요하지 않음을 입증합니다.



### Supportive Token Revealing for Fast Diffusion Language Model Decoding (https://arxiv.org/abs/2606.04236)
- **What's New**: 본 논문에서는 AXON이라는 새로운 모듈을 제안합니다. AXON은 기존의 parallel decoding 전략에 추가하여 사용할 수 있는 training-free 모듈로, 마스크된 토큰에 대해 신뢰도 및 의존성을 기반으로 정보가 풍부한 토큰을 선택하여 추가적인 문맥을 제공합니다. 이를 통해 디퓨전 언어 모델의 품질-지연(time latency) 트레이드오프를 개선할 수 있습니다.

- **Technical Details**: AXON은 마스크된 토큰들을 모니터링하고 필요할 때만 개입하는 방식으로 동작합니다. 이 모듈은 유용한 문맥을 제공할 수 있는 확신 있는 마스크된 토큰을 선택하기 위해 주의(attention), 불확실성(uncertainty), 신뢰도(confidence) 신호를 사용하여 후보 토큰에 점수를 매깁니다. AXON의 선택 과정은 서브모듈러(submodular) 함수 목표를 따릅니다.

- **Performance Highlights**: 다수의 디퓨전 언어 모델과 패러럴 디코더를 대상으로 한 실험에서 AXON이 품질-지연 트레이드오프를 개선하는 데 성공했음을 보여주었습니다. AXON은 기능 평가 횟수를 줄이면서 정확도를 유지하거나 개선하여 기존의 강력한 기준선보다 나은 성능을 발휘합니다.



### MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A (https://arxiv.org/abs/2606.04231)
Comments:
          Accepted at ACL 2026 (Industry Track)

- **What's New**: 본 논문은 MM-RAG(다중 모달 검색 보강 생성)의 최근 발전에 대한 논의에서 출발한다. 기존 방법들이 복잡한 기업 문서의 구조적 정보를 명시적으로 처리하지 않고, 사전 훈련된 임베딩 모델에 의존한 데 반해, 본 연구에서는 문서 구조 인식 스플릿을 통해 문서 구조를 능동적으로 추출하고 표현하는 방식을 제안한다. 이를 통해 MM-BizRAG는 다양한 문서 유형에 대해 최적화된 파이프라인을 구현하며, 수직 구조와 수평 구조에 각각 적합한 처리를 통해 효율성을 높인다.

- **Technical Details**: MM-BizRAG는 문서 처리에 있어 독특한 접근 방식을 채택한다. 각 문서는 LLM 기반 분류기를 통해 수직 구조(V)와 수평 구조(H)로 구분된 후, 레이아웃 인식 파싱(layout-aware parsing)이 적용된다. 이 과정은 페이지 내 텍스트 블록, 테이블 및 그림을 정렬된 형태로 추출하여 자연스러운 독서 순서를 유지하는 동시에 재구성된 표현으로 변환한다.

- **Performance Highlights**: MM-BizRAG는 대규모의 이질적인 기업 데이터셋과 SlideVQA, FinRAGBench-V 같은 공공 벤치마크에서 기존 비전 중심 기법들보다 최대 32% 더 높은 성능을 보여준다. 연구진은 FastRAGEval이라는 새로운 메트릭을 도입하여 생성적 회수의 정밀도를 높이면서도 비용을 절반으로 줄이는 효과를 얻었다. 이러한 결과는 재활용하지 않고도 풍부한 맥락 기반 생성을 가능하게 하여 다중 모달 생성을 한층 강화한다.



### Incremental Sheaf Cohomology on Cellular Complexes: O(1)-in-n Lazy Edit Processing under Bounded Local Geometry (https://arxiv.org/abs/2606.04227)
Comments:
          2 figures, 2 tables, 1 algorithm; code at this https URL

- **What's New**: 본 연구에서는 동적으로 변화하는 1차원 세포 복합체에서 첫 번째 시프(cohomology) $H^1(X; \mathcal{F})$의 점진적 유지(incremental maintenance)를 위한 알고리즘 프레임워크를 제시합니다. 기존의 $H^1$ 계산 방식은 $O(n^3)$의 시간 복잡도를 요구하며, 각 수정(edit) 후 전체 재계산(full recomputation) 비용은 $O(mn^3)$에 달합니다. 본 알고리즘은 국소 기하학(local geometry) 가정 하에 각 수정이 제한된 국소 coboundary 블록에만 영향을 미치도록 하여, 복합체 크기 $n$에 대해 $O(1)$의 시간 복잡도로 lazy streaming edits를 처리할 수 있게 합니다.

- **Technical Details**: 알고리즘의 주요 기여로는 Locality Lemma, Incremental Maintenance Theorem, Zero-Drift Theorem 등이 있습니다. Locality Lemma에서는 상한이 있는 세포 크기($v_{max}$), 줄기 차원($d$), 신경 차수($D$)하에서 단일 수정이 최대 2개의 국소 데이터에만 영향을 준다는 것을 증명합니다. Incremental Maintenance Theorem에서는 각 수정이 $O(v_{max}^{3} \cdot d^{3})$의 시간 내에 처리될 수 있음을 보이며, 이는 실제로 상수로 취급되는 경우 $O(1)$로 간주됩니다. Zero-Drift Theorem은 유지된 $H^1$이 동기화 지점에서 배치 조립(batch assembly)과 일치함을 증명하여, 측정된 드리프트가 발생하지 않음을 보여줍니다.

- **Performance Highlights**: 실험에서는 최대 $5 \times 10^{6}$개의 정점과 $1.7 \times 10^{7}$개의 수정이 포함된 Barabasi-Albert 그래프에서 성능을 평가하였습니다. lazy per-edit 업데이트 지연시간은 35 $us$로 측정되었으며, 동기화 지점에서 제로 드리프트가 확인되었습니다. 문헌에서의 다른 동적 알고리즘들과 비교했을 때, 본 알고리즘은 시프를 유지하는 데 있어 효율성을 제공하며, 필터링 구조를 통한 효율적인 업데이트가 가능함을 확인하였습니다.



### PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification (https://arxiv.org/abs/2606.04226)
Comments:
          Accepted at ICRA 2026 (Vienna); published on arxiv for archival purposes. See also this https URL

- **What's New**: 이번 연구에서는 PerceptTwin이라는 자동화된 파이프라인을 소개하며, 로봇의 인식 스택에서 생성된 의미 장면 표현을 기반으로 상호작용 가능한 시뮬레이션을 구축합니다. 이는 기존의 수동적인 시뮬레이션 생성 과정의 대안을 제시하며, 로봇 계획의 검증과 수정에 있어 큰 잠재력을 가지고 있습니다. 특히 PerceptTwin은 개방형 어휘(Open-Vocabulary) 객체 맵, 3D 자산 생성, 그리고 일반적인 조건 검사를 통합합니다.

- **Technical Details**: PerceptTwin은 3D 자산을 생성하거나 찾아내고, 인식된 객체 포인트 클라우드를 사용하여 이들을 지역화합니다. 또한 로봇-객체 친화성(affordance)을 예측하고 시뮬레이션 내에서 계획을 테스트할 수 있는 기능을 가지고 있습니다. 연구에서는 LLM(대형 언어 모델)을 활용하여 계획의 정확성을 검증하는 LLM 판별기를 도입하여, 계획이 인간의 선호와 일치하는지 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면, PerceptTwin을 통한 피드백은 계획의 성공률을 평균 39% 향상시키고, 인간의 계획 검증 정확성을 평균 18% 개선하는 것으로 나타났습니다. 이는 PerceptTwin이 로봇 계획의 안전성과 신뢰성을 높일 수 있는 기초 역할을 할 것으로 기대됩니다. 코드 및 자료는 오픈 소스로 제공되어, 연구자들이 활용할 수 있습니다.



### DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities (https://arxiv.org/abs/2606.04205)
- **What's New**: DetectZoo는 AI 생성 콘텐츠 탐지를 위한 최초의 다기능 툴킷으로, 텍스트, 이미지 및 오디오 모달리티에 대한 통합 인터페이스를 제공합니다. 이 툴킷은 데이터 수집 및 전처리부터 모델 평가에 이르는 전체 실증적 파이프라인을 표준화하여 연구자들이 최신 탐지기를 체계적으로 벤치마킹할 수 있게 합니다. 61개의 탐지기를 위한 참조 구현, 22개의 벤치마크 데이터셋을 위한 네이티브 로더 및 공통 인터페이스를 통해 여러 메트릭을 보고하는 표준화된 평가 파이프라인도 포함되어 있습니다.

- **Technical Details**: DetectZoo는 모달리티 간의 비교를 용이하게 하기 위해 61개의 탐지 방법을 단일 코드베이스에 통합하고 22개의 벤치마크 데이터셋 및 표준화된 평가 파이프라인과 함께 제공됩니다. 통합된 모달리티의 접근 방식을 통해 연구자들은 반복 가능한 비교를 실시할 수 있으며, 이는 AI 생성 콘텐츠 탐지의 효과적인 연구 환경을 조성합니다. 각 탐지기는 독립적이며 동일한 인터페이스를 통해 접근 가능하며, 사전 학습된 가중치를 자동으로 캐시하여 원래 발표된 결과를 재현합니다.

- **Performance Highlights**: DetectZoo는 다중 모달 AI 포렌식의 진입 장벽을 낮춰 연구자들이 도메인 간 성능 차이를 식별하고 강력하고 일반화 가능한 탐지 기법의 개발을 가속화할 수 있도록 합니다. 이는 AI 생성 콘텐츠 탐지의 연구 성과를 더욱 효과적으로 발전시키고, 개별 논문 코드베이스에 대한 의존도를 줄이는데 기여합니다. 모든 구성 요소는 공개적으로 사용 가능하며, pip를 통해 쉽게 설치할 수 있어 연구자들이 접근할 수 있는 유용한 리소스가 됩니다.



### Notarized Agents: Receiver-Attested Confidential Receipts for AI Agent Actions (https://arxiv.org/abs/2606.04193)
Comments:
          22 pages. Reference implementation at this https URL

- **What's New**: 본 논문은 AI 에이전트의 관찰 가능성(agent observability)의 구조적 문제를 해결하기 위한 프로토콜의 새로운 클래스를 제안합니다. 현존하는 시스템에서는 에이전트가 자신의 행동을 기록할 때 발생할 수 있는 변조(tampering) 및 조작 위험이 존재합니다. 제안된 프로토콜인 Sello는 에이전트의 운영자나 이 에이전트 자체에 의존하지 않고도 신뢰할 수 있는 로그를 생성할 수 있는 방법을 제공합니다.

- **Technical Details**: Sello 프로토콜은 네 가지 주요 속성을 바탕으로 구성됩니다: (P1) 수신자 측 서명, (P2) JWS를 통한 권한 부여 토큰과 연결된 소유자 공개키에 대한 HPKE 암호화, (P3) 증인 공동 서명 Merkle 로그에 대한 게시, (P4) 토큰 참조에 의한 소유자 측 발견. 각 속성은 기존 시스템에서 찾아볼 수 없는 독특한 조합을 이루고 있습니다. 이러한 구조적 전환을 통해 에이전트가 자신의 추적 기록을 위조할 수 없도록 설계되었습니다.

- **Performance Highlights**: Sello는 다양한 기존 영수증 프로토콜(Signet, AgentROA 등)과 비교하여 모든 요구 사항을 충족시키는 유일한 시스템임을 강조합니다. 그러나 이 논문은 Sello가 해결하지 못하는 한계도 명시하며, 여기에는 억제 공격(suppression attack), 서비스 간의 공모(collusion), 영수증 발행 인센티브 문제 등의 이슈가 포함됩니다. 이러한 논의는 Sello의 실용적인 응용과 향후 연구 방향에 대한 중요한 통찰을 제공합니다.



### Metric-Aware Hybrid Forecasting for the CTF4Science Lorenz Challeng (https://arxiv.org/abs/2606.04191)
- **What's New**: 본 연구에서는 CTF4Science Lorenz 챌린지에 대한 접근 방식을 설명하고 있습니다. 이 벤치마크는 짧은 시간 예측(short-horizon forecasting), 긴 시간 분포 일치(long-time distribution matching), 그리고 궤적 재구성(trajectory reconstruction) 등 서로 다른 여덟 가지 작업 쌍으로 구성되어 있습니다. 단일 모델이 모든 메트릭을 지배하는 것이 아니라, 우리는 메트릭 인식 메트릭 기계식 시스템을 구축하여 각 메트릭에 다른 예측기를 할당했습니다.

- **Technical Details**: 우리의 방법은 클래식 신호 처리(classical signal processing), 신경망 제거(Neural denoising), 물리 기반의 Lorenz 피팅을 결합하여 메트릭 영역별로 제출물을 구성하는 시스템 문제로 발전했습니다. 단순 스무딩이 재구성에 더 유리하다는 걸 인지하고, 짧은 수평 작업을 위한 직접적인 Lorenz 궤적 촬영(trajectory shooting)을 활용했습니다. 또한 긴 시간 메트릭은 시간 순서를 무시하고 최종 500행에서 주변 히스토그램만 평가한다는 사실을 이용했습니다.

- **Performance Highlights**: 최종 제출물은 공공 리더보드에서 83.83551점을 기록했습니다. 이후 같은 아이디어를 기반으로 한 작은 후속 스택이 83.85529점을 달성했습니다. 이 연구의 핵심 제출물은 전체 방식을 포착하면서도 간단하여 재현 가능하고 분석 가능하다는 점에서 중요한 의미를 지닙니다.



### Dual Advantage Fields (https://arxiv.org/abs/2606.04188)
Comments:
          Accepted by ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning

- **What's New**: 이번 연구에서는 오프라인 목표 조건 강화 학습에서 차량을 많이 교체하는 문제를 다룬다. Dual Advantage Fields (DAF)라는 정책 추출 방법을 도입하여, 이 방법이 어떻게 전역 가치 필드를 지역적 이점 신호로 변환하는지를 설명한다. Dual goal representations을 활용하여, 목표 방향과 일치하는 행동을 평가하는 새로운 방법론을 제시한다.

- **Technical Details**: DAF는 동작 효과 모델을 학습하고 행동 선택을 위한 정량적인 평가 지표를 생산하여, 기하학적 테스트를 통해 지역적인 이점을 계산한다. 이는 상태 표현의 변형으로 나타나며, 목표의 방향과 정렬되는 행동을 선호하게 된다. 이 방법은 고정된 오프라인 데이터에서 정책을 추출하며, 별도의 목표 조건 행동 가치 함수 학습 없이 동작한다.

- **Performance Highlights**: 실험 결과, DAF는 OGBench에서 운동, 조작, 퍼즐 작업을 수행하는 데 있어 성능이 향상되었다. 지역적으로 올바른 행동이 목표로의 직접적인 이동과는 차별화되는 환경 설정에서 강력한 성과를 보였다. DAF는 오프라인 GCRL(Goal-Conditioned Reinforcement Learning)에서 일관된 성능을 제공하며, 다양한 도메인에 대해 효과적으로 작동한다.



### Exact Unlearning in Reinforcement Learning (https://arxiv.org/abs/2606.04182)
Comments:
          ICML Spotlight

- **What's New**: 이 논문은 강화学习(Reinforcement Learning, RL)에서의 정확한 기계 학습 제거(exact unlearning) 문제를 규명합니다. 이는 사용자의 데이터 삭제 요청이 있을 경우 이를 효율적으로 제거할 수 있도록 설계된 프레임워크를 제안합니다. 삭제된 사용자의 데이터가 모델에 영향을 미치지 않는 결과를 생성하는 것이 본 연구의 주요 목표입니다.

- **Technical Details**: 이 연구에서는 $ho$-TV 안정성($ho$-TV stability)을 갖춘 보통 일반화된 마르코프 결정 프로세스(MDPs)에 대한 강화 학습 알고리즘을 구성합니다. 이 알고리즘은 예상 컴퓨팅 비용이 재훈련의 $ho 	imes 	ext{ln}(T)$ 분의 $ho$에 해당하며, 구체적으로 고찰한 경우, 보상 효용을 최대화하는 경과를 모델링합니다.

- **Performance Highlights**: 논문에서 제안된 알고리즘은 보상 재훈련에서 필요한 비용을 대폭 줄이며, 저자들은 본 알고리즘의 후한 경계(Regret bound)가 $	ilde{	ext{O}}(H^2 	ext{SAT} + H^3 S^2 A + rac{H^{2.5} S^2 A}{ho})$ 인 것을 밝혔습니다. 이는 다른 $ho$-TV 안정적인 RL 알고리즘에 대해 최소 민맥스 최적(minimax optimal)을 달성함을 의미합니다.



### A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models (https://arxiv.org/abs/2606.04177)
Comments:
          preprint

- **What's New**: 이번 연구는 AI 생성 텍스트를 설명하기 위한 해석 가능한 언어적 특성(interpretable linguistic features)의 신뢰성을 평가하는 대규모 실증 연구입니다. 27개의 LLM과 10개의 텍스트 도메인에서 284개의 언어적 특성을 분석하여, 비전문가 사용자도 이해할 수 있도록 기계 생성 텍스트를 구분할 수 있는 신뢰할 수 있는 신호를 제시합니다.

- **Technical Details**: 연구에서는 다양한 모델(27 LLM)과 도메인(10 text domains)에서 생성된 텍스트의 언어적 특성을 284개에 걸쳐 분석하였으며, 이를 통해 범모델(cross-model) 및 범도메인(cross-domain) 일반화(settings) 하에서 신호의 강력함을 평가했습니다. 분석을 통해, 언어적 특성만으로도 AI 생성 텍스트와 인간이 작성한 텍스트를 구분할 수 있는 기준(classifiers)을 확립하였습니다.

- **Performance Highlights**: 결과적으로, 많은 기존의 지표(indicators)는 특정 문맥(context)에 강하게 의존하는 반면, 어휘 풍부성(leixcal richness) 측정은 다양한 모델 계열과 텍스트 도메인 전반에 걸쳐 강력한 신뢰성을 유지함을 보여주었습니다. 이러한 결과는 AI 생성 언어에 대한 더 신뢰할 수 있고 해석 가능한 분석의 기초를 제공합니다.



### MimeLens: Position-Agnostic Content-Type Detection for Binary Fragments (https://arxiv.org/abs/2606.04171)
Comments:
          18 pages, 2 figures, 15 tables. Models released on Hugging Face (this https URL reference training code at this https URL

- **What's New**: 본 논문은 MimeLens라는 새롭게 개발된 BERT 스타일의 인코더 패밀리를 소개합니다. 기존의 파일 타입 분류기들과는 달리, MimeLens는 파일의 임의의 오프셋에서 임의의 바이트 청크를 입력으로 받아 파일 콘텐츠를 분류할 수 있는 능력을 갖추고 있습니다. MimeLens는 libmagic 분류 체계의 125개 MIME 레이블을 기반으로 하며, 전체 파일이나 고정된 크기의 입력이 필요하지 않습니다.

- **Technical Details**: MimeLens는 선행 학습된 인코더를 활용하여 다른 파일 타입 분류 시스템과의 차별점을 둡니다. 이 모델은 개별 파일 내에서 균일하게 무작위 오프셋에서 샘플링된 바이트 조각에 기반하여 훈련됩니다. 이를 통해, 완전한 파일의 머리 부분뿐 아니라 데이터 스트림 중간이나 헤더가 없는 조각 등 다양한 형태와 크기의 입력을 처리할 수 있습니다.

- **Performance Highlights**: MimeLens는 libmagic 레이블 데이터에서 Magika v1.1보다 +10.7%의 정확도로 우수한 성능을 보여주었으며, 특히 UDP 패킷과 무작위로 선택된 디스크 블록에서 두 배 이상의 정확도로 분류할 수 있습니다. 그러나 MimeLens의 CPU에서의 샘플 처리 속도는 Magika보다 1~2배 느립니다; 반면, 소비자 GPU 또는 배치 작업에서는 Magika와 비슷한 성능을 보입니다.



### Smart Transportation Without Neurons -- Fair Metro Network Expansion with Tabular Reinforcement Learning (https://arxiv.org/abs/2606.04167)
Comments:
          16 pages

- **What's New**: 이 논문은 Metro Network Expansion Problem (MNEP)을 다루며, 이는 Transport Network Design Problem (TNDP)의 한 부분으로 메트로 시스템을 확장하여 여행 수요를 충족시키는 데 집중합니다. 기존의 방법들은 전문가 정의 제약을 사용하여 검색 공간을 줄이는 정확한 방식과 휴리스틱 접근법을 필요로 했습니다. 하지만 본 연구에서는 복잡한 순차 결정 과정에서 효과적인 Deep Reinforcement Learning (Deep RL) 대신 MNEP가 충분히 작은 문제임을 보여주고, 이를 Non-Markovian Rewards Decision Process (NMRDP)로 재정의하여 탭룰 RL을 사용하여 유사한 성과를 도출합니다.

- **Technical Details**: MNEP를 해결하기 위해 전통적인 RL 방법을 재정형화하여 탭룰 접근 방식을 통해 훈련 에피소드를 크게 줄이면서도 경쟁력 있는 성과를 달성할 수 있음을 보여줍니다. 논문은 사회적 공정성을 고려한 다양한 보상 함수를 포함하여, 효율성과 공정성을 모두 충족하는 방식으로 문제를 접근합니다. 또한 두 개의 실제 환경인 시안(Xi’an)과 암스테르담(Amsterdam)에서 실행하여 훈련 에피소드를 18배, CO2 배출량을 12배 줄이면서도 Deep RL과 경쟁할 수 있는 성능을 입증합니다.

- **Performance Highlights**: 제안하는 탭룰 RL 접근 방식은 백박스 형태의 Deep RL 모델보다 해석 가능성이 높으며, 자원 소모가 적은 효율적인 솔루션을 제공합니다. 연구 결과는 기존 MNEP 문제 해결에 있어 코드를 포함한 데이터셋 및 하이퍼파라미터 설정을 제공하여, 다른 조합 최적화 문제에 적용할 수 있는 가능성을 제시합니다. 이런 접근 방식은 MNEP의 효율성과 공정성을 동시에 충족시키기 위해 설계되어 있으며, 정확한 정보를 원하는 결정권자에게 중요한 해석성을 제공합니다.



### ADAPTOOD: Uncertainty-Aware Fine-Tuning for Out-of-Distribution ECG Time Series Models (https://arxiv.org/abs/2606.04164)
Comments:
          11 pages

- **What's New**: 이 논문에서는 ADAPTOOD라는 새로운 프레임워크를 제안하여 분포 차이의 심각성을 정량화하고 이를 활용하여 시간 데이터 시리즈에 대한 세밀한 조정을 유도합니다. 특히, 이 프레임워크는 새로운 데이터 입력과 훈련 분포 간의 편차를 반영하는 데이터 불확실성을 활용하여 OOD(Out-of-Distribution) 데이터를 처리합니다. 이러한 접근 방식은 기존의 적응 방법들이 고정된 가정을 사용하여 세부 수준을 간과한 문제를 해결합니다.

- **Technical Details**: ADAPTOOD는 데이터 불확실성을 활용하여 분포 차이의 심각성을 파악하고, 이를 바탕으로 모델의 세밀한 조정을 안내합니다. 시스템은 새로운 입력으로부터 학습하는 강도를 조절하며, 저랭크(low-rank) 모델 업데이트 및 적응 하이퍼파라미터 최적화를 결합하여 세밀한 조정을 보다 효율적으로 수행합니다. 이 방법은 신뢰할 수 있는 타깃 배포 분포와 훈련 분포 간의 차이를 평가하여 조정을 단순화합니다.

- **Performance Highlights**: ADAPTOOD는 OOD 작업에서 기존 방법들보다 최대 7% 높은 정확도와 12.9% 높은 정밀도를 달성하며, 분포 차이의 심각성이 증가해도 강력한 성능을 유지합니다. 또한, 이 프레임워크는 모든 메트릭에서 일관된 성능을 보여주어, 신뢰할 수 있는 바이오기록 해석 모델 개발에 중요한 기여를 합니다.



### EvalStop: Using World Feedback to Detect and Correct Reward Overoptimization in Multi-Tenant RLHF Platforms (https://arxiv.org/abs/2606.04145)
- **What's New**: 이번 논문에서는 EvalStop이라는 새로운 스케줄링 프리미티브를 제안하여 RLHF(강화 학습을 통한 인간 피드백) 워크로드에서 성능을 극대화하는 방법을 탐구합니다. 기존의 스케줄러들은 기존 피드백과 품질 신호를 활용하지 못하고 있으며, EvalStop은 연속적으로 감소하는 평가 점수에 기초하여 작업을 종료시키는 방식으로 자원을 효율적으로 관리합니다. 이 연구는 RLHF 최적화 압박 아래에서의 보상 과최적화(reward overoptimization) 문제를 해결하기 위한 중요한 기여를 합니다.

- **Technical Details**: EvalStop은 평가 점수 감소를 기반으로 한 중단 감지 시스템으로, 각 작업의 평가 점수를 지속적으로 추적하고, k개의 연속적인 감소가 발생하면 작업을 종료하고 최상의 체크포인트를 저장합니다. 이 시스템은 다양한 스케줄러와 함께 작동할 수 있으며, RLHF 작업의 GPU 자원 할당을 최적화합니다. 실험 결과, EvalStop은 80%의 RLHF 워크로드에서 98%의 정밀도(precision)와 99%의 재현율(recall), 1.5%의 잘못 탐지율(FPR)을 기록하여 SRTF-Est와 비교해 9%의 JCT 개선과 22%의 낭비된 계산 감소를 보여주었습니다.

- **Performance Highlights**: EvalStop의 성능은 모든 테스트된 기본 스케줄러에 대해 9-25%의 JCT 개선을 달성하였으며, 평가 노이즈와 해킹 비율이 변경되더라도 안정적인 탐지 품질을 유지했습니다. 구체적으로, 노이즈 표준편차가 0.05 이하일 때 정밀도는 최소 91%에서 유지되었으며, 해킹 비율이 20–80%일 때도 최소 89%의 정밀도로 성능이 확인되었습니다. 따라서 EvalStop은 RLHF 워크로드에서 효율적인 자원 사용과 뛰어난 성능을 동시에 달성할 수 있는 강력한 도구입니다.



### Physics-Informed Machine Learning for Short-Term Flood Prediction (https://arxiv.org/abs/2606.04143)
Comments:
          This paper has been accepted for publication in IGARSS 2026. The final authenticated version will be available through IEEE Xplore

- **What's New**: 본 연구는 물리적 정보를 머신러닝에 통합한 Physics-Informed Machine Learning(PIML) 프레임워크를 제안합니다. 이 프레임워크는 LSTM 모델의 손실 함수에 수문학적 지식을 직접 포함하여 모델의 예측 정확도를 향상시킵니다. 특히, 강수량과 방류 트렌드 간의 일관성을 보장하기 위해 'Trend Alignment' 제약을 도입하여 데이터가 부족한 환경에서도 모델의 안정성을 높입니다.

- **Technical Details**: 제안된 방법은 표준 LSTM 모델을 사용하여 강화되는 방식으로, LSTM의 손실 함수에 물리적 일관성을 확보하기 위한 제약 조건을 추가합니다. 이 모델은 강수량이 증가하면 방류가 증가해야 한다는 Trend Alignment 제약을 포함하며, 불규칙한 변화를 억제하는 Temporal Smoothness 제약도 적용합니다. 물리적 제약을 통해 모델이 물리적으로 타당한 예측을 학습하도록 유도하며, 이를 통해 예측의 신뢰성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 PIML 모델은 데이터가 부족한 환경에서 표준 LSTM 모델보다 더 높은 성능을 보였습니다. 특히, Nash-Sutcliffe Efficiency(NSE)는 기존 모델의 0.20에서 0.23으로 증가하며, 복잡한 수리 유체 역학적 방정식 없이도 물리적으로 일관성 있는 예측을 가능하게 했습니다. 극한 기후 시나리오에서도 기본 모델이 불안정한 행동을 보이는 반면, 물리정보가 포함된 모델은 방향 일관성을 유지하여 신뢰성을 더욱 강화했습니다.



### Caught in the Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by LLM Agents (https://arxiv.org/abs/2606.04141)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 에이전트가 신뢰할 수 없는 컨텐츠와 신뢰할 수 있는 자격 증명 정보를 동일한 컨텍스트 창에 저장하는 위험성을 연구합니다. 저자들은 사전 출력 모니터링, 캘리브레이션된 헌터니 토큰 감지 및 누적 누출 회계의 세 가지 방어 방법을 제안하여 자격 증명 누출을 방지하고자 합니다. 이 연구는 'Agentic Immune System (AIS)'라는 프로토타입을 개발하여 이러한 방어 방법들을 구현하고 평가합니다.

- **Technical Details**: 연구에서는 활성화 프로브(activation probes)를 사용하여 에이전트가 자격 증명을 요청하는 행위를 출력 토큰이 생성되기 전에 감지할 수 있는지 검토했습니다. 헌터니 토큰(honeytoken) 생성과 감지 과정은 차별적으로 개인 정보 보호 모델을 통해 통계적으로 엄밀하게 수행되며, 다중 턴 대화 내의 누적 누출을 모니터링하는 새로운 프로토타입이 제안되었습니다. 저자들은 이 시스템이 실제로 완전한 방어 기법은 아니지만, 연구 방향을 제시하고 설계 상의 트레이드오프를 드러내는 데 목표를 두고 있다고 강조합니다.

- **Performance Highlights**: 모델의 활성화 기능을 이용한 감지는 안전한 프롬프트와 자격 증명 요청 프롬프트를 높은 정확도로 구별할 수 있도록 지원합니다. 작은 다중 턴 실험에서는 누적 회계가 단일 턴 감지기가 놓친 공격을 탐지하는 데 효과적임을 보여주었습니다. 이러한 결과는 자격 증명 누출 방어에 있어 출력 텍스트 레벨 필터에만 의존하기보다는 보다 종합적인 접근 방식의 필요성을 강조합니다.



### HighTide: An Agent-Curated Open-Source VLSI Benchmark Su (https://arxiv.org/abs/2606.04126)
- **What's New**: HighTide는 진화하는 AI 보조 벤치마크 스위트로, 다양한 설계 언어와 기술 노드를 포괄하는 오픈 소스 스위트를 제공합니다. 이 스위트는 Bazel 기반의 점진적 RTL-to-GDS 컴파일과 원격 캐싱을 지원하며, 설계 라이프사이클 전반을 아우르는 12가지 에이전트 기술을 통한 AI 보조 설계 큐레이션 기능도 포함되어 있습니다. 각 설계의 의사결정 로그로 튜닝 근거를 장기적으로 저장할 수 있는 기능도 제공하여, 오픈 소스 하드웨어 생태계와 함께 발전할 수 있도록 고안되었습니다.

- **Technical Details**: HighTide는 RISC-V 아키텍처 중심의 기존 벤치마크 스위트의 한계를 극복하고, 다양한 설계 축을 가진 프로세서를 포함하여 총체적이고 동적으로 발전할 수 있는 환경을 제공합니다. 이 스위트는 Verilog, SystemVerilog, Chisel 및 두 개의 Python 기반 생성기를 포괄하는 언어의 범위를 가지고 있으며, 셀 카운트 범위는 20,000 미만에서 1.5백만 이상까지 다양합니다. 각 설계는 덜 대표적인 전통적인 하드웨어 디자인에 의존하지 않고, 점진적으로 증가하는 설계의 복잡성을 반영합니다.

- **Performance Highlights**: HighTide는 고차원 언어부터 RTL-to-GDS까지 여러 설계를 아우르며, 최신 하드웨어 기술 발전을 반영하여 벤치마크 스위트를 제공합니다. 이 스위트는 설계의 변화가 자주 일어나는 오픈 소스 커뮤니티와 함께 진화하며, 기계 학습(ML) 모델에 보다 적합한 교육 데이터를 제공합니다. 다양한 구조의 설계를 통해 ML 및 EDA 개발자들이 보다 신뢰할 수 있는 결과를 도출할 수 있도록 지원합니다.



### Semantic Constraint Synthesis for Adaptive Trajectory Optimization via Large Language Models (https://arxiv.org/abs/2606.04123)
Comments:
          7 pages, 4 figures, Presented as a short paper at IEEE CVPR 2026, AI4Space Workshop

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)을 활용하여 자연어로 표현된 임무 요구사항 및 제약조건을 실행 가능한 궤적 최적화 코드와 해당 수학적 공식으로 변환하는 새로운 프레임워크를 제안합니다. 이는 우주 임무의 증가하는 빈도와 복잡성으로 인해 효율적인 궤적 최적화 문제의 수립이 필요함을 강조합니다. LLM을 통해 임무 의도를 높은 수준에서 설명하고, 이를 정형화된 최적화 모델로 연결할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 이 연구는 궤적 최적화 문제를 해결하기 위해 자연어 임무 요구사항을 정형적 수학 공식으로 변환하는 자동화된 구조를 세 가지 단계(수학 표현 생성, 실행 가능한 코드 생성, 수치 OCP 솔버 실행)로 설계했습니다. 베이스 OCP 문제와 새로운 임무 요구사항을 결합해 LLM에 의해 최적화 문제를 자동으로 구성할 수 있도록 하며, Python을 사용하여 코드를 생성합니다. 주요 입력으로는 기본 OCP 코드, 기본 OCP 수식, 임무 요구사항 텍스트가 포함되며, LLM은 이들을 기반으로 코드 생성 및 오류 처리를 진행합니다.

- **Performance Highlights**: 제안된 프레임워크는 우주선 근접 작전 시나리오를 대상으로 시험되었으며, 세 가지 대안 프레임워크와 비교하여 성능을 평가했습니다. 실험 결과, LLM 기반의 접근 방식은 각 경우에서 높은 성공률을 보였고, 특히 수학적 제약 조건을 토대로 한 재구성에서 신뢰성을 입증하였습니다. 이 연구는 우주 차량의 궤적 설계에서 LLM의 활용 가능성을 강조하며, 복잡한 임무 요건을 간결하게 처리할 수 있는 새로운 접근법을 제시합니다.



### SaliMory: Orchestrating Cognitive Memory for Conversational Agents (https://arxiv.org/abs/2606.04120)
- **What's New**: 이 논문에서는 SALIMORY라는 새로운 프레임워크를 소개한다. SALIMORY는 사용자의 사실, 선호도, 작업 메모리를 관리하기 위해 인지적으로 구조화된 기억을 유지하는 단일 언어 모델을 훈련한다. 이 접근 방식은 메모리 연관 실패율을 1/3로 줄이고, 최첨단 기술 대비 10% 이상 더 높은 엔드 투 엔드 정확도를 달성하며, Good Personalization 비율을 두 배 이상 향상시킨다.

- **Technical Details**: SALIMORY는 세 가지 상호 보완적인 메모리 저장소를 갖춘 기억 관리 프레임워크로 설계되었다. 여기에는 검증 가능한 사실에 대한 정보, 주관적인 선호를 반영한 데이터 세트, 사용자에게 중요한 최근 세부 정보를 포함하는 작업 메모리가 포함된다. 세 가지 핵심 역할(선택적 주의, 통합, 단서 기반 활용)을 통해 이 모델은 강화학습(Reinforcement Learning)을 통해 학습하고 높은 품질의 메모리 형성을 보장한다.

- **Performance Highlights**: 이 연구는 새로운 LoCoMo-P13n 벤치마크를 도입하며, 이는 개인화된 쿼리를 기반으로 한다. 실험 결과, 9B 모델을 사용한 SALIMORY는 최첨단 기술 대비 10.2% 더 높은 엔드 투 엔드 정확도를 달성하였고, Good Personalization 비율에서 무려 23.5포인트의 개선을 이끌어내었다.



### dMX: Differentiable Mixed-Precision Assignment for Low-Precision Floating-Point Formats (https://arxiv.org/abs/2606.04115)
- **What's New**: 본 연구는 dMX라는 차별 가능한 혼합 정밀도 양자화 프레임워크를 도입하여, 각 레이어에서 학습 가능한 부동 소수점 비트너스를 할당할 수 있도록 하였습니다. 기존의 일반적인 비트 너스가 모든 레이어에 일률적으로 적용되는 단점을 해결하며, 다양한 LLM에 대한 실험을 통해 성능을 검증합니다. 주요 기여로는 더 매끄러운 최적화를 위한 부동 소수점 비트 너스의 변경 가능성을 포함합니다.

- **Technical Details**: dMX는 각 레이어의 부동 소수점 형식을 단일 학습 가능한 오프셋으로 매개변화하여 연속적인 최적화 문제로 성립합니다. 이를 통해, 이산 양자화 형식 간의 갑작스런 변화를 피할 수 있습니다. 또한, 비트 넓이를 하드웨어 호환 형식으로 점진적으로 변환하기 위한 온도 기반의 굽힘 주기를 사용하여 최적화하는 구조를 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, dMX는 Llama, Qwen3, SmolLM2와 같은 다양한 LLM에서 KL 다이버전스 기반의 레이어 선택 휴리스틱보다 지속적으로 우수한 성능을 보였으며, 모델 품질과 평균 비트 너스 간의 효율적인 균형을 이루는 양자화를 달성하였습니다. 이를 통해 최종적으로 최적의 정확도와 효율성을 동시에 확보하는 데 성공하였습니다.



### AgenticDiffusion: Agentic Diffusion-based Path Planning for Vision-Based UAV Navigation (https://arxiv.org/abs/2606.04111)
- **What's New**: 이 논문에서는 AgenticDiffusion이라는 새로운 다중 시점 UAV 내비게이션 프레임워크를 제안합니다. 이 프레임워크는 언어에 기반한 추론, 오픈 어휘 타겟 그라운딩, 비전 기반 확산 계획, 그리고 비선형 모델 예측 제어(NMPC)를 통합하여 UAV의 내비게이션을 예측합니다.

- **Technical Details**: AgenticDiffusion 프레임워크는 사용자의 자연어 지시를 기반으로 동기화된 1인칭 뷰(FPV) 및 상단 뷰(observations) 관측을 분석합니다. 이 프레임워크는 Grounding DINO를 사용하여 세부 목표를 로컬라이즈하고, 두 관측 시점에서의 시각적 특성을 평가하여 내비게이션에 가장 적합한 관점을 결정합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 40회의 실제 UAV 내비게이션 실험에서 80%의 임무 성공률을 달성했습니다. 또한, 확산 계획기는 100%의 경로 생성 성공률을 기록하여 매우 우수한 성능을 입증했습니다.



### SymTRELLIS: Symmetry-Enforced Voxel Latents for 3D Generation (https://arxiv.org/abs/2606.04108)
- **What's New**: SymTRELLIS는 단일 보기 3D 생성 모델의 대칭성을 강화하는 새로운 방법입니다. 이 방법은 VAE(Variational Autoencoder) 또는 flow 모델을 재훈련하지 않고도 TRELLIS.2 생성 중에 대칭성을 강화할 수 있도록 설계되었습니다. 핵심 아이디어는 3D 생성 과정에서 대칭성을 강제하고, 최종 결과물의 기능적 요구 사항을 충족시키는 것입니다.

- **Technical Details**: SymTRELLIS는 공간 변환의 잠재적 작용을 선형 연산자로 모델링하고, 가벼운 spatial-transform latent mapper를 사용하여 비대칭 3D 데이터에 대해 학습합니다. 생성 과정에서 velocity symmetrization을 통해 예측된 흐름 속도(velocity)를 대칭적으로 평균화하며, 이는 ODE(Ordinary Differential Equation) 단계에서 적용됩니다. 이 방식은 대칭 구조의 명확한 추정을 가능하게 하여 더 복잡한 대칭 구조도 처리할 수 있도록 합니다.

- **Performance Highlights**: 심층적인 Benchmark를 통해 SymTRELLIS는 266개의 엄격히 대칭적인 객체에 대해 TRELLIS.2, Hunyuan3D-2.1, TripoSG와 비교하여 모두 대칭 오차 지표를 현저히 줄이며, 기본 모델과 비슷한 재구성 정확도를 유지합니다. 3D 출력물을 테스트한 결과, 대칭성이 올바르게 적용된 모델은 안정적으로 회전하며 성능이 개선되었습니다.



### Building The Ph(ysical)AI Layer Of Machine Intelligenc (https://arxiv.org/abs/2606.04106)
Comments:
          102 pages, 11 Figures

- **What's New**: 이번 논문에서는 물리 원칙(physical principles)을 인코딩하는 원칙 기반의 파운데이션 모델을 제안합니다. 이러한 모델은 대규모 통계적 상관관계를 학습하는 대신, 신호 이론(signal-theoretic) 원칙을 통해 이전에 본 적이 없는 분야로의 전이를 가능하게 합니다. 저자들은 라디오 주파수(RF) 데이터에 기반하여 학습한 단일 도메인에서 다양한 분야로의 교차 도메인 전이를 달성하였습니다.

- **Technical Details**: 제안된 방법론은 기본적으로 주파수 성분으로 신호를 분해하는 푸리에 분해(Fourier decomposition) 및 예측 가능한 변환을 학습하는 대칭 학습(symmetry learning)에 기반합니다. PlanFormer라는 새 아키텍처는 신호 이론 원칙을 내포하고 있으며, 다양한 도메인에서 전이를 가능하게 하기 위해 RF 데이터를 학습 데이터로 사용합니다. 설계 시 주파수를 보존하는 풀링 기법과 에너지 보존을 위한 구성을 조성하여 구체적 목표에 기여합니다.

- **Performance Highlights**: 교차 도메인 전이를 위한 실험에서는 15가지의 다양한 작업에서 1.99M 파라미터의 동결 인코더가 평균 77.7% 정확도를 기록했으며, 물리적 기반 과제에서는 84.5%, 의미적 과제에서는 70.0%의 성과를 보였습니다. 이 성능 결과는 원칙 기반 학습이 물리적 구조를 포착할 수 있음을 나타내며, 의미적 이해는 추가적인 추상이 필요함을 시사합니다.



### Proof-Carrying Agent Actions: Model-Agnostic Runtime Governance for Heterogeneous Agent Systems (https://arxiv.org/abs/2606.04104)
Comments:
          25 pages, 2 tables, 3 figures. Implementation-informed systems paper with bounded public validation

- **What's New**: 이 논문은 Proof-Carrying Agent Actions (PCAA)라는 새로운 거버넌스 모델을 제안합니다. 이 모델은 특정 벤더의 세션 기록이 아닌 행동 인증서(action certificate)를 중심으로 설계되어, 다양한 런타임에서 일관되게 적용될 수 있도록 합니다. PCAA는 사전 행동 허가, 행동 개방, 가정 캡처, 승인 및 결과 마감의 다섯 가지 체크포인트를 조직하여 보다 안전한 행동 관리를 가능하게 만듭니다.

- **Technical Details**: PCAA의 핵심 설계 선택은 행동 인증서가 기본 신뢰 객체로 사용된다는 것입니다. 이는 액션을 안정적인 거버넌스 용어를 통해 라우팅하고, 정책이나 모호성이 있는 경우 인간의 감독 하에 행동을 검토하며, 재생 가능한 형식으로 결과를 증명하는 방법으로 운영됩니다. PCAA는 외부 요인을 인식하여 배 boundary facts와 같은 정보를 포함하여 인증서에 통합하며, 승인 방식도 단일 비트가 아닌 명시적인 집행 가능성 클래스에 의해 기술됩니다.

- **Performance Highlights**: PCAA는 이기종 에이전트 제어 평면에서의 참조 구현을 통해 연구되었으며, 24개의 실행 가능한 시드를 기반으로 96개의 트레이스를 확장한 보호된 벤치마크에서 경로 품질을 유지하면서 다양한 실패 모드를 노출했습니다. 이 연구는 런타임 충돌 없이 이동이 가능한 인증서 기반 행동 관리의 실행 가능성을 보여줍니다. 결과적으로, PCAA는 효율성이 높고 다양한 런타임에 적응할 수 있는 공통 거버넌스 객체를 유지할 수 있습니다.



### The Differentiable Auditory Loop (DAL): An ML Framework for Hyper-Personalized Hearing Aids (https://arxiv.org/abs/2606.04103)
- **What's New**: 이번 연구에서는 청각 장애의 근본적인 인코딩 기능 장애를 해결하기 위한 새로운 오픈 소스 프레임워크인 Differentiable Auditory Loop (DAL)을 소개합니다. 이 프레임워크는 개인 맞춤형 청각 보조 기기 설계를 가능하게 하며, 깊은 신경망을 최적화하여 손상된 청각 뉴럴 활동 패턴을 정상 청각 기준과 일치시킵니다. 이를 통해 기존의 고정된 주파수 종속 증폭 및 압축 방식의 한계를 극복하고, 복잡한 환경에서도 향상된 청취 지원을 제공합니다.

- **Technical Details**: DAL의 기본 구성 요소는 인간의 달팽이관 기능을 모델링한 CARFAC이며, 이는 JAX로 포팅되어 사용됩니다. SEANet이라는 완전 합성곱 UNet 생성기를 채택하여 웨이브폼을 처리합니다. 이 모델은 정상 청각을 고려한 CARFAC 모델의 출력과 개인 청각 장애를 맞춘 CARFAC 모델의 출력을 비교하여 미세 조정을 수행하며, 이를 통해 입력 신호의 노이즈 제거와 청각 손실 보상이 이루어집니다.

- **Performance Highlights**: DAL로 최적화된 SEANet 모델은 기존 마스터 청각 보조 기기(MHA) 기초 모델보다 뛰어난 성능을 보였습니다. 본 연구의 결과는 기계 학습 기반의 청각 보조 기기 신호 처리를 개인화하는 실제적인 경로를 제공합니다. 다음 단계로 하드웨어 배치가 계획되어 있으며, 이를 통해 실제 임상 시험이 가능해질 것입니다.



### POLARIS: Guiding Small Models to Write Long Stories (https://arxiv.org/abs/2606.04095)
- **What's New**: POLARIS(Policy Optimization with LLM-as-a-judge rewards and Anchored-Reference Injection for Storywriting)가 새로운 방식으로 등장했습니다. 이 모델은 긴 형식의 창작 글쓰기에 대한 문제를 해결하기 위해 개발되었으며, 기존의 소형 모델들보다 향상된 성과를 보여줍니다. POLARIS는 구조화된 Story Quality rubric을 가진 LLM 수치 평가자와 인간 작성 이야기 보조(AI)를 통해 훈련됩니다.

- **Technical Details**: POLARIS는 약 1.4K 개의 프롬프트-스토리 쌍으로 구성된 데이터 세트를 사용하여 Qwen3.5-9B 모델에 적용됩니다. 이 모델은 GRPO(Group Reinforcement Policy Optimization) 방식을 사용하며, 인간이 작성한 이야기와 같은 고보상 앵커를 통해 성과를 높입니다. 연구에 사용된 하드웨어는 4개의 A100 GPU입니다.

- **Performance Highlights**: POLARIS-9B는 다섯 가지 기준에서 평가된 결과, 기존의 더 큰 모델들과 비교하여 경쟁력을 보여주고 있습니다. 또한, 훈련 길이의 3배까지 발전된 품질을 유지하며, 대부분의 기존 모델들이 품질 저하를 겪는 구간에서도 효과성을 발휘합니다. 블라인드 인간 평가에서도 POLARIS-9B는 기본 Qwen3.5-9B보다 선호되며 Qwen3.5-27B와 유사한 수준을 기록했습니다.



### Large Language Models Hack Rewards, and Society (https://arxiv.org/abs/2606.04075)
Comments:
          14 pages, 9 figures, 7 tables

- **What's New**: 이 논문에서는 강화학습(Reinforcement Learning, RL)이 대형 언어 모델(Large Language Models, LLMs)에서 보상(reward) 학습을 이끄는 방법을 설명합니다. 연구자들은 사회 규제가 보상 함수와 구조적으로 유사하다는 점을 주장하고, LLM이 이러한 규제에서 발생하는 허점을 이용할 가능성을 제시합니다. 새로운 사회적 해킹(societal hacking) 개념이 도입되어, RL 훈련 과정에서 모델이 사회적 규칙을 조작할 수 있음을 잠재적으로 탐구합니다.

- **Technical Details**: 논문에서는 SocioHack이라는 72개의 사회적 환경을 제공하여 모델이 사회 규칙을 해킹할 수 있는 가능성을 연구합니다. 각 환경은 자연어로 규제(specification)를 정의하는 튜플로 구조화되어 있으며, 이를 통해 RL 훈련이 어떻게 사회 구조를 탐색하고 조작하는지를 분석합니다. 훈련 과정에서 모델은 주어진 프롬프트에 기반하여 전략을 생성하며, 이 과정은 최적화의 불확실성을 조절하여 비효율적인 탐색을 방지합니다.

- **Performance Highlights**: 실험 결과, RL이 손상된 규칙을 61.25%의 회수율(recall)과 90.85%의 정밀도(precision)로 재발견할 수 있음을 보여줍니다. 그러나 현재의 안전 장치들은 여전히 불완전하며, 모델이 해로운 지시를 받을 때만 반응합니다. 이는 사회적 환경에서 최적화 압력이 지속됨에 따라 모델과 규제 간의 지속적인 상호 작용 및 진화가 발생할 수 있음을 시사합니다.



### Adaptive Patching Is Harder Than It Looks For Time-Series Forecasting (https://arxiv.org/abs/2606.04074)
- **What's New**: 이 논문은 시계열 Transformer에서 적응형 패칭(adaptive patching)의 효용성에 대해 새로운 통찰력을 제공합니다. 특히, 콘텐츠에 따라 조정된 패칭이 균일하게 조정된 패칭보다 우수한 성능을 발휘할 조건을 탐구합니다. 저자들은 포인트 예측 손실(pointwise forecasting loss)에서 복잡한 영역이 반드시 미세 패칭(finer patching)에 의한 손실 감소로 이어지지 않음을 강조합니다.

- **Technical Details**: 저자들은 패칭을 예산화된 비트레이트 할당(budgeted bitrate allocation)으로 모델링하고, 동적 패칭 규칙이 균일 기준선을 초과하여 성과를 내기 위해 충족해야 할 명시적인 임계값을 도출합니다. 또한 지역적(로컬) 및 전역적(global) 개선의 한계를 설정하며, 최적의 패칭 크기에 대한 일관된 이점을 찾기 어렵다는 구조적 결과도 제시합니다.

- **Performance Highlights**: 세 가지 대표 아키텍처에 대한 제어된 고립 연구를 통해, 동적 패칭 방식 대신 균일 패칭 크기를 적용했을 때, 검증에서 선택된 균일 기준선의 테스트 MSE가 동적 변형과 비슷하거나 더 나은 결과를 보여주었습니다. 결과적으로 적응형 패칭은 잘 조정된 균일 기준선에 대해 평가되어야 하며, 그 효용성은 더 작은 패치가 실제로 예측 손실을 줄여줄 수 있는 상황인지에 따라 달라집니다.



### TPA-AD: A Two-Stage Pseudo Anomaly-Guided Method for Bearing Time-Series Anomaly Detection (https://arxiv.org/abs/2606.04073)
- **What's New**: 본 논문은 오직 정상 샘플만으로 학습할 수 있는 환경에서 축 박스 베어링의 시간 시계열 이상 감지를 위한 두 단계의 의사 이상 기반 방법론(Two-stage Pseudo Anomaly-guided Anomaly Detection, TPA-AD)을 제안합니다. TPA-AD는 정상 경계 근처에서 의사 이상 윈도우를 생성하고, 이를 통해 대조 학습(contrastive learning)을 수행하여 이상 민감한 표현을 학습합니다. 기존의 방법들이 실질적인 이상 데이터에 의존하는 것과는 달리, TPA-AD는 의사 이상을 구성하여 정상 경계의 분리 가능성을 향상시킵니다.

- **Technical Details**: 이 방법의 첫 번째 단계에서는 정상 샘플의 재구성 모델을 사용하여 각 특성 목표오차를 제어하며 정상 경계 근처에서 의사 이상 윈도우를 생성합니다. 두 번째 단계에서는 정상 윈도우와 의사 이상 윈도우 간의 대조 학습을 수행하고, k-최근접 이웃(KNN) 방법을 사용하여 이상 점수(anomaly score)를 계산합니다. 이러한 접근법은 연속적이고 불연속적인 기능이 혼합된 시나리오에서도 함께 다룰 수 있는 강점을 가지고 있습니다.

- **Performance Highlights**: 주요 실험은 베어링 결함 감지 데이터셋과 저하 과정 데이터셋에서 수행되었으며, 추가적으로 13개의 공개 TSAD 데이터셋에 대한 확장 실험도 포함됩니다. 실험 결과, 제안된 방법은 상대적으로 안정적인 이상 반응을 보이며 저하의 변화에 민감하고, 공공 TSAD 벤치마크 및 실제 고속열차 관련 베어링 데이터에 대해 넓은 적용 가능성을 입증했습니다.



### Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation (https://arxiv.org/abs/2606.04067)
- **What's New**: 최근 클라우드 호스팅된 LLM(대형 언어 모델)의 사용이 증가함에 따라 사용자 요청이 필요하지 않은 민감한 정보를 포함하는 경우가 많아졌습니다. 이 연구에서는 사용자 요청을 보다 안전하게 처리하기 위해 Contextual Integrity(맥락의 무결성) 개념을 도입하고, 민감한 정보를 최소화하는 방식으로 요청 내용을 재작성하는 방법을 제안합니다. 이러한 요청 재작성 기법을 DelegateCI-Bench라는 새로운 벤치마크를 통해 평가하며, 이는 3,167개의 샘플로 구성됩니다.

- **Technical Details**: DelegSI-Bench는 태스크 중심의 Contextual Integrity 벤치마크로, 사용자가 보낼 수 있는 요청을 두 가지 범주, 즉 태스크 필수 스팬(ℰ)과 비필수 민감 스팬(𝒩)으로 분류합니다. 이 연구는 또한 강화 학습을 활용하여 민감한 스팬을 구분하고, 요청 재작성기는 태스크에 필수적인 정보를 유지하면서 과도한 민감 정보 노출을 방지하는 방식으로 학습됩니다. 이를 통해 달성된 모델은 프라이버시와 유틸리티 간의 최적의 균형을 이루고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 재작성기는 기존의 온디바이스 기반선보다 평균 10.1% 더 높은 유틸리티를 기록했습니다. 이는 태스크에 필요한 정보는 유지하는 동시에 불필요한 민감 정보를 효과적으로 제거할 수 있음을 시사합니다. 이 연구는 다양한 태스크 타입과 실사용 쿼리 샘플을 통해 새로운 프라이버시 중시 요청 재작성 방식의 유효성을 입증합니다.



### LLM Compression with Jointly Optimizing Architectural and Quantization choices (https://arxiv.org/abs/2606.04063)
- **What's New**: 본 논문에서는 기존의 프리트레인(Pre-trained)된 대형 언어 모델(LLMs)을 엣지 디바이스(Edge devices)에 적합하게 압축하는 혁신적인 접근법을 제안합니다. 기존의 Neural Architecture Search(NAS) 기법들은 흔히 아키텍처 검색과 양자화(Quantization)를 분리하여 처리하였으나, 본 연구에서는 이를 통합하여 상호 최적화를 이루는 차별화된 NAS 프레임워크를 소개합니다. 이 방식을 통해 메모리 사용량을 크게 줄이고 정확도를 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구팀은 대형 언어 모델 압축을 제약 최적화 문제로 접근하였습니다. 검색 공간은 ζ∈S로 파라미터화(Parameterized)되어 있으며, 후보 네트워크는 이 확률 분포에서 샘플링 됩니다. 이 과정에서 아키텍처 파라미터를 최적화하고 양자화 방식을 조정함으로써, 각 레이어에서의 구조적 제약을 동시 최적화할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 저자들은 제안한 모델이 이전의 NAS-양자화(NAS-then-Quantization) 방식보다 최대 1.4배 더 빠른 추론 속도를 보임을 실험적으로 입증했습니다. 또, 일곱 가지 추론 작업에서 동등한 지연 시간에서 평균 6% 높은 정확도를 달성하는 성능을 기록했습니다. 이러한 결과들은 제안된 내용을 통해 압축 및 양자화 단계에서의 혁신적인 접근이 실제 적용 가능성을 크게 높임을 나타냅니다.



### Spectral Scaling Laws of Muon (https://arxiv.org/abs/2606.04058)
- **What's New**: 이 논문은 Muon 최적화기에서 사용되는 momentum 행렬의 singular value 스펙트럼 변화에 대한 최초의 체계적인 연구를 제시합니다. 다양한 크기의 모델(77M에서 2.8B 파라미터)에 대한 분석을 수행하며, 훈련 중 singular value quantiles가 안정화되는 방식을 추적합니다. 이 안정화 과정은 레이어 타입과 모델 크기에 따라 달라지며, 파라미터 크기에 따른 힘 법칙(power law)과의 관계를 밝혀냅니다.

- **Technical Details**: Muon은 Newton-Schulz(NS) 반복(iteration)을 사용하여 momentum 행렬을 근사적으로 정규 직교화(orthonormalization)합니다. NS가 단순 근사인 관계로, singular value가 작은 방향들은 제대로 정규 직교화되지 않는 한계가 있습니다. 이 연구는 비율적으로 어떤 방향들이 정규 직교화되어야 Muon의 장점을 유지할 수 있는지를 확인하기 위한 통제된 실험을 수행합니다.

- **Performance Highlights**: 연구 결과, 미드 레이어까지는 모델 크기(M)에 따라 약한 스케일링(약 M^{-0.25})이 있으며, 기존의 5단계 NS 구성으로도 적절한 정규 직교화가 가능함을 보여줍니다. 그러나 후기 레이어는 M^{-0.96}의 급격한 스케일링을 보이며, 더 많은 NS 반복이나 조정된 계수를 사용하지 않으면 NS 실패 영역으로 접어들 수 있습니다. 이 법칙들은 실제 구현 시 레이어별로 최적의 NS 구성을 선택할 수 있는 원칙적이고 실용적인 가이드를 제공하여 불필요한 계산을 피하면서 업데이트 품질을 유지할 수 있도록 합니다.



### The Invisible Lottery: How Subtle Cues Steer Algorithm Choice in LLM Code Generation (https://arxiv.org/abs/2606.04057)
- **What's New**: 최근 대형 언어 모델(LLMs)이 다수의 유효한 알고리즘 솔루션을 위한 프로덕션 코드를 생성하는 데 널리 사용되고 있습니다. 이 논문은 알림의 맥락적 요소(이하 cue) 가이드라인이 모델이 선택하는 알고리즘에 미치는 영향을 분석하며, 46,535개의 실험을 통해 cue가 알고리즘 선택에 미치는 영향을 명확히 규명했습니다. 개발자는 이러한 알고리즘 선택 과정에서의 불투명성으로 인해 성능, 보안 및 유지 관리에 미치는 영향을 파악하기 어려운 실정입니다.

- **Technical Details**: 이 연구에서는 알고리즘 선택 문제를 정의하고, 고정된 정확도 하에 알고리즘 선택을 평가합니다. 19종의 cue 유형과 11개의 작업에서 100 pp까지의 알고리즘 가족 분포 변화를 관찰하며, 중립적인 placebo cue에서도 평균 26 pp의 변화를 보입니다. 이 방식은 알고리즘 선택이 cue에 의해 체계적으로 조정되며, 실험을 통해 이러한 경향을 정량적으로 측정했습니다.

- **Performance Highlights**: 알고리즘 선택의 변화는 모델마다 달라, 동일한 cue가 서로 다른 모델에서는 상이한 결과를 초래할 수 있음을 보여줍니다. 예를 들어, 특정 개인화된 cue는 알고리즘의 선택을 크게 변화시켜 성능을 최적화할 수 있습니다. 이 연구를 통해 발표된 결과들은 LLM의 코드 생성 과정에서 우연히 발생하는 성능 변동이 실제로는 "보이지 않는 복권(invisible lottery)"로 작용할 수 있음을 시사합니다.



### A Goal-Set Characterization of Task Composition in the Boolean Task Algebra (https://arxiv.org/abs/2606.04053)
- **What's New**: 이번 연구에서는 Boolean Task Algebra (BTA)의 구조적 가정을 재검토하고, 최적의 확장 Q-값 함수의 공간에서 붕괴를 형식화합니다. 결정론적 MDP(마르코프 결정 프로세스)에서 모든 최적의 확장 Q-함수는 일반적(task) 및 비어 있는(empty) 작업에 의해 완전히 결정된다는 점을 확인했습니다. 이러한 관찰을 바탕으로 목표 집합 기반의 새로운 작업 조합 방법을 소개했습니다.

- **Technical Details**: 연구에서는 두 개의 구성 요소만으로 모든 최적의 확장 Q-함수를 구축할 수 있음을 증명합니다. 이를 통해, 두 개의 학습된 확장 Q-함수에서 적절한 슬라이스만 선택하여 조합된 Q-값 함수를 재구성할 수 있습니다. 이를 통해 작업의 조합 시 학습 비용이 절감되며, 정책 성능을 유지하면서도 효율적인 시간 절약이 가능하다고 설명합니다.

- **Performance Highlights**: 다양한 실험을 통해 추가적인 기본 작업을 학습하는 것이 더 나은 성능으로 이어지지 않는다는 것을 보여주었습니다. 결정론적 MDP와 확률적 MDP 모두에서 조합 시간의 최적화를 연구하고, 확률적 설정에서 각 목표 집합이 최적의 정책을 가질 수 있는 가능성을 제시합니다. 이러한 결과는 BTA 스타일의 조합이 확률적 환경에서 한계를 가진다는 점을 강조합니다.



### RUBAS: Rubric-Based Reinforcement Learning for Agent Safety (https://arxiv.org/abs/2606.04051)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 에이전트의 도구 사용 안전성을 향상시키기 위한 RUBAS라는 새로운 프레임워크를 소개합니다. 기존의 정렬(alignment) 방법은 만족도 신호와 정적(supervised) 감독에 의존하여 다양한 위험을 고려하기에 충분하지 않았습니다. RUBAS는 도구 사용, 인수(argument) 안전성, 반응 안전성 및 유용성을 포함한 네 가지 차원으로 에이전트 행동을 분해하고, 이를 통해 강화 학습을 최적화합니다.

- **Technical Details**: RUBAS는 에이전트 안전성을 평가할 수 있는 구조화된 점수 체계를 도입합니다. 각 차원(r_i)에 따라 새로운 이진 기준(c_{i,j})을 사용하여 에이전트의 행동을 세분화하고 평가합니다. 이러한 기준은 필요한 도구 호출 또는 올바른 실행 순서를 준수했는지와 같은 검증 가능한 신호를 제공합니다. RUBAS는 강화 학습 중에 다차원적인 질적 피드백을 스칼라 보상으로 전환하는 방법을 사용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 RUBAS는 표준 정렬 기준 대비 안전성을 크게 개선하고, 도구 기반의 환각(hallucinations)을 줄이며, 경쟁력 있는 유용성을 유지하는 것을 보여주었습니다. RUBAS의 결과는 다차원 척도 보상이 안전 중시 도구 사용 환경에서 LLM 에이전트를 정렬할 수 있는 효과적인 훈련 신호를 제공한다는 것을 제안합니다.



### LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection (https://arxiv.org/abs/2606.04050)
Comments:
          ICML 2026 Spotlight

- **What's New**: LiftQuant는 고정된 비트 너비(예: 2비트, 3비트) 대신 연속적인 비트 너비 제어를 가능하게 하여 대형 언어 모델(LLM)의 최적 배치를 지원하는 새로운 프레임워크입니다. 이 혁신적인 접근법은 "lift-then-project" 메커니즘을 통해 더 높은 차원의 공간에서 저차원 가중치 벡터를 근사화합니다. LiftQuant는 70B 모델을 2.4비트로 압축하여 24GB GPU에 적합하게 만들 수 있습니다.

- **Technical Details**: LiftQuant는 비트 너비를 추상화하여 연속적인 설계 공간으로 전환합니다. 기존의 양자화 방식과 달리 비트 너비가 정수 집합에 구속되지 않으며, 다양한 비트 너비(예: 2.4비트)를 정의할 수 있습니다. 이 프레임워크는 고차원 리프팅 공간에서 단순한 1비트 격자를 통해 가중치 벡터를 구성하여 효율적인 압축을 가능하게 합니다.

- **Performance Highlights**: LiftQuant는 고성능 구현에서 최신 정수 양자화 기법을 초월하는 성능을 보여주었습니다. 70B 모델이 2.4비트로 압축될 수 있을 뿐만 아니라, 일반적인 12GB GPU에 32B 모델을 2.5비트로 배치할 수 있는 가능성도 열어줍니다. 또한 저비용 선형 변환 및 1비트 균일 양자화기만을 사용하는 효율적인 디코딩 경로를 통해 하드웨어 친화적인 특성을 유지합니다.



### Unlocking Feature Learning in Gated Delta Networks at Sca (https://arxiv.org/abs/2606.04048)
- **What's New**: 이 논문은 Gated Delta Network를 위한 Maximal Update Parametrization (μP)의 완전한 수식을 공식적으로 도출합니다. 연구자는 모든 가중치 클래스에 대해 초기화 분산과 학습률 조정 규칙을 정립하고, 이는 기존 방법론에서 벗어난 독창적인 학습률 스케일링을 요구한다는 점을 발견했습니다. 이를 통해 모델의 효과적인 하이퍼파라미터 전이를 실현할 수 있음을 보여줍니다.

- **Technical Details**: Gated Delta Network의 재귀 상태는 데이터 의존적인 스칼라 게이팅을 통한 전체 행렬 업데이트 방식에 차별화되어 있습니다. 이 논문은 전체 순전파 과정을 통해 좌표 크기 추정값을 엄밀히 전파하고, 학습률 스케일링을 도출하며, 비표준 학습률 조정을 통한 효율적인 학습을 가능하게 합니다. 특히, 게이팅 가중치 매트릭스에 대해 Θ(1/d) 학습률 스케일링을 요구하고, 스칼라 게이팅 매개변수는 Θ(d) 스케일링을 필요로 하며 이는 기존의 μP 설정과 다릅니다.

- **Performance Highlights**: Gated Delta Network의 언어 모델을 여러 너비에 걸쳐 사전 훈련한 결과, μP 수식이 AdamW와 SGD 최적화 기법 모두에서 제로샷 학습률 전이를 가능하게 한다는 것을 확인했습니다. 반면, 표준 매개변수화는 전이가 불가능했으며, 이를 통해 이론적 도출의 타당성과 실질적인 효율성을 입증했습니다. 이 작업은 Gated Delta Network의 특징 학습 가능성을 확장하는 데 기여합니다.



### Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation (https://arxiv.org/abs/2606.04046)
Comments:
          Accepted at ICML 2026

- **What's New**: 본 논문에서는 SceneDiver라는 새로운 방법을 제안하여, 시각-언어 의사결정(vision-language decision making) 과제에서의 인식 한계(perceptual limitation)를 극복하고자 합니다. 기존의 시각-언어 모델(VLMs)과 시각-언어-행동 모델(VLAs)이 각각의 장점을 갖고 있지만, 시각적 환각(visual hallucinations) 문제로 인해 성능 제한을 겪고 있습니다. SceneDiver는 장기 계획 능력을 활용하여, 먼저 전체 장면 그래프(scene graph)를 구축하고 이를 통해 작업을 간단한 하위 문제로 분해하는 방식으로, 효과적으로 중요한 객체에만 집중할 수 있도록 설계되었습니다.

- **Technical Details**: SceneDiver의 중심은 거친 단계에서 세밀한 단계로 진행되는 초점(focus) 계획 수립입니다. 첫 번째 단계에서는 이미지 데이터를 구조화된 그래프 표현으로 변환하여 장면을 전반적으로 이해합니다. 두 번째 단계에서는 VLM이 각 지역 하위 장면을 탐색하여 중요한 객체를 식별하도록 합니다. 또한, 실시간 의사결정에 필요한 지연 시간을 충족하기 위해 가벼운 어댑터(adapter)를 설계하여 VLA 모델에서 효과적인 초점 능력을 추출합니다.

- **Performance Highlights**: 다양한 로봇 조작 및 방 탐색 과제를 통해 실험한 결과, SceneDiver는 조작 작업에서 10%-15%, 탐색 작업에서 최대 16%의 성능 향상을 보여주었습니다. 또한, LIBERO-plus 벤치마크에서 성공률이 9.6% 개선되었으며, 이는 의사결정의 강건성을 향상시키는데 기여했습니다. 이 모든 성능 향상과 함께 계산 효율성도 유지되었으며, 실시간 배포에 적합합니다.



### Bayes-Sufficient Representations in Supervised Learning (https://arxiv.org/abs/2606.04045)
- **What's New**: 이번 연구에서는 Bayes-sufficient representation의 개념을 형식화하고, 이를 통해 고정된 supervised decision problem에서 relevancy의 의미를 살펴봅니다. Bayes quotient라는 새로운 객체를 도입하여, 주어진 joint distribution과 loss에 따라 정보를 효율적으로 분리하는 방법을 제시합니다. 이 프레임워크를 통해 Bayes-minimal과 sufficiency의 개념을 명확히 구분할 수 있으며, 정보를 보존하는 방법을 설명합니다.

- **Technical Details**: Bayes-sufficient representation은 특정 예측 규칙을 구현하기 위한 정보를 포함하는 것을 의미합니다. 연구에서는 Bayes-action이 거의 유일한 경우와 비슷한 케이스에서 Bayes quotient를 정의하고, sufficiency와 minimality를 sigma-algebra의 포함 관계로 특징짓습니다. 이는 특정 loss 하에 필요한 최소한의 정보를 결정하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 이 논문은 다양한 실험을 통해 Bayes-sufficient representation의 특성을 입증하였으며, 각 실험에서 정보를 유지하는 방식, 즉 sufficiency와 non-required information의 구분을 명확히 하였습니다. 실제 데이터인 iNaturalist를 이용한 실험을 통해 이론의 유용성을 입증하였고, supervised learning의 맥락에서 loss가 어떻게 representation target을 유도하는지 설명합니다.



### Channel-Oriented Design for EEG-to-Music Reconstruction (https://arxiv.org/abs/2606.04040)
- **What's New**: 이 논문에서는 뇌-컴퓨터 인터페이스(BCI) 분야에서 EEG 신호를 사용한 음악 복원에 대한 새로운 접근 방식을 제안합니다. 기존의 연구가 주로 시각 및 언어에 초점을 맞춘 반면, 이 연구는 EEG 신호를 활용하여 자연스러운 음악을 복원하는 과제를 다룹니다. 저자들은 채널 혼합이 신호에 미치는 해로운 영향을 강조하고, 이를 해결하기 위해 채널 지향의 디자인 원칙을 제시합니다.

- **Technical Details**: 제안된 방법은 전극을 명시적인 토큰으로 취급하는 채널별 토큰화(channel-wise tokenization), 일관성을 강제하는 채널별 다중 관점 자기 증류(channel-wise multi-view self-distillation), 그리고 노이즈와 아티팩트에 대한 불변성을 향상시키는 채널별 데이터 증강(channel-wise data augmentation)이라는 세 가지 주요 구성 요소로 구성됩니다. 이러한 접근 방식은 EEG 신호의 약하지만 정보성 있는 신호를 보존하고, 음악 표현 영역과의 안정적인 정렬을 가능하게 합니다.

- **Performance Highlights**: 논문에서 제시된 방법은 기존의 EEG2Mel 및 여러 최신 모델과 비교하여 더 우수한 성능을 보여줍니다. 성능 지표에서 CLAP 점수 0.683 및 50종 식별 정확도 0.487을 달성하며, 각 구성 요소가 최종 성과에 의미 있는 기여를 한다는 점을 실증 연구를 통해 입증합니다. 제안된 설계는 채널 수준의 표현을 보존하여 해석 가능성을 향상시키며, 특정 자극에 대한 기여를 보다 명확히 드러냅니다.



### Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization (https://arxiv.org/abs/2606.04039)
Comments:
          Accepted at KDD 2026

- **What's New**: 이번 논문에서는 Neural-guided Ant Colony Optimization (ACO)의 한계를 극복하기 위한 DyNACO라는 새로운 프레임워크를 소개합니다. 기존 ACO는 static priors(정적 사전 정보)에 기반하여 훈련되지만, 실제 적용 시에는 동적인 iterative search(반복적 탐색) 과정에 문제가 있습니다. DyNACO는 feromone distribution(페로몬 분포)와 기존의 솔루션을 주기적으로 관찰하여 동적인 신경망 가이드를 달성합니다.

- **Technical Details**: DyNACO는 ACO backend와 perturbation-based(변동 기반) 시스템을 결합하여 대규모 계산에서의 효율성을 보장합니다. 또한 scope-restricted refinement mechanism(범위 제한 정제 메커니즘)을 통해 안정적인 credit assignment(크레딧 할당)을 수행합니다. 이 시스템은 TSP(Traveling Salesman Problem) 문제를 100,000 노드 인스턴스까지 확장 가능하며, 기존 neural baselines(신경망 기준)보다 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: 적용된 DyNACO는 unguided solver(비지도 솔버)와 비교하여 전체 실행 시간을 줄이는 동시에 성능을 향상시킵니다. CVRP(Capacity Vehicle Routing Problem)로의 확장을 통해도 1% 미만의 신경망 오버헤드를 가지면서 일관되게 비지도 기준을 개선합니다. 연구 결과는 동적 가이드가 정적 사전 정보보다 우월한 이유를 설명하며, 신경망 훈련과 반복적 탐색 동역학의 정렬의 필요성을 강조합니다.



### Unpredictable Safety: Domain-Dependent Compliance and the Transparency Gap in Open-Weight LLMs (https://arxiv.org/abs/2606.04035)
- **What's New**: 이 연구는 도메인에 따라 안전성 행동이 다르게 나타나는 것을 체계적으로 분석했습니다. 연구팀은 7개의 윤리적 도메인에서 5개의 오픈 웨이트 LLMs(12B–70B 모델)를 대상으로 4200회의 상호작용을 통해 실험을 진행했습니다. 결과적으로, 컴플라이언스(compliance) 비율은 14.7%(인신매매)에서 85.7%(감시 설계)까지 다양하여, 안전성 행동이 도메인에 따라 큰 편차를 보임을 확인했습니다.

- **Technical Details**: 각 연구는 윤리적 영역 내 고유한 하위 도메인을 포함하여 20개의 시나리오를 테스트했습니다. 각 시나리오는 분석적(Analytical)과 운영적(Operational) 조건으로 작성되었으며, 5개의 모델을 사용하여 600회의 상호작용을 처리했습니다. 주요 평가는 컴플라이언스 비율과 강력한 거부율이었으며, 조사자에 의해 안전 행동을 평가하는 데 사용된 지표는 이론적으로 체계적인 방법으로 검토되었습니다.

- **Performance Highlights**: 연구 결과는 7개의 도메인을 세 개의 등급으로 나누어 96%의 도메인 간 변Variance을 설명했습니다. 인신매매(14.7%)에서 감시(85.7%)로의 71퍼센트 포인트 간의 격차는 안전 행동이 도메인에 따라 크게 달라진다는 핵심 실증 결과입니다. 이 연구는 현재의 안전 메커니즘들이 신뢰성을 보장하기 위해 필요한 투명성과 일관성이 부족하다는 점을 강조하고 있습니다.



### Do Transformers Need Three Projections? Systematic Study of QKV Variants (https://arxiv.org/abs/2606.04032)
Comments:
          Accepted at ICML 2026 (PMLR vol. 306). 26 pages, 12 figures, 16 tables. Code: this https URL

- **What's New**: 이번 논문에서는 Transformer 모델에서 쿼리, 키, 값 (QKV) 어텐션의 효용성을 재조명하고, 서로 다른 프로젝션 공유 제약을 통하여 결과를 비교하고 있습니다. 세 가지 제약 구조인 Q=K-V (통합된 쿼리-키, 분리된 값), Q-K=V (분리된 쿼리, 통합된 키-값), 그리고 Q=K=V (모든 프로젝션 통합)를 평가하고 있습니다. 이 연구는 QKV의 필요성에 대한 질문을 던지며, 프로젝션 공유의 효과를 정량적으로 분석한 것입니다.

- **Technical Details**: 연구에서 제안된 세 가지 프로젝션 공유 제약을 통해, 프로젝션 매트릭스 수를 줄이고 파라미터 수와 계산 오버헤드를 현저히 감소시킬 수 있음을 보여줍니다. 올바른 비대칭성을 유지할 경우, 상대적인 성능 손실을 최소화하면서도 자원 효율성을 높일 수 있습니다. 또한, 기존의 GQA (Grouped Query Attention)나 MQA (Multi-Query Attention)와는 달리, 프로젝션 매트릭스를 공유함으로써 메모리 효율성과 처리량을 동시에 최적화할 수 있습니다.

- **Performance Highlights**: 프로젝트 공유를 통해 Q-K=V는 KV 캐시 용량을 50% 줄이면서도 300M 파라미터 모델에서 오직 3.1%의 퍼플렉시티 증가를 기록했습니다. 1.2B 파라미터 모델에서도 비슷한 경향을 보여주며, 대규모 모델에서도 품질이 안정적으로 유지됨을 확인했습니다. 또한, Q-K=V와 헤드 공유를 결합함으로써, 메모리 효율성을 더욱 강화할 수 있는 가능성을 제시하였습니다.



### Position: Deployed Reinforcement Learning should be Continua (https://arxiv.org/abs/2606.04029)
Comments:
          Accepted to the ICML 2026 Position Paper Track. See this https URL

- **What's New**: 이 논문에서는 Reinforcement Learning (RL) 분야의 기존 훈련-수정(train-then-fix) 패러다임의 한계를 지적하고, 지속적인 RL 문제로서 새롭게 비춰진 관점을 제시합니다. 저자들은 실행 후에도 최적화 가능한 능력을 갖추지 않은 에이전트가 평가 보상 신호를 받을 수 있는 상황을 다루면서, 지속적인 학습이 필수적임을 강조합니다. 이 새로운 접근법은 배포된 에이전트가 항상 적응하는 것이 중요함을 제안하고 있습니다.

- **Technical Details**: 기존의 RL 모델은 주로 오프라인에서 훈련되고 배포된 후 고정화됩니다. 그러나 이 논문에서는 에이전트 배포 후에 발생하는 비정상성(non-stationarity) 문제를 다룰 수 있는 지속적인 학습 솔루션의 필요성을 제안합니다. 특히, Markov Decision Process (MDP)에서의 기존의 한계와 대체 기법으로서 역사 프로세스(history process) 개념이 도입되어 환경의 복잡한 특성을 반영한다는 점이 강조됩니다.

- **Performance Highlights**: 저자는 RL의 기존 사례들, 예를 들어 AlphaGo와 OpenAI Five의 성공적인 배포를 통해 지속적인 학습의 중요성을 보여줍니다. 대부분의 기존 RL 시스템은 배포 후 바뀌는 환경과의 상호작용을 통해 적응하지 못하고 고정된 정책을 사용하게 되는데, 이는 실질적인 적용에서 vieler한 과제를 제공합니다. 논문은 평가 피드백을 기반으로 한 지속적 적응의 필요성을 통해 RL 커뮤니티가 전통적인 접근법에서 벗어나야 할 이유를 제시하고 있습니다.



### MaskForge: Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models (https://arxiv.org/abs/2606.04027)
Comments:
          28 pages, 7 figures, 11 tables. Preprint

- **What's New**: 이번 연구는 MaskForge라는 새로운 공격 기법을 제안합니다. MaskForge는 기본적인 마스크 인플 필링(native infilling) 능력을 활용하여 dLLM의 공격 표면을 최적화된 검색으로 구축하는 완전 블랙박스(adaptive attack) 기법입니다. 이는 고정된 템플릿이 아닌 구조적 패턴의 라이브러리를 이용하여 특정 목표에 맞춰 적응할 수 있습니다.

- **Technical Details**: MaskForge는 세 가지 주요 기능을 갖추고 있습니다. 첫째, 구조적 패턴 추상화는 특정 목표 타입에 맞춰 재사용이 가능한 패턴을 정의합니다. 둘째, UCB(Upper Confidence Bound) 기법을 활용하여 높은 보상을 주는 패턴과 저조한 보상을 주는 패턴 간의 균형을 맞추며 탐색을 최적화합니다. 셋째, 스코어 기반의 폴백(fallback) 메커니즘을 통해 목표에 맞는 템플릿을 동적으로 생성할 수 있습니다.

- **Performance Highlights**: MaskForge는 테스트에서 뛰어난 공격 성공률을 보였습니다. 5개의 공개 dLLM에서 평균 공격 성공률은 79.3%로, 기존의 경쟁 dLLM 모델보다 17.6% 향상된 결과를 기록했습니다. 또한, AdvBench에 적용될 경우 88.2%의 성공률을 보이며, 이는 구조적 패턴이 dLLM의 본질을 포착하는 데 기여함을 보여줍니다.



### The Biomimetic Architecture of Software 4.0 (https://arxiv.org/abs/2606.04025)
Comments:
          14 pages

- **What's New**: 이 논문은 소프트웨어 4.0이라는 새로운 패러다임을 소개합니다. 이는 인간 지능, 신경 AI 및 자가 반영적인 기호 기초의 자율적 이종 집합체로서, 소프트웨어의 구조적 무결성을 스스로 검증하고 수정할 수 있는 자가 조절 네트워크로 변환합니다. Recognitive라는 프로그래밍 언어와 플랫폼이 이 아키텍처를 구현하여, 기존의 복잡한 외부 구조에서 독립하여 지능의 시대에 진입할 수 있는 기회를 제공합니다.

- **Technical Details**: 소프트웨어 4.0는 endo-homoiconicity 및 exo-homoiconicity 개념을 통해 파라다임의 발전을 이룹니다. endo-homoiconicity는 데이터 구조와 소스 텍스트 간의 엄밀한 구조적 동형성을 형성하고, exo-homoiconicity는 외부 지능이 시스템의 내적 기호 토폴로지를 투명하게 인식할 수 있게 합니다. 이러한 구조는 기존의 프로그래머 중심의 접근을 초월하고, 구조적 무결성을 강화하며, 자동화된 추론 과정을 가능하게 합니다.

- **Performance Highlights**: 기존 소프트웨어 3.x 프레임워크는 구조적 제약을 확률적으로 시뮬레이션하는 비효율성을 내포하고 있지만, 소프트웨어 4.0은 결정론적 기초를 통해 효율적인 추론을 가능하게 합니다. 결과적으로, 이는 깊은 의미 탐색과 가설 탐색으로의 전환을 촉진하여 훨씬 더 높은 성능을 보장합니다. 이 논문은 기존의 소프트웨어 공장 접근 방식을 넘어서, 지능 시대에 적합한 새로운 이론적 기반을 제시합니다.



### CodegenBench: Can LLMs Write Efficient Code Across Architectures? (https://arxiv.org/abs/2606.04023)
Comments:
          29 pages, 22 figures

- **What's New**: 본 연구에서는 CodegenBench라는 새로운 벤치마크 수트를 도입하여 LLM이 다양한 CPU 아키텍처에서 효율적인 병렬 코드를 생성하는 능력을 평가합니다. CodegenBench는 x86_64, Sunway, Kunpeng의 세 가지 하드웨어 플랫폼을 포괄하며, 106개의 표준 BLAS(Basic Linear Algebra Subprograms) 루틴과 각 슈퍼컴퓨터 아키텍처에 적합화된 20개의 전문 계산 커널로 구성되어 있습니다. 이 연구는 LLMs의 성능 저하를 초래하는 요인들을 분석하고, 향후 연구를 위한 데이터셋과 자동화된 평가 인프라를 오픈소스로 제공합니다.

- **Technical Details**: CodegenBench는 LLM의 코드 생성 능력을 평가하기 위한 포괄적인 프레임워크로, BLAS의 표준 수치 루틴 및 Sunway와 Kunpeng 아키텍처 각각의 특화된 작업을 포함합니다. 각 BLAS 서브루틴은 행렬 차원, 보폭 및 데이터 타입 등이 달라지는 다양한 매개변수 조합 하에서 평가되어 LLM이 생성한 코드의 성능과 정확성, 에지 케이스 처리 능력을 포괄적으로 검토합니다. 연구의 주요 초점은 이식성, 아키텍처 인식, 데이터 부족 문제의 극복으로, 이를 통해 LLM의 알고리즘 기반 코드 생성 능력을 강화할 수 있습니다.

- **Performance Highlights**: 연구 결과, 최신 LLM은 x86_64와 같은 보편적인 아키텍처에 대한 최적화된 코드를 생성할 수 있지만, 문서화가 부족하고 훈련 데이터가 제한된 도메인 특정 아키텍처에서 성능 저하를 경험하는 것으로 나타났습니다. LLM은 구현 길이 및 작업 복잡성과 같은 코드 품질에 영향을 미치는 요소들을 분석한 결과, 중간 난이도의 문제에 대해 간결한 코드 스니펫이 필요한 경우에 가장 효과적이라는 것을 보여줍니다. 이러한 결과는 다양한 아키텍처에서 LLM의 코드 생성 플랫폼 간 일반화의 한계를 강조합니다.



### Gravity-Aware Hierarchical Routing for Lightweight SensorLLM on Human Activity Recognition (https://arxiv.org/abs/2606.04019)
- **What's New**: 최근 연구에 따르면, 센서 언어 정합성(sensor-language alignment)을 통한 사람의 활동 인식(HAR)에 있어 두 단계 프레임워크가 의미적 모델링 능력을 향상시킬 수 있다는 사실이 밝혀졌다. 본 연구에서는 Stage 2의 백본(backbone) 모델을 압축할 경우 동적인 활동은 잘 인식되지만, 고정된 저동작(static) 클래스에서의 성능이 크게 저하된다는 문제를 지적한다. 이를 해결하기 위해 우리는 경량 포스트 정합성(adaptation)을 위한 중력 인식(gravity-aware) 계층형 라우팅 헤드를 제안하며, 이 방법은 새로운 대규모 프리트레인(pretraining) 프레임워크가 아니라 이미 정합성된 모델 위에 구축된다.

- **Technical Details**: 본 논문은 중력 인식 계층형 라우팅 헤드를 통해 경량 SensorLLM의 Stage 2 분류 성능을 개선하는 방법을 다룬다. 이 방법은 센서의 각 채널에서 축적된 평균(mean)과 표준편차(std)를 활용하여 자세와 중력 방향에 따른 통계적 단서를 추출하고, 이를 통해 정적 전문가(static expert)와 전체 전문가(full expert)를 소프트 라우팅(soft routing)으로 결합한다. 이는 정적 클래스의 분류 문제를 보다 세분화하여 다루고, 로드 밸런싱 손실(load-balancing loss)을 통해 안정적인 학습을 도모한다.

- **Performance Highlights**: MHealth 데이터셋을 이용한 실험 결과, 제안된 설계는 매개변수 오버헤드를 최소화하면서 매크로 F1 점수를 크게 향상시켰으며, 이러한 성과는 주로 정적 클래스에 집중되는 경향을 보였다. 실험 결과는 동적 활동에 대한 강력한 성능을 유지하면서도 정적 클래스의 인식을 획기적으로 개선할 수 있는 가능성을 보여준다. 본 연구는 단일 데이터셋에 대한 결과를 향상시키며 향후 더 넓은 평가를 위한 기초를 다지는 것을 목표로 한다.



### The Variance Brain Foundation Models Forgot: Third-Order Statistics Predict Cognition Where Billion-Parameter Models Fa (https://arxiv.org/abs/2606.04010)
Comments:
          37 pages, 16 figures, 23 tables

- **What's New**: 이번 연구에서는 Brain foundation models (BFMs)이 fMRI 신호 기반으로 인지적 수행을 예측하는 데 실패하고 있음을 규명합니다. 특히, 이 값들은 약 80,000개의 파라미터를 가진 기능적 연결 행렬(functional connectivity matrix)로부터 linear regression보다 낮은 성능을 보입니다. 연구팀은 이 문제의 원인이 variance allocation problem에 있다고 주장하며, 이를 극복하기 위해 새로운 linear pipeline을 구축하였습니다. 이 새로운 접근법은 기존의 BFMs보다 우수한 성과를 보여주었습니다.

- **Technical Details**: 연구에서는 Brain foundation models가 인지 예측성을 위해 필수적인 높은 차수(cognition-relevant signal)가 잘 캡쳐되지 않음을 발견하였습니다. 특히, BFM이 예측한 cognitive 정보가 fMRI 신호의 co-skewness(third-order co-skewness tensor)를 효과적으로 포착하지 못한다는 점을 지적합니다. 연구팀은 Tucker decomposition을 통해 co-skewness tensor를 분해하고, 이를 기반으로 새로운 기능적 연결 기능을 생성하여 기존의 raw FC 및 pretrained BFM보다 우수한 성과를 거두었습니다.

- **Performance Highlights**: 새로운 접근법은 사전 훈련(pretrained) 없이도 모든 데이터셋과 parcellation에서 기존의 기능적 연결성(raw FC)과 BFM을 초월하는 성능을 보여줍니다. BrainLM 모델의 경우, 모델 크기가 증가함에 따라 성능이 저하되는 경향이 확인되었으며, 연구팀은 BFM의 주요 문제는 아키텍처나 모델 크기가 아니라 사전 훈련 목표라고 강조합니다. 마지막으로, Cumulate-informed finetuning loss는 BFM의 정확도를 raw-FC 기준선까지 끌어올리기도 하였습니다.



### Counterfactual Explanations for Deep Two-Sample Testing (https://arxiv.org/abs/2606.04009)
Comments:
          17 pages

- **What's New**: 최근의 심층 이중 샘플 테스트(deep two-sample test)는 높은 차원의 구조적 데이터에서도 민감도(sensitivity)를 향상시키지만, 검정 통계량(test statistic)이나 p-값(p-value) 외에는 어떤 데이터 특징이 귀무가설(null hypothesis)의 기각을 유도하는지에 대한 통찰을 제공하지 못한다. 이를 해결하기 위해 본 연구에서는 샘플 수준의 편집(sample-level edits)을 생성하여 출처 집단(source group)에서 목표 집단(target group)으로 관찰을 이동시키는 반사실적 설명(counterfactual explanation) 프레임워크를 제안한다.

- **Technical Details**: 제안된 방법은 확산 오토인코더(diffusion autoencoder)와 사전 훈련된 심층 이중 샘플 테스트 모델(pretrained deep two-sample test model)을 결합하여 최대 평균 차이(maximum mean discrepancy, MMD) 목표를 최적화함으로써 타당한 반사실적 사례(plausible counterfactuals)를 생성한다. 이 반사실적 변환은 두 샘플 간의 통계적 가까움을 증대시키며, LPIPS(learned perceptual image patch similarity) 지표를 사용하여 원본 샘플과의 최소성을 체크한다.

- **Performance Highlights**: 합성 2D 형태 데이터 세트와 MRI 코호트에서 평가한 결과, 반사실적 변환은 원본 샘플에 비해 p-값(p-value)을 일관되게 증가시켰으며, 이는 편집된 출처 집합이 통계적으로 목표 분포에 더 가까워졌음을 나타낸다. MRI에서는 국소적인 변화가 알려진 해부학적 차이와 일치하며, 검출된 그룹 차이에 대한 특징을 설명하는 해석 가능한 증거를 제공한다.



### Neural Radiated-Noise Fields for Unmanned Underwater Vehicle Noise Spectrum Prediction in Three-Dimensional Scenes (https://arxiv.org/abs/2606.04008)
- **What's New**: 본 논문에서는 무인 수중 차량(UUV)의 방사 소음 스펙트럼을 3차원 위치, 유도기 수 위치, UUV의 요각 및 주파수를 연속 함수로 표현하는 신경 방사 소음 필드(Neural Radiated Noise Field, NRNF)를 설계합니다. 이는 기존의 전통적인 물리 기반 모델링이 갖추고 있지 않은 연속적인 공간 스펙트럼 반응 모델링 문제를 해결합니다. NRNF는 위치와 주파수에 대한 사인 곱 인코딩을 사용하며, 환경 구조 및 전파 특성을 명시적으로 나타내기 위해 학습 가능한 3차원 장면 특성 격자를 도입합니다.

- **Technical Details**: NRNF 모델 아키텍처는 복잡한 수중 환경에서의 UUV 방사 소음의 스펙트럼 특성을 포착하기 위해 형성되었습니다. 이 모델은 UUV 및 유도기 위치, 요각 및 주파수를 입력으로 받아, 해당 파워 스펙트럼 밀도(Power Spectral Density, PSD)를 출력으로 예측하는 구조입니다. 또한, 장면 특성 격자가 이러한 파라미터를 고려하여 방사 소음의 스펙트럼 예측 기능을 강화합니다.

- **Performance Highlights**: 연구 결과, NRNF는 50에서 5000 Hz 대역에서 평균 예측 오차 3.5 dB를 달성했습니다. 수평 외삽(horizontal extrapolation)은 가장 쉬운 반면, 깊이 외삽(depth extrapolation)은 가장 어려워, 교차 실행 일반화(cross-run generalization)는 중간 난이도에 해당합니다. 장면 특성 격자가 모델의 예측 안정성과 공간 일반화를 크게 향상시키는 데 기여했음을 보여주는 결과 역시 포함되었습니다.



### Early Detection of Alzheimer's Disease Using Explainable Machine Learning on Clinical Biomarkers: A Multi-Class Classification Study Using the Alzheimer's Disease Neuroimaging Initiative (ADNI) Datas (https://arxiv.org/abs/2606.03995)
- **What's New**: 이 연구는 일상적인 임상 평가에서 정상 인지(NC), 경도 인지 장애(MCI), 알츠하이머병(AD)의 세 가지를 구별하는 XGBoost 분류기를 개발하였습니다. 이 연구에서는 ADNI(Alzheimer's Disease Neuroimaging Initiative)에서 얻은 8개의 임상 특성을 사용하였으며, 이는 임상 분야에서 중요한 진단 도구로 자리잡을 전망입니다. 특히, 해석 가능한 기계 학습 모델을 통해 세 가지 클래스를 거의 완벽하게 탐지하는 성과를 이뤘고, 이는 향후 진단 변화에 중요한 영향을 줄 것으로 기대됩니다.

- **Technical Details**: 분석에서 활용된 특징들은 MMSE(Mini-Mental State Examination), CDR Clinical Dementia Rating, CDR Sum of Boxes(CDR-SB), MoCA(Montreal Cognitive Assessment), FAQ(Functional Activities Questionnaire), 나이, 성별, 교육 수준을 포함합니다. XGBoost 모델을 위한 하이퍼파라미터 최적화는 Optuna를 사용하여 수행되었으며, SMOTE 기법을 통해 클래스 불균형을 해결했습니다. SHAP(Shapley Additive exPlanations) 값은 또한 특성 수준의 설명 가능성을 제공하여 예측의 투명성을 높였습니다.

- **Performance Highlights**: 본 연구의 결과, 1,641명의 피험자 중 테스트 세트에서 매크로 AUC는 0.982(95% CI: 0.965-0.995)로 나타났습니다. 정확도는 0.943, 균형 정확도는 0.932였으며, 매크로 F1 점수는 0.927로 기록되었습니다. SHAP 분석에 따르면, CDR Global은 NC와 MCI를 예측하는 중요한 요소로 작용하였으며, CDR-SB와 MMSE가 함께 AD 분류를 주도하는 것으로 분석되었습니다.



### Constraint-Enhanced Physical Search through Correlation Matching (https://arxiv.org/abs/2606.03554)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 논문은 물리적 시스템이 탐색 프로세스에 끼치는 제약의 원리를 제안합니다. 제안된 제약 강화 물리적 탐색 원리에 따르면, 탐색 중의 시간적 상관관계가 물리적 업데이트 동역학에서 유도된 공간적 상관관계와 일치해야 합니다. 이를 통해, 탐색 효율성은 강한 무작위성이나 최대의 반상관 관계로 개선되는 것이 아니라, 시간적 상관관계와 물리적 업데이트 스케일이 맞춰질 때 생성되는 증거로 전환됩니다.

- **Technical Details**: 최소한의 TOW(투그 오브 워) 밴딧 모델을 활용하여, 이 논문은 시간적으로 구조화된 탐색과 제약이 유도하는 정보 재분배를 조사합니다. 연구의 중심 사상은 이러한 상관 구조가 호환될 때 효율적인 물리적 탐색이 나타난다는 것입니다. 본 모델에서는 선택된 옵션이 보상을 나누는 방식과 시간이 흘러가는 동안의 신호 변동이 탐색의 순서를 형성하는 데 기여하도록 설계되었습니다.

- **Performance Highlights**: 시뮬레이션 결과, 저자는 중간 부정 상관관계에서의 효율성 향상을 나타내는 강력한 영역을 확보했습니다. 데이터는 시간적 비중복성과 제약으로 인한 차별적 증거 생성을 결합한 시스템의 향상된 효율성을 강조하고 있습니다. 이 논문은 탐색을 위한 기본 물리적 원칙을 제시하며, 제약과 변동성이 구조화된 상관관계를 생성하고, 효율적인 탐색은 이러한 상관관계와 업데이트 동역학의 상호작용에서 발생한다는 사실을 다룹니다.



### DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning (https://arxiv.org/abs/2509.10247)
Comments:
          8 pages, 11 figures, 1 table

- **What's New**: 이번 논문은 DiffAero를 소개하며, 이는 경량화된 GPU 가속의 완전 미분 가능한 시뮬레이션 프레임워크로, 효율적 쿼드로터(quadrator) 제어 정책 학습을 위해 설계되었습니다. DiffAero는 환경 수준(environment-level)과 에이전트 수준(agent-level)의 병렬 처리(parallelism)를 지원하며, 다중 동역학 모델과 사용자 정의 가능한 센서 스택(IMU, 깊이 카메라, LiDAR)을 통합합니다. 또한, 기존의 시뮬레이터들과는 달리 DiffAero는 높은 성능의 시뮬레이션을 제공하는 동시에, 미분 가능하고 하이브리드 학습 알고리즘을 탐색할 수 있는 연구 플랫폼 역할도 수행합니다.

- **Technical Details**: DiffAero는 GPU에서 물리 시뮬레이션과 렌더링을 완전히 병렬화하여 CPU-GPU 데이터 전송 병목 현상을 제거하고, 시뮬레이션 처리량을 현저히 개선합니다. 논문에서는 Aerial Gym에 비해 물리 시뮬레이션에서 1.8배, 깊이 렌더링에서 9.6배의 속도를 달성했다고 합니다. 이 프레임워크는 네 가지 미분 가능한 동역학 모델, 세 가지 센서 모달리티, 세 가지 비행 작업을 지원하는 모듈식이고 확장 가능한 구조를 가지고 있으며, PyTorch 기반 인터페이스가 통합된 학습을 지원합니다.

- **Performance Highlights**: DiffAero는 실험 결과에 따르면 소비자용 하드웨어에서 몇 시간 내에 강력한 비행 정책을 학습할 수 있음을 보여줍니다. 또한, 이 시스템은 시뮬레이션과 실제 비행 실험을 통해 데이터 효율성이 뛰어난 미분 가능하고 하이브리드 학습 알고리즘 개발을 지원하며, 시뮬레이션에서 실제로 전이(simum-to-real transfer)가 원활하게 이루어질 수 있도록 합니다. 코드는 커뮤니티에서 참고할 수 있도록 공개되었습니다.



### How do machines learn? Evaluating the AIcon2abs method (https://arxiv.org/abs/2401.07386)
Comments:
          textual review (spelling and grammar); reorganization of the elements of some figures; New references included

- **What's New**: 이번 연구는 AIcon2abs 방법(AI from Concrete to Abstract: Demystifying Artificial Intelligence)을 확장하였으며, 이는 다양한 연령대 (K-12 학생 포함)의 대중들이 머신러닝 (ML)을 이해하는 데 도움을 주기 위해 설계되었습니다. 이 방법은 접근이 용이한 WiSARD 알고리즘을 활용하여 간단한 사용자 경험을 제공하며, 기술적 배경이 없는 사용자에게도 적합합니다.

- **Technical Details**: AIcon2abs 방법은 WiSARD 알고리즘을 사용하여 비연결 환경에서도 유효하게 머신러닝 과정을 시각화하고 상호작용할 수 있게 합니다. WiSARD는 최소한의 데이터셋으로도 학습 가능하여, 사용자가 데이터를 추가함에 따라 머신의 정확도가 점진적으로 향상되는 과정을 관찰할 수 있게 합니다. 이 알고리즘은 학습한 내용을 시각적으로 표현하는 기능도 제공하여, 분류된 데이터의 중요한 특성을 강조합니다.

- **Performance Highlights**: AIcon2abs는 34명의 브라질 참가자(어린이 5명, 청소년 5명, 성인 24명)와 함께 6시간의 원격 코스로 실험되었습니다. 참가자 대부분이 이 방법에 대해 긍정적인 피드백을 주었으며, 결과적으로 설정한 목표를 달성함에 있어 높은 정도의 만족도를 보였습니다. 이 연구는 CEP-HUCFF-UFRJ 연구 윤리 위원회의 승인을 받았습니다.



### AI from concrete to abstract: demystifying artificial intelligence to the general public (https://arxiv.org/abs/2006.04013)
Comments:
          23 pages; 2 tables; 47 figures; review comment: Included references for the final published peer-reviewed version of this pre-print: this https URL and this https URL typos corrected

- **What's New**: 이 논문에서는 인공지능(AI)을 일반 사람들이 이해할 수 있도록 돕기 위한 새로운 방법론인 'AIcon2abs'를 제안합니다. 이 방법론은 특히 아이들을 포함한 일반 대중이 인공지능을 쉽게 이해할 수 있도록 구체적인 활동을 통해 접근합니다. 또한, 실질적인 학습 머신 개발과 그 학습 과정을 관찰하는 방식으로, 인공지능에 대한 신비감을 줄이고자 합니다.

- **Technical Details**: AIcon2abs는 시각적 프로그래밍(visual programming)과 WiSARD weightless 인공 신경망을 결합한 방법론입니다. 이 접근법은 기존 프로그램에서 외부 모듈로 다뤄지던 기계 지능을 프로그램의 주요 구성 요소로 통합하여 교육합니다. 학습 및 분류 작업이 다른 프로그래밍 구조와 동일하게 구성 요소로 포함됨으로써, 인공지능의 기본 개념을 보다 명확하게 배울 수 있도록 돕습니다.

- **Performance Highlights**: AIcon2abs의 적용을 통해 데이터에서 학습할 수 있는 프로그램과 일반 컴퓨터 프로그램 간의 차이가 더욱 뚜렷하게 드러나게 됩니다. WiSARD weightless 인공 신경망 모델의 단순함 덕분에 학습 및 분류 작업의 내부 실현 과정을 쉽게 시각화하고 이해할 수 있다는 장점이 있습니다. 이는 학습자들이 인공지능에 대한 기본 개념을 보다 효과적으로 이해하는 데 기여할 것으로 기대됩니다.



### CR-Seg: Attention-Guided and CoT-Enhanced Coarse-to-Refined Reasoning Segmentation (https://arxiv.org/abs/2606.03564)
- **What's New**: 이번 논문에서는 복잡한 언어로 설명된 대상 객체를 분할하는 Reasoning Segmentation 문제를 다루고 있습니다. 기존의 방법들은 멀티모달 대형 언어 모델(MLLM)과 분할 모델 간의 정합성 문제를 겪고 있었으며, 이 문제를 해결하기 위해 Attention-Guided 및 CoT-Enhanced Coarse-to-Refined Reasoning Segmentation(이하 CR-Seg)이라는 새로운 두 단계 프레임워크를 제안합니다. 이 프레임워크는 MLLM의 attention map을 기반으로 하여 초기 분할을 개선합니다.

- **Technical Details**: CR-Seg는 Extract Attention Maps and Points (EAP) 모듈을 통해 coarse target localization을 위한 attention map을 추출하고, 이를 SAM에 통합하여 마스크를 개선합니다. 또한, Global-to-Local Chain-of-Thought (GLCoT) 접근 방식을 도입하여 모델이 전역 장면 맥락에서 지역 타겟 세부 사항으로 점진적으로 추론하도록 유도합니다. 이러한 방법을 통해 내재된 응답 의미를 유지하면서도 정합성 문제를 완화할 수 있습니다.

- **Performance Highlights**: 영향력 있는 실험을 통해 CR-Seg는 기존 방법들에 비해 높은 성능을 나타냈으며, Dummy에서 동작한 여러 복잡한 테스트에 대해서도 효과적인 결과를 보고했습니다. 이를 통해 CR-Seg의 강력한 타겟 분별력을 입증하였으며, 특히 동일 카테고리 객체 간의 미세한 차별화에서 두각을 나타냈습니다.



### P$^2$-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization (https://arxiv.org/abs/2606.03376)
- **What's New**: 본 연구는 Perceptual Processing Direct Preference Optimization (P$^2$-DPO)이라는 새로운 훈련 패러다임을 제안합니다. 이 방법은 모델이 자신의 선호 쌍을 생성하고 학습함으로써 인식 병목 현상을 직접 해결합니다. 기존의 비전 무관 선호 쌍의 한계를 극복하며, 데이터의 비효율성을 줄이기 위한 방법론이 추가되었습니다.

- **Technical Details**: P$^2$-DPO는 두 가지 새로운 유형의 선호 쌍을 도입합니다: (1) Focus-and-Enhance Preference Pair는 인식 병목 현상을 해결하기 위해 이미지의 세밀한 디테일에서 개선된 출력과 저하된 출력을 대조하며, (2) Visual Robustness Preference Pair는 깨끗한 신호와 노이즈가 있는 신호의 출력을 대조하여 시각적 견고성을 강화합니다. 이 과정에는 동적 가중치 조정 및 보정 손실(Calibration Loss)도 포함되어 효율적인 학습 루프를 생성합니다.

- **Performance Highlights**: 실험 결과는 P$^2$-DPO가 강력한 기준 모델들을 초과하며, 비용이 많이 드는 인간 피드백 없이도 비교 가능한 훈련 데이터 및 비용으로 성과를 달성함을 보여줍니다. 또한, Attention Region Fidelity (ARF) 및 이미지 저하 시나리오에 대한 평가를 통해 P$^2$-DPO가 인식 병목 현상을 해결하고 시각적 견고성을 향상시키는 데 효과적임을 입증했습니다.



### Implement Kubernetes Pod-Level Remote Attestation for Confidential Workloads on dstack (https://arxiv.org/abs/2606.03323)
- **What's New**: dstack-capsule 플랫폼은 Kubernetes에서 Pod 수준의 원격 증명을 가능하게 하는 새로운 접근 방식을 제안합니다. 기존의 Confidential Containers (CoCo) 모델에서 발생하는 자원 오버헤드를 줄이며, 여러 Pod가 단일 Confidential VM을 공유하면서도 각 Pod의 독립적인 신원을 보장합니다. 새로운 이중 계층 증명 아키텍처를 통해 하드웨어 기반의 신뢰성을 확보합니다.

- **Technical Details**: dstack-capsule의 구조는 정적 플랫폼 측정과 동적 Pod 신원을 기반으로 한 두 계층의 증명 시스템을 특징으로 합니다. 각 Pod의 신원 정보는 TDX Quote의 report_data 필드에 삽입되어 하드웨어에 의해 서명됩니다. 추가적으로 시스템은 다층 산Sandbox(격리) 구조를 갖추어 인프라 보안을 강화하고, 특정 노드를 설정 모드에서 보안 모드로 원자적으로 전환하기 위한 특허된 특권 퓨즈 메커니즘을 도입합니다.

- **Performance Highlights**: dstack-capsule은 기존 CoCo와 비교했을 때, Pod 단위의 검증을 가능하게 하면서도 자원 소모를 최소화하는 성과를 보입니다. 본 연구에서는 dstack-capsule의 보안 특성, 증명 정확성, 성능을 평가하여, 기존의 per-VM 격리의 오버헤드 없이 높은 수준의 보안을 달성하는 것을 입증하였습니다.



New uploads on arXiv(cs.RO)

### GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors (https://arxiv.org/abs/2606.05160)
Comments:
          Project page: this https URL

- **What's New**: GRAIL은 로봇 시연을 위한 전통적인 물리적 설정을 대체하는 디지털 생성 파이프라인입니다. 이 시스템은 물리적 환경 재구성이 필요 없이 다양한 3D 자산과 장면을 조합하여 상호작용을 합성합니다. GRAIL은 생성된 비디오로부터 피사체와 카메라 파라미터를 지정하여 4D 인간-객체 상호작용 궤적을 reconstruct합니다.

- **Technical Details**: GRAIL 시스템은 비디오 기반 모델(VFM)을 사용하여, 주어진 3D 물체 구성에서 시작해 상호작용을 제어합니다. 생성된 데이터는 로봇이 따라 하기 쉬운 궤적을 자동으로 생산하여 실제 환경에서의 시뮬레이션 학습에 적합합니다. 이러한 구조는 전체적인 인간-객체 상호작용을 효과적으로 관리하며, 다양한 환경 조건에서의 로봇 동작을 최적화합니다.

- **Performance Highlights**: GRAIL은 20,000개 이상의 시퀀스를 생성하여 다양한 픽업 및 계단 오르기 작업을 수행하는 로봇 제어 정책을 훈련했습니다. 훈련된 시각 정책은 실제 환경에서 84%의 성공률로 다양한 객체를 집어 올리고, 90%의 성공률로 계단을 오르는 성과를 거두었습니다. 이는 GRAIL이 실제 로봇 학습에서 매우 유용함을 입증합니다.



### X4Val: Learning Neural Surrogates for Variance-Reduced Policy Evaluation (https://arxiv.org/abs/2606.05159)
- **What's New**: 본 논문에서는 X4Val이라는 새로운 프레임워크를 소개합니다. X4Val는 비대칭 다중 도메인 데이터에서의 실제 세계 성능 메트릭(metric) 추정치를 분산 감소(variance reduction)를 통해 개선합니다. 이 프레임워크는 실제 및 보조 도메인에서 수집된 샘플을 공유 표현 공간에 임베딩하고, 이를 통해 실제 메트릭의 전이 가능한 예측기를 학습하여 높은 신뢰도의 성능 추정을 가능하게 합니다.

- **Technical Details**: X4Val는 총 3단계로 진행됩니다. 첫째, 여러 도메인에서 수집된 데이터 샘플들을 공유 표현 공간로 임베딩합니다. 둘째, 이 임베딩과 보조 특징을 사용하여 실제 메트릭을 예측하는 신경망 서브리게이트(neural surrogate)를 학습합니다. 마지막으로 배운 서브리게이트를 통계적으로 유효한 제어변량(control variate) 추정기에 통합하여 직접적인 패어링이 없더라도 분산 감소를 가능하게 합니다.

- **Performance Highlights**: X4Val는 자율 주행 및 실제 로봇 조작 태스크에서 최대 38.4%의 분산 감소를 달성할 수 있음을 보였습니다. 또한, X4Val는 강력한 기준선(baseline)과 비교하여 일관된 성능 향상을 보여주는 실험 결과를 제공합니다. 이를 통해 비대칭 보조 데이터(non-paired, heterogeneous data)가 엄격한 로봇시스템 검증의 샘플 효율성을 크게 향상시킬 수 있음을 입증하였습니다.



### HORIZON: Recoverability-Governed Curriculum for Physical-Domain Scaling (https://arxiv.org/abs/2606.05143)
Comments:
          16 pages, 9 figures

- **What's New**: 이번 논문에서는 로봇 정책을 확장하는 과정에서 복잡한 물리적 환경을 효과적으로 다룰 수 있는 방법에 대해 논의합니다. 연구진은 정책 학습에 적합한 물리적 동역학의 정도를 조절하여 직접적인 데이터를 생성할 수 있도록 하는 '회복 가능성(recoverability)'이라는 개념을 도입했습니다. 이를 통해 HORIZON이라는 커리큘럼을 통해 물리적 도메인의 확장을 실현하고자 합니다.

- **Technical Details**: HORIZON은 각 단계에서 복구가 가능한 한계 내에서 물리적 도메인을 확장하도록 설계되었습니다. 이 방법은 고정된 무작위화(randomization)를 지속적인 물리적 성장 과정으로 전환함으로써, 학습 가능한 한계를 유지하면서도 새롭고 유용한 행동을 탐색할 수 있도록 합니다. 특히, 동적 변화와 정책의 상호작용을 고려한 계산 방식이 도입되었습니다.

- **Performance Highlights**: 실험을 통해 HORIZON의 효과적인 물리적 도메인 확장이 확인되었습니다. 연구 결과, 복구가 불가능한 실패를 피하면서도 정책 학습에 유용한 데이터 수집이 가능하다는 것이 드러났습니다. 또한, 고립된 전문가들을 학습하는 오프라인 기법보다 정책 상호작용이 중요하다는 점이 강조되었습니다.



### Generalization of World Models under Environmental Variability for Vision-based Quadrotor Navigation (https://arxiv.org/abs/2606.05015)
- **What's New**: 이번 연구는 환경의 변동성에 대한 월드 모델의 강건성을 체계적으로 분석합니다. 비전 기반 쿼드로터 내비게이션을 테스트 문제로 삼아, 여러 수준의 환경 무작위성 하에 DreamerV3 기반의 월드 모델을 훈련하고 이를 교차 환경 검증을 통해 평가합니다. 결과적으로, SSL 사전 훈련 중 월드 모델의 강건성이 시뮬레이션에서 실제 세계로의 전이 성능을 예측하는데 중요한 역할을 한다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 3단계의 과정을 통해 월드 모델을 분석합니다. 첫 번째 단계는 변동성이 다른 환경에서 데이터 수집을 포함하며, 두 번째 단계는 DreamerV3 월드 모델을 하이퍼파라미터 최적화와 함께 SSL 사전 훈련하는 것입니다. 마지막으로, RL 세부 조정 단계에서는 각 무작위성 수준에 최적화된 월드 모델을 사용하여 훈련을 진행합니다.

- **Performance Highlights**: 연구에 따르면, 교차 환경 SSL 검증에서 잘 일반화된 모델은 실제 환경에서 성공적으로 배포되었습니다. 이러한 모델은 0.67m 너비의 좁은 틈을 통과하는데 성공했으며, 강건한 월드 모델의 질이 실제 환경에서의 성능에 중요한 영향을 미치는 것으로 나타났습니다. 하이퍼파라미터 검색 결과는 DreamerV3 기반 시스템 훈련을 위한 실용적인 가이드를 제공합니다.



### Potential-Guided Flow Matching for Vision-Language-Action Policy Improvemen (https://arxiv.org/abs/2606.04968)
- **What's New**: 이 논문은 ForesightFlow라는 새로운 자기 유도 흐름 정책(self-guided flow policy)을 소개합니다. 이 정책은 각 생성된 작업 청크에 대해 학습된 성공 잠재성 벡터(success-potential vector)를 추가하여, 일반적인 비평가(critic) 없이 후보 작업을 제안하고 점수 매길 수 있는 기능을 제공합니다. 이를 통해 행동 업데이트와 잠재 업데이트를 다르게 감독하여, 보다 효율적인 훈련을 가능하게 합니다.

- **Technical Details**: 논문에서는 ForesightFlow가 혼합 품질의 데이터를 기반으로 VLA 흐름 정책을 미세 조정(fine-tunes)한다고 설명합니다. 여기에는 증강된 작업-잠재 끝점(augmented action-potential endpoint), 스테이지 수준의 성공 잠재성 타겟(stage-level success-potential targets), 분리된 장점 가중치 흐름 매칭(decoupled advantage-weighted flow matching), 자기 유도 최상의 K 후보 추론(self-guided best-of-K inference)을 포함합니다. 이 과정에서 모든 후보 작업은 통계적으로 평가되며, 각 작업 청크에 대한 내재적 점수가 부여됩니다.

- **Performance Highlights**: ForesightFlow는 5개의 BEHAVIOR-1K 시뮬레이션 작업과 5개의 실세계 이족 조작 작업에서 평가되었습니다. 그 결과, ForesightFlow는 모방 기준(imitation baselines)보다 향상된 성능을 보였고, 시뮬레이션 성공률에서 가장 강력한 별도 비평가 기준을 일치시키며, 실제 성과를 개선하고 훈련 컴퓨트(compute)를 38% 줄였습니다. 이 논문은 가치 환각(value hallucination) 문제를 감소시키고, 단일 단계 추정기가 후보 순위를 유지할 수 있도록 하며, 자기 유도 샘플링이 장기 실행에서 성능을 개선한다고 밝히고 있습니다.



### WAM-Nav: Asymmetric Latent World-Action Modeling for Unified Visual Navigation (https://arxiv.org/abs/2606.04907)
- **What's New**: WAM-Nav는 임베디드 비주얼 내비게이션을 위한 Latent World-Action Model로, 동작 생성과 잠재적 비주얼 예측을 동시에 학습함으로써 더 강력하고 예측 가능한 내비게이션 결정을 허용하는 새로운 접근 방식을 제안합니다. 기존의 반응형 정책들은 관찰을 직접 행동으로 매핑하는데, WAM-Nav는 비주얼 특징과 제어 동역학 간의 깊은 결합을 통해 장애물을 미리 예상하고 피하는 능력을 강化합니다. 또한, WAM-Nav는 이미지 목표, 포인트 목표, 목표 없음 탐색을 위한 통합 정책을 지원하는 것입니다.

- **Technical Details**: WAM-Nav는 대칭적 공동 확산을 위한 공유 Diffusion Transformer를 활용하여 장기 행동과 단기 비주얼 예측을 동시에 생성하는 비대칭 행동-예측 모델링을 구현합니다. 또한, episode-level ego-motion 이력과 순차적 비주얼 관찰을 통합하는 이중 흐름 맥락 조건부 메커니즘을 도입하여 부드럽고 일관된 궤적 생성을 장려합니다. 이러한 접근법은 여러 목표 유형 간의 균형 잡힌 표현을 유지하는 통합 목표 정렬 모듈과 결합되어 단일 정책 내에서 다양한 탐색을 지원합니다.

- **Performance Highlights**: WAM-Nav는 ClutterScenes와 InternScenes benchmarks에서 엄청난 일반화 성능을 보여주며, Image-Goal과 Point-Goal 내비게이션에서 각각 15.7%와 3.3%의 성공률 향상을 기록했습니다. 또한, 실세계 가까운 시뮬레이션-리얼 전이에서도 평균 85%의 작업 성공률을 달성하여 다양한 실내 및 실외 환경에서 효과적인 성능을 보였습니다.



### D$^3$-MoE:Dual Disentangled Diffusion Mixture-of-Experts for Style-Controllable End-to-End Autonomous Driving (https://arxiv.org/abs/2606.04884)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 D$^3$-MoE (Dual Disentangled Diffusion Mixture-of-Experts)라는 새로운 자율주행 계획 프레임워크를 소개합니다. 이 프레임워크는 기존의 스타일 평균화 문제를 극복하기 위해 궤적 모델링을 두 가지 상호 보완적 축으로 나누었습니다. 행동 축에서는 스타일 조건화된 확산 프로세스를 통해 다양한 후보 궤적을 동시에 생성할 수 있으며, 물리적 축에서는 독립적으로 작동하는 전문가들이 조합되어 최종 궤적을 형성합니다.

- **Technical Details**: D$^3$-MoE는 세분화된 궤적 생성과 선택 과정을 독립적으로 수행하는 구조로 설계되었습니다. 이를 위해, 각각의 전문가가 스타일 지침에 따라 서로 다른 궤적 구성 요소를 생성하며, 후속 모듈은 사용자 선호도나 평가 점수를 기반으로 최적의 궤적을 선택합니다. 이러한 분리된 라우터들은 수동 레이블 없이 지반 진실 물리학을 기반으로 한 자기 감독 타겟을 통해 훈련되었습니다.

- **Performance Highlights**: NAVSIM 벤치마크에서의 포괄적인 평가를 통해 D$^3$-MoE가 88.2 PDMS 및 84.3 EPDMS의 성능을 달성했음을 확인하였습니다. 이 연구의 Best-of-Three 앙상블 전략을 활용하면 성능이 91.3 PDMS 및 87.5 EPDMS로 상승하여 계획의 질과 스타일 제어 가능성에서 두드러진 개선을 보여줍니다.



### Teaching Robots to Say 'I Don't Know' : SENTINEL for Uncertainty-Aware SLAM (https://arxiv.org/abs/2606.04853)
Comments:
          6 pages, 10 figures, 3 tables, This paper was accepted at Uncertainty in Open-World Robotics Workshop in conjunction with Internation conference of robotics and automation (ICRA 2026)

- **What's New**: 이번 논문에서는 기존의 저가형 2D LiDAR의 한계를 극복하기 위해 새로운 신뢰도 추정 프레임워크인 SENTINEL을 소개합니다. 이 시스템은 라벨 및 훈련이 필요 없으며, 로봇이 노이즈 있는 센서 데이터를 탐지하고 조정할 수 있도록 돕습니다. SENTINEL은 라이다와 RGB-D 카메라 간의 일관성을 활용하여 정밀한 신뢰도 점수를 생성합니다.

- **Technical Details**: SENTINEL은 저가 라이다의 스캔 통계와 크로스 모달 깊이 일관성 신호를 결합하여 실시간으로 신뢰도 점수를 계산합니다. 이 시스템은 GPU나 학습 데이터에 의존하지 않고 11.5Hz의 빈도로 작동합니다. 세가지 핵심 기술 기여로는 장애물 탐지 능력이 있는 신뢰도 프레임워크와 센서 독립성 발견, 실물 검증의 필요성이 강조됩니다.

- **Performance Highlights**: SENTINEL은 GEFIER R1 로봇 플랫폼에서 다양한 표면 조건에서의 성능을 평가받았으며, 이는 라이다 신뢰도 평가에서 효과적으로 작동함을 보여줍니다. 실험에서 유리, 거울 등 여러 표면을 사용하여 깨끗한 스캔과 결함이 있는 스캔을 명확하게 구별할 수 있었습니다. 이는 로봇이 부정확한 데이터에 기반하여 잘못된 작업을 수행하는 것을 방지하는 데 기여합니다.



### M3imic: Learning a Versatile Whole-Body Controller for Multimodal Motion Mimicking (https://arxiv.org/abs/2606.04829)
- **What's New**: 최근 연구에서는 여러 기능을 지원하는 유니버설 전체 신체 제어 프레임워크인 Multi-Modal Mimic (M3imic)을 제안하였습니다. 이 프레임워크는 로봇 관절 각도, 인간 포즈 경로, 그리고 말단 효과기 포즈를 하나의 공유 잠재 공간으로 매핑하여 서로 다른 모션 레퍼런스 모달리티를 통합합니다. M3imic은 대규모 강화 학습을 활용하여 여러 모션 레퍼런스 모달리티 간의 SIM-TO-REAL 전이를 가능하게 합니다.

- **Technical Details**: M3imic은 Markov Decision Process (MDP)로 구성된 강화 학습 문제로 전체 신체 제어를 정의합니다. 각 상태는 로봇의 감지 상태와 참조 모션 상태로 구성되며, 정책은 현재 상태에 맞는 행동을 출력합니다. 본 연구는 다양한 모션 모달리티와 정밀도 변화를 지원하는 데이터를 기반으로 하며, 커리큘럼 학습 전략을 통해 훈련을 안정화합니다.

- **Performance Highlights**: 제안된 모델은 Unitree G1 로봇을 활용한 시뮬레이션 및 현실 세계 실험에서 우수한 성능을 보였습니다. 시뮬레이션에서는 98.42%의 성공률을 기록했으며, 이는 다양한 모션 레퍼런스에 대한 뛰어난 일반화 능력을 나타냅니다. 이와 같은 성과는 M3imic이 각기 다른 모달리티를 효과적으로 통합할 수 있음을 보여줍니다.



### HapTile: A Haptic-Informed Vision-Tactile-Language-Action Dataset for Contact-Rich Imitation Learning (https://arxiv.org/abs/2606.04825)
- **What's New**: HapTile 데이터셋은 기존의 비주얼-언어-행동(Vision-Language-Action) 데이터셋의 한계를 극복하고, 직접적인 촉각 피드백을 통합했습니다. 이는 로봇의 엔드 이펙터에서의 촉각 피드백과 원격 조작 시 촉각 정보를 포함한 데모로 이루어져 있습니다. 이 데이터셋은 다양한 일상적 조작 작업을 포함하며, 언어 지시를 통해 작업 목표를 조정할 수 있는 것을 특징으로 합니다.

- **Technical Details**: HapTile 데이터셋은 1,726개의 데모로 구성되어 있으며, 38개의 작업과 9가지 조작 기술이 포함되어 있습니다. 데이터 수집은 촉각 정보를 반영하는 원격 조작 인터페이스를 통해 9명의 인간 조작자에 의해 수행되었습니다. 데이터는 15Hz로 샘플링된 언어 지시, 비주얼 및 촉각 데이터, 로봇 상태, 행동 경로와 동기화되어 기록됩니다.

- **Performance Highlights**: 이 논문에서는 두 개의 기본 모델을 사용하여 HapTile을 통한 접촉 기반 정책 학습의 효과를 평가하는 벤치마킹 연구를 진행했습니다. 데이터셋은 기존의 처리 방식보다 향상된 조작 품질과 작업 인식을 제공합니다. 이 연구는 모든 조작에서 다양하고 일상적인 작업을 포함하는 데이터셋의 필요성을 강조하며, 이 데이터셋과 관련된 코드도 오픈 소스로 제공될 예정입니다.



### Real-World Deployment of a 5G-Connected Edge-Controlled Aerial Robot in Industrial Subterranean Mines (https://arxiv.org/abs/2606.04818)
Comments:
          6 pages, 8 figures, MED 2026

- **What's New**: 이 논문은 5G 연결된 공중 로봇의 최초의 실제 자율 비행을 소개하며, 에지 오프로드된 제어기가 적용된 경우를 다루고 있습니다. 로봇은 활성 산업 지하 광산 내부에서 작동하며, 고급 제어기는 근처의 쿠버네티스 기반 에지 클러스터에 배치됩니다. 5G New Radio (NR) Standalone (SA) 네트워크를 통해 로봇과 에지 간의 통신이 이루어지며, 제안된 5G 에지 기반 시스템은 실제 산업 환경에서 평가됩니다.

- **Technical Details**: 제어 방법으로는 Model Predictive Controller (MPC)가 사용되며, 이를 통해 로봇이 광산 환경을 매끄럽고 충돌 없이 탐색할 수 있도록 하는 제어 동작이 생성됩니다. 이 시스템은 인간 운영자가 지정한 웨이포인트(waypoint)를 기반으로 UAV의 경로를 제어하며 비선형 MPC(NMPC)를 활용하여 실시간 요구사항을 충족합니다. 시스템은 네트워크 제어 시스템이며, 통신 지연의 영향을 받으며, 상태 정보를 지연된 값으로 처리합니다.

- **Performance Highlights**: 이 연구는 중앙 집중식 클라우드 서버와 비교하여 에지 기반 시스템이 높은 계산 능력과 낮은 지연을 제공하는 데 기여하며, 특히 지하 광산 환경에서의 로봇 시스템의 안전하고 효율적인 배치를 검증합니다. 실제 실험 결과는 5G 연결 에지 제어 로봇 플랫폼이 비LAB 환경에서 자율적으로 작동할 수 있는 가능성을 보여줍니다. 이로 인해 향후 산업 및 사회적 응용을 위한 실질적인 해결책을 제공하는 데 기여합니다.



### SoftPINCH: EMG-Driven Soft Exoskeleton Assistance for Finger Flexion and Grasping (https://arxiv.org/abs/2606.04776)
Comments:
          Submitted to 18th International Conference on the Simulation of Adaptive Behavior (SAB 2026)

- **What's New**: 이 논문은 SoftPINCH라는 EMG 구동 부드러운 착용식 외골격 시스템을 제안합니다. 이 시스템은 엄지와 검지의 구부림과 집게 그립에 대한 도움을 제공하며, 텐던 구동의 부드러운 외골격과 손끝의 자석 접촉 센싱 및 신경적 EMG 디코딩을 활용합니다. 이를 통해 임상 및 재활 기술에 새로운 방향성을 제시합니다.

- **Technical Details**: SoftPINCH는 LSTM, CNN+LSTM, CNN+LSTM with attention 등 세 가지의 주제 독립 디코딩 아키텍처를 평가하여 EMG 신호를 실시간으로 디코딩합니다. 연구 결과 CNN+LSTM 모델이 99.4%의 높은 정확도를 기록하였고, 이는 단독 LSTM보다 성능이 우수했습니다. 이 모델은 상대적으로 낮은 구조적 복잡성을 가지면서 실시간 배치 및 기능 평가에 적합한 것으로 나타났습니다.

- **Performance Highlights**: 기능적 평가 결과, SoftPINCH의 활성화 도움으로 분리된 손가락 구부림 및 물체 집기 중 근육 노력의 감소를 확인할 수 있었습니다. 특히, 무게가 있는 물체를 잡을 때 최대 하중에서 92.6%의 근육 노력 저하가 나타났습니다. 이러한 결과는 SoftPINCH가 실시간 EMG 구동의 부드러운 로봇 제어를 통해 직관적이고 낮은 노력을 요하는 집게 그립 지원을 가능하게 함을 입증합니다.



### COP-Q: Safety-First Reinforcement Learning for Robot Control via Cholesky-Ordered Projection (https://arxiv.org/abs/2606.04749)
Comments:
          7 pages, 6 figures, 2 tables

- **What's New**: 이 논문에서는 안전 제약을 충족하면서 수익을 극대화하는 안전한 로봇 제어를 위한 새로운 접근법인 Cholesky-Ordered Projection Q-learning (COP-Q)을 제안합니다. 기존의 방법들과는 달리 COP-Q는 보상 수치와 안전 수치 간의 상관관계를 통합하여 벡터 값 Q-값 추정을 수행합니다. 이러한 접근은 샘플 효율성을 높이고 안전성을 보장하면서 과도한 보수성을 줄이는 데 기여합니다.

- **Technical Details**: COP-Q는 객체 간 공분산을 포함하여 벡터 값 Q-함수에 대한 일반화된 신뢰 구간을 생성하고, Cholesky 분해를 통해 보상에 대한 우선 순위를 순차적으로 표현합니다. 이 방법은 안전성의 보수성을 유지하면서 보상 목표의 과도한 보수성을 적응적으로 줄이는 역할을 합니다. 또한 COP-Q는 대부분의 기존 심층 Q-학습 프레임워크와 호환 가능하며, 컴퓨팅 오버헤드가 최소화됩니다.

- **Performance Highlights**: Brax의 로봇 보행 및 Safety-Gymnasium의 안전한 내비게이션 테스트를 통해, COP-Q는 안전 성능을 강력하게 달성하며, 대표적인 기준선에 비해 경쟁력 있는 또는 개선된 샘플 효율성을 보였습니다. 이는 COP-Q가 안전 성능과 샘플 효율성 간의 균형을 이룰 수 있는 가능성을 보여줍니다.



### CADENCE: Predicting Realized MAPF Execution Time Beyond Sum of Costs (https://arxiv.org/abs/2606.04746)
Comments:
          7 pages, 4 figures, 3 tables and this paper was accepted at Multi-Agent Robotic Systems: Real-World Collaboration and Interaction a workshop at the international conference of robotics and automation (ICRA 2026)

- **What's New**: 이 논문에서는 CADENCE (Coordination and Action-Driven Estimation for Networked Continuous Execution)라는 새로운 접근법을 제안하여 MAPF(Multi-Agent Path Finding) 알고리즘의 평가 격차를 다룹니다. 특히, 사전 실행 정보가 어떻게 최종 완료 시간을 예측할 수 있는지를 실험적으로 조사하였습니다. 총 120개의 계획을 수립하고 15가지 시나리오를 통해 각기 다른 환경에서의 동작 성능을 분석하여, 전통적인 SoC(Sum of Costs) 지표와는 차별화된 요소를 발견했습니다.

- **Technical Details**: 연구는 고정된 7x7 작업 셀에서 7개의 차동 드라이브 로봇을 사용하여 실행 시간을 측정했습니다. 논문에서는 SoC, 원시 운동 부담(primitive motion burden), 상호작용 인식 조정 구조(interaction-aware coordination structure) 등 세 가지 요소를 분석하여, 각 요소가 실행 시간 예측에 미치는 영향을 비교했습니다. 원시 운동 부담은 SoC보다 약 48.6%-59.8%의 평균 절대 오차(MAE)를 감소시키는 가장 강력한 지표로 나타났습니다.

- **Performance Highlights**: 최종 결과는 원시 운동 부담이 SoC에 비해 실행 시간 예측에서 가장 신뢰할 수 있는 추가 신호로 작용한다는 것을 보여주었습니다. 연구는 수행한 실험이 하드웨어에서 실행된 실제 수행 시간 정보로 구성되어, 예측 모델의 정확성을 높이는 방향으로 진전을 이루었다는 점에서 의의가 있습니다. 이 연구는 로봇 간의 협력이 포함된 복잡한 작업 환경에서의 MAPF 알고리즘 성과 향상에 기여할 것으로 기대됩니다.



### CoRe-MoE: Contrastive Reweighted Mixture of Experts for Multi-Terrain Humanoid Locomotion with Gait Adaptation (https://arxiv.org/abs/2606.04718)
Comments:
          Kailun Huang, Zikang Xie, Yanzhe Xie and Panpan Liao contributed equally to this work. Corresponding authors: Renjing Xu and Haohui Huang

- **What's New**: 이번 연구에서는 CoRe-MoE라는 새로운 두 단계 강화 학습 프레임워크를 제안합니다. 이 프레임워크는 보행과 달리기 사이의 매끄러운 전환을 달성하면서 자연스럽고 안정적인 이동을 유지할 수 있도록 설계되었습니다. Terrain-adaptive 전략을 포함하여 사용자는 복잡한 지형에서도 안정적인 보행을 유지할 수 있게 됩니다.

- **Technical Details**: CoRe-MoE는 격자 모델을 통해 보행 생성과 지형 적응을 분리하여 동작합니다. 첫 번째 단계에서는 안정적인 보행 정책을 학습하여 자연스럽고 부드러운 전환 능력을 확보합니다. 이후, 지형 인식 MoE 브랜치가 추가되고, 대비 학습(Objective)을 통해 지형 표현을 구성하여 전환 전문화를 지속적으로 촉진합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면 CoRe-MoE는 성공률, 이동 안정성, 다중 지형 적응성 측면에서 기존 방법들을 능가하는 것으로 나타났습니다. Unitree G1 로봇에서 제로샷 배치를 통해 복잡한 지형에서도 안정적인 보행 성능을 달성하며, 외부 간섭 아래에서도 정확한 발 위치와 동적 안정성을 유지하는 효과를 입증했습니다.



### VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training (https://arxiv.org/abs/2606.04708)
- **What's New**: 이번 연구에서는 Universal Manipulation Interface (UMI) 데이터와 대규모 Vision-Language-Action (VLA) 모델 훈련 간의 두 가지 주요 불일치를 식별하였습니다. 이를 해결하기 위해 VISTA라는 프레임워크를 제안하며, 이는 UMI 데이터를 시각적 및 물리적 요구 사항에 맞추어 조정합니다. VISTA는 UMI-VQA라는 대규모 비전-언어 데이터셋을 구축하고, 물리적 타당성을 확보하기 위해 체계적인 검증 파이프라인을 도입하였습니다.

- **Technical Details**: VISTA는 세 가지 상호 작용하는 구성 요소로 이루어져 있습니다. 첫째, UMI-VQA는 손목 장착 어안 카메라에 최적화된 최초의 대규모 VQA 데이터셋으로, 비주얼 지식과 언어 기반 질문-응답 쌍을 포함합니다. 둘째, 물리적 유효성을 보장하기 위해, 모든 궤적에 대해 데이터를 사전 검증하고 궤적의 연속성, 자기 충돌 위험 및 실행 신뢰도를 평가합니다. 셋째, 두 단계의 공동 훈련 과정을 통해 VLA 모델의 성능을 향상시킵니다.

- **Performance Highlights**: VISTA는 다양한 시뮬레이션 및 실제 조작 작업에서 기존 강력한 기준선인 π₀.₅, LingBot-VLA 및 Wall-X를 눈에 띄게 초월하는 성능을 보여주었습니다. 특히, UMI-VQA를 포함할 경우 정책 성능이 일관되게 향상되었으며, 물리적 검증 점수가 배치 성공을 강하게 예측하는 것으로 나타났습니다. 이를 통해 VISTA는 유망한 로봇 정책 학습을 위한 필수적인 요소로 자리 잡았습니다.



### BPDA-GMM: Bayesian Probabilistic Data Association via Gaussian Mixture Models for Semantic SLAM (https://arxiv.org/abs/2606.04618)
- **What's New**: 이 논문에서는 BPDA-GMM이라는 온라인 베이esian PDA(Probabilistic Data Association) 프레임워크를 제안합니다. 이 프레임워크는 세멘틱 SLAM(동시적 위치추정 및 맵핑) 환경에서 객체 수준의 지도가 확장됨에 따라 발생하는 문제를 해결합니다. 기존 방식들이 고정된 랜드마크 집합을 전제로 두거나, 수동으로 조정된 가설 확률에 의존하는 한계를 극복하기 위해 새로운 방법론을 도입하였습니다.

- **Technical Details**: BPDA-GMM은 Dirichlet-process prior를 이용하여 중국식 식당 프로세스(CRP) 모델을 구축합니다. 이는 축적된 증거가 기존 랜드마크를 선호하도록 하고, 집중 매개변수가 새로운 랜드마크에 확률 질량을 할당하는 구조입니다. 세멘틱 검출 시, 후보 객체는 공동 기반의 세멘틱-기하학적 게이트를 통해 선택되어 CRP-가중 연관 확률이 계산되고, 객체 랜드마크는 세멘틱 가우시안 형태로 갱신됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 최신 기법들에 비해 더 나은 궤적 정확도, 세멘틱 맵 품질, 그리고 인식 혼잡(perceptual aliasing) 및 분류자 오류에 대한 강인성을 보임을 확인하였습니다. 최종적으로, 제시된 구현은 공개적으로 사용 가능하며, 다양한 환경에서의 성능 개선 가능성을 보여줍니다.



### MineXplore: An Open-Source Reinforcement Learning Exploration Benchmark for GNSS-Denied Underground Environmen (https://arxiv.org/abs/2606.04569)
Comments:
          7 pages,11 figures, Submitted to the workshop Xplore:Cross-Disciplinary aspects of Exploration in Robotics, Reinforcement Learning and Search Held at International Conference on Robotics and Automation (ICRA)

- **What's New**: 이번 연구는 GPS가 제한된 지하 광산에서 자율 로봇의 탐색을 위한 새로운 오픈 소스 네비게이션 벤치마크인 MineXplore를 선보입니다. MineXplore는 실제 지하 구리 광산 데이터셋을 기반으로 하며, GPU 가속 학습 파이프라인과 호환되는 시뮬레이션 벤치마크입니다. 이 환경은 복잡한 터널 네트워크를 관리하는 초기 솔루션 부족을 해결하고, 실제 광산 토폴로지를 반영하여 안정적인 정책 학습을 지원합니다.

- **Technical Details**: MineXplore는 Leung et al. (2017)의 칠레 지하 광산 데이터를 사용하여 구축되었으며, 복잡하고 비볼록한 터널 구조를 세밀하게 재현합니다. 이 환경은 104,423 제곱미터의 터널 네트워크를 포함하고, 주요 특징으로는 직사각형과 비대칭 교차 단면, LiDAR에서 유래한 불규칙한 벽 기하학, 세 개의 지형 마찰 구역, 5도의 경사 및 주기적인 조명을 포함합니다. 이러한 요소들은 자율 로봇이 실제 환경에서 내비게이션을 수행하는 데 필요합니다.

- **Performance Highlights**: MineXplore에서 구현된 PPO(Proximal Policy Optimization) 알고리즘은 총 5개의 독립된 무작위 시드로 훈련 video quality를 보내며 최상의 롤링 커버리지는 88.89%로 확인되었습니다. 3개의 시드가 90% 커버리지 목표에 도달하였으며, 이는 실제 지하 환경에서 안정적이고 재현 가능한 정책 학습이 가능하다는 것을 입증합니다. 이 연구는 GPU 가속 시뮬레이션 환경에서의 성능 극대화 가능성을 보여주며, 미래의 지하 탐색 연구에 중요한 기초 자료가 될 것입니다.



### MAD: Mapping-Aware World Models for Agile Quadrotor Fligh (https://arxiv.org/abs/2606.04534)
Comments:
          12 pages, 14 figures

- **What's New**: 본 논문에서는 Mapping-Aware Dreamer (MAD)라는 기하학적으로 인식 가능한 월드 모델을 소개합니다. 이 모델은 비전 기반 쿼드로터 비행을 위한 것으로, 주어진 상황에서 시각 정보로부터 로보센트릭 점유 그리드 맵과 가시성 그리드 맵을 재구성합니다. 기존의 이미지를 직접 재구성하는 방식 대신, MAD는 점유 및 가시성을 함께 학습하여 충돌 회피에 직접 관련된 상태를 인코딩합니다.

- **Technical Details**: MAD는 GPU 병렬 방식으로 작동하는 DiffAero 시뮬레이터에서 훈련되며, 이는 각종 점유 및 가시성 평가를 위해 초당 4.84×10^8 보컬을 처리할 수 있습니다. 이 시스템은 시각적 정보를 빠르게 처리하고, 자율적인 비행을 위한 두 가지 정책 학습 모드인 MAD-Dreamer와 MAD-PPO 및 MAD-SHAC를 통해 성능을 극대화합니다. 또한 이 모델은 부분 가시성과 제한된 라틴성에서도 안정적인 비행을 가능하게 합니다.

- **Performance Highlights**: MAD를 기반으로 한 에이전트들은 비주얼 내비게이션과 레이싱 작업에서 기존 비전 기반 베이스라인보다 더 높은 성공률과 빠른 비행 속도를 기록했습니다. 시뮬레이션에서는 9.66 m/s, 실제 실험에서는 5.05 m/s에 도달하며, 안전한 실내외 비행이 가능하다는 것을 증명했습니다. 이러한 결과는 모듈형 공중 내비게이션과 종단 간 학습 간의 효율적인 균형을 보여줍니다.



### Cooperative Circumnavigation for Multiple Unmanned Surface Vehicles Without External Localization (https://arxiv.org/abs/2606.04518)
Comments:
          17 pages, 15 figures

- **What's New**: 이 논문은 외부 로컬라이제이션 없이 작동하는 다수의 무인 수상 차량(USV)을 위한 협력적 목표 에워싸기 프레임워크를 제안합니다. 이 프레임워크는 USV가 제한된 온보드 센싱을 사용하여 목표 주위에 균일한 원형 형상을 유지하는 것을 목표로 합니다. 제안된 접근법은 USV 간의 비대칭 센싱 관계를 구별하는 이종 감지 전략을 채택하여, 비협력 대상에 대한 향상을 꾀합니다.

- **Technical Details**: 이 논문에서는 최대 코렌트로피 칼만 필터(Maximum Correntropy Kalman Filter, MCKF)와 의사 선형 칼만 필터(Pseudo-Linear Kalman Filter, PLKF)를 사용하여 상대 위치를 추정합니다. 또한, 커플링 오실레이터 기반의 형상 컨트롤러를 설계하여 시스템의 관찰 가능성을 보장하면서 에워싸기를 달성합니다. 이때 센서의 비대칭을 고려하여, 비협력 대상에 대한 통과측정 값은 패시브 센서를 사용하여 수집합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 실제 시뮬레이션을 통해 에워싸기 목표를 효과적으로 달성하며, 관측 가능성을 보장하는 데 성공했음을 보여줍니다. 또한, 이 방법은 비협력 대상의 일시적인 차폐 상황에서도 USV의 상대적 위치를 유지하는 데 강인성을 발휘합니다. 제안된 방법은 측정 노이즈의 영향을 최소화하면서도 신뢰할 수 있는 상대적 로컬라이제이션을 제공하여, 실제 적용 가능성을 더욱 높입니다.



### TransTac: Visuo-Tactile Modality Transition via Ultraviolet-Encoded Transparent Elastomers (https://arxiv.org/abs/2606.04477)
Comments:
          Accepted at IEEE International Conference on Robotics and Automation (ICRA) 2026. 8 pages, 7 figures

- **What's New**: 이 논문에서는 투명한 자외선 인코딩(binocular UV-encoded) 비전 기반 촉각 센서인 TransTac을 소개합니다. 이 새로운 시스템은 시각적 관찰과 마커 기반 촉각 재구성을 단일의 콤팩트 장치로 통합하여 고해상도 접촉 기하학을 복원합니다. 이 기술은 접촉 상태의 신뢰할 수 있는 감지를 위한 혁신적인 방법을 제시합니다.

- **Technical Details**: TransTac 시스템은 UV 반사 마커가 내장된 투명한 엘라스토머를 사용하여 접촉 표면을 시각적으로 추적합니다. Delaunay 기반의 스테레오 매칭 알고리즘을 통해 강력한 삼각측량을 수행하며, 빠른 마커 검출 모델을 개발하여 변형과 조명 조건에서도 안정적인 감지를 가능하게 합니다. 이는 접촉 면의 신뢰성 있는 재구성을 지원합니다.

- **Performance Highlights**: TransTac은 촉각 이미지에서 최대 83.3%의 제로샷 인식 정확도를 기록하여 불투명한 촉각 기준선보다 약 50% 향상된 성능을 보였습니다. 또한, 임베딩 분석 결과 자연 이미지에 대한 크로스 모달 정렬이 증가함을 보여주며, RGB-D 깊이 신뢰성의 저하를 정량적으로 평가하여 비주얼-촉각 통합을 통한 기하학적 범위의 확장을 입증합니다.



### OSCAR: Omni-Embodiment Skeleton-Conditioned World Action Model for Robotics (https://arxiv.org/abs/2606.04463)
Comments:
          Project page: this https URL

- **What's New**: OSCAR는 다양한 로봇 구현에서 일반화할 수 있는 정밀한 비디오 세계 모델로, 로봇 정책 평가에 도움을 줍니다. 기존의 모델들이 실제 로봇 평가에서 여러 가지 도전 과제에 직면해 있음을 인지하고, 이를 해결하기 위한 대규모 데이터 파이프라인을 구축하였습니다. 특히, 다양한 작업과 시나리오를 포함하는 고품질의 통합 훈련 데이터셋을 제공하여, 로봇 정책 평가의 신뢰성을 높이고 있습니다.

- **Technical Details**: OSCAR의 핵심은 2D kinematic skeleton 렌더링을 사용하여 다양한 로봇 팔과 인간 손 동작을 일반화하는 데 집중합니다. 이 skeleton 렌더링은 키네마틱 체인에만 의존하며, 기본 로봇 구현을 변경하더라도 동일한 표현을 사용할 수 있는 장점을 가지고 있습니다. Cosmos-Predict2.5-2B 비디오 모델을 단일 GH200 GPU에서 파인튜닝하여 액션 팔로잉과 모션 일관성이 크게 향상되었습니다.

- **Performance Highlights**: OSCAR는 RoboArena에서의 로봇 정책 성공률을 평가하고, 현실 세계 구현과의 상관관계를 입증하며, 정책 평가 비용 절감과 정책 개발 반복 속도를 높일 수 있는 가능성을 보여줍니다. 기존의 대규모 모델들과 비교할 때, OSCAR는 더 작은 모델로도 뛰어난 성능을 발휘하며, 이러한 결과는 향후 로봇 정책 평가의 새로운 패러다임을 제시하고 있습니다.



### Think Fast and Far: Long-Horizon Online POMDP Planning via Rapid State Sampling (https://arxiv.org/abs/2606.04355)
Comments:
          @inproceedings{Liang2026Thinking, title = {Think Fast and Far: Long-Horizon Online POMDP Planning via Rapid State Sampling}, author = {Yuanchu Liang and Edward Kim and this http URL Knoll and Wil Thomason and Zachary Kingston and Lydia E. Kavraki and Hanna Kurniawati}, year = 2026, booktitle = {International Journal of Robotics Research (to appear)} }

- **What's New**: 이번 연구는 Reference-Based Online POMDP Planning via Rapid State Space Sampling (ROP-RAS3)이라는 새로운 온라인 POMDP 솔버를 제안합니다. 이 솔버는 매우 빠른 샘플링 기술을 활용하여 상태 공간을 샘플링하고 다양한 매크로 액션을 생성합니다. 이를 통해 고품질의 정책을 추론하고, 현대 온라인 POMDP 솔버의 기본 제약인 액션 공간의 소진 없는 해법을 제시합니다.

- **Technical Details**: ROP-RAS3는 매크로 액션의 샘플링을 위해 VAMP(벡터 가속 모션 계획) 프레임워크를 사용하는 참조 기반 POMDP 계획자입니다. 기존 POMDP와의 차별점은 참조가 믿음 상태 전이 대신 정책으로 정의되므로 적합한 매크로 액션을 생성하는 데 용이하다는 것입니다. 이 접근법은 연속 액션 공간을 처리할 수 있는 수학적 근거를 제공하며, 샘플링된 액션 수에 따라 수렴 속도가 달라집니다.

- **Performance Highlights**: ROP-RAS3는 최대 3000개의 탐색 단계와 35차원 상태 공간을 가진 다양한 장기 지평 POMDPs에서 평가되었으며, 다른 최신 방법들에 비해 성공률에서 여러 배 출중함을 보여줍니다. 실제 로봇 시연에서도 ROP-RAS3는 움직이는 보행자를 피하면서 목표로 이동하는 스마트한 동작을 수행하는 유일한 방법으로 실증되었습니다. 본 연구는 ISRR24 논문의 이론과 실증 결과를 확장합니다.



### Instant-Fold: In-Context Imitation Learning for Deformable Object Manipulation (https://arxiv.org/abs/2606.04269)
- **What's New**: 이번 논문에서는 Deformable Object Manipulation (DOM)을 위한 새로운 프레임워크인 Instant-Fold를 제안합니다. Instant-Fold는 단일 인간의 시연을 통해 다양한 조작 모드를 추론하고 실행할 수 있는 imitation learning의 한 형태입니다. 특히, 이 방법은 gradient 업데이트 없이도 다양한 조작 모드를 직접적으로 인식하고 적용할 수 있는 장점이 있습니다.

- **Technical Details**: Instant-Fold는 시간 대비 대조적 사전학습(temporal contrastive pretraining)을 통해 변형에 대한 인지 시각 표현을 학습합니다. 이후, 시연을 조건으로 하는 flow-matching transformer 정책을 이용해 조작 모드에 따라 동작을 예측합니다. 이 프레임워크는 의류 접기와 같은 시뮬레이션 작업에 적용되며, 3D 위치와 사전 훈련된 변형 인식 의미적 특징을 결합한 geo-semantic token으로 물체를 표현합니다.

- **Performance Highlights**: Instant-Fold는 실제 데이터를 추가 수집하거나 미세 조정(finetuning) 없이도 다양한 folding 모드에 일반화하며 실세계 환경에서도 성능을 발휘합니다. 이는 기존의 복잡한 조작 절차를 단순화하고, 하나의 시연만으로도 새로운 동작을 효과적으로 학습할 수 있도록 합니다. 이 연구는 수많은 유사한 연구와 차별화되는 방향으로 DOM 분야에서의 새로운 가능성을 제시하고 있습니다.



### RSC: Decentralized Rigid Formation Flocking for Large-Scale Swarms via Hybrid Predictive Control and Online Reconfiguration (https://arxiv.org/abs/2606.04248)
Comments:
          8 pages, 4 figures, two-column format

- **What's New**: 이 논문에서 제안하는 Rigid Swarm Control (RSC)는 분산형 경량 구조의 형성 집단 운동 제어 프레임워크로, UAV 군집이 복잡한 환경에서 지정된 기하학적 구성을 유지하며 안전하게 이동하도록 설계되었습니다. RSC는 근본적으로 지역적 정보만을 사용하여 각 UAV의 제어를 계산하고, 장애물 회피 및 목표 추적을 통합하여 리드-팔로워 역할 교환 메커니즘을 도입합니다. 이로 인해 RSC는 기하형태 유지를 위한 정확한 거리 제약을 충족할 수 있으며, 임무 수행을 방해하지 않고도 agility를 제공합니다.

- **Technical Details**: 연구에서 제안하는 방법은 3개의 주요 요소로 구성됩니다. 첫째, 제한된 시간 범위 내의 궤적 예측을 통해 UAV의 움직임을 예측하고, 둘째, APF(Artificial Potential Field) 방식의 안정을 유지하면서 궤적을 안내하는 하이브리드 제어 메커니즘을 통하여 지역 최솟값을 방지합니다. 마지막으로, 온라인 리더-팔로워 재구성 전략을 통해 각 UAV는 동적으로 역할을 변경하며, 장애물을 통과한 이후 신속한 재조정을 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, RSC는 25대의 UAV를 사용하여 33~66개의 장애물이 존재하는 환경에서 10%의 상대적인 경계 길이 오류를 유지하며 충돌 없는 작동을 달성했습니다. 기존의 분산형 및 APF 기반 방법은 성공률이 5% 미만인 반면, RSC는 83%의 성공률을 기록하여 복합적인 과제 수행의 신뢰성 높은 성과를 보여주었습니다.



### What Are We Actually Benchmarking in Robot Manipulation? (https://arxiv.org/abs/2606.04233)
Comments:
          31 pages, 6 figures

- **What's New**: 본 연구에서는 로봇 조작의 벤치마크 점수가 실제 조작 능력의 일반적인 지표로서 신뢰할 수 없는 이유를 네 가지 실패 모드로 분류하였습니다. 저자들은 'shortcut solvability', 'lack of statistical significance', 'creeping overfitting', 그리고 'data-source dependence'라는 측면에서 기존의 다섯 가지 벤치마크를 평가하였습니다. 이 연구를 통해 수많은 벤치마크에서 자주 보고되는 점수가 진정한 진전을 측정하는 데에 얼마나 유효한지를 평가할 수 있는 기준을 제시하였습니다. 벤치마크 점수를 진전의 증거로 사용할 때 적용할 수 있는 진단 기준을 개발하여 발표하였습니다.

- **Technical Details**: 본 연구에서는 LIBERO, CALVIN, SimplerEnv, RoboCasa, RoboTwin 2.0와 같은 유명 벤치마크를 대상으로 네 가지 진단 기준을 적용하였습니다. 'shortcut solvability'는 특정 점수를 달성한 모든 정책이 조작 능력을 가져야 함을 의미합니다. 'statistical significance'는 벤치마크 점수의 통계적 유의성을 평가하고, 'creeping overfitting'은 시간이 지남에 따라 모델이 벤치마크에 지나치게 적응하여 일반화 능력이 떨어지는 경우를 분석합니다. 마지막으로 'data-source dependence'는 훈련 데이터와 테스트 데이터의 근접성에 따라 점수가 달라지는 현상을 다룹니다.

- **Performance Highlights**: LIBERO와 CALVIN은 여러 진단 기준을 통과하지 못하며, RoboCasa와 RoboTwin 2.0은 상대적으로 더 나은 성과를 보입니다. 특히 LIBERO에서는 언어 인코더 없이도 0.09B 프로브가 높은 점수를 기록하였고, 대부분의 보고된 성과가 통계적으로 유의하지 않음을 발견했습니다. CALVIN에서는 블록 포즈를 무작위화했을 때 모든 테스트 정책의 성과가 떨어졌습니다. 저자들은 벤치마크 점수의 해석을 도와줄 수 있는 네 가지 진단 기준을 공개하였으며, 새로운 벤치마크 작성자는 이를 활용하여 점수를 발표하기 전 검증할 수 있습니다.



### PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification (https://arxiv.org/abs/2606.04226)
Comments:
          Accepted at ICRA 2026 (Vienna); published on arxiv for archival purposes. See also this https URL

- **What's New**: 이번 연구에서는 PerceptTwin이라는 자동화된 파이프라인을 소개하며, 로봇의 인식 스택에서 생성된 의미 장면 표현을 기반으로 상호작용 가능한 시뮬레이션을 구축합니다. 이는 기존의 수동적인 시뮬레이션 생성 과정의 대안을 제시하며, 로봇 계획의 검증과 수정에 있어 큰 잠재력을 가지고 있습니다. 특히 PerceptTwin은 개방형 어휘(Open-Vocabulary) 객체 맵, 3D 자산 생성, 그리고 일반적인 조건 검사를 통합합니다.

- **Technical Details**: PerceptTwin은 3D 자산을 생성하거나 찾아내고, 인식된 객체 포인트 클라우드를 사용하여 이들을 지역화합니다. 또한 로봇-객체 친화성(affordance)을 예측하고 시뮬레이션 내에서 계획을 테스트할 수 있는 기능을 가지고 있습니다. 연구에서는 LLM(대형 언어 모델)을 활용하여 계획의 정확성을 검증하는 LLM 판별기를 도입하여, 계획이 인간의 선호와 일치하는지 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면, PerceptTwin을 통한 피드백은 계획의 성공률을 평균 39% 향상시키고, 인간의 계획 검증 정확성을 평균 18% 개선하는 것으로 나타났습니다. 이는 PerceptTwin이 로봇 계획의 안전성과 신뢰성을 높일 수 있는 기초 역할을 할 것으로 기대됩니다. 코드 및 자료는 오픈 소스로 제공되어, 연구자들이 활용할 수 있습니다.



### Towards Estimating Normal and Shear Interface Pressures in Prosthetic Sockets via Least Squares and Mechanics Modeling (https://arxiv.org/abs/2606.04222)
- **What's New**: 이 논문은 의치 소켓의 피팅 과정을 자동화하고 객관화하기 위해 새로운 테스트베드를 소개합니다. 이 테스트베드는 두 가지 보완적인 검증 신호를 사용하여 희소한 압력 센서 아래에서 모델 성능을 평가합니다. 목표는 전반적인 하중과 국소적인 하중 측정을 효과적으로 분석하는 것입니다.

- **Technical Details**: 테스트베드는 소켓 조립체와 파일론 조립체로 구성되어, 인공 잔여 리브가 소켓에 삽입되어 통제된 접촉 조건을 만듭니다. 6축 로드 셀을 사용하여 전반적인 힘과 모멘트를 측정하며, 내부에는 압력 센싱 클러스터가 배치되어 국소 하중을 제공합니다. 이러한 구성은 물리적 모형을 위한 신뢰할 수 있는 데이터를 제공하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 정적 하중 하에서의 검증 결과, 상수 바이어스 항을 추정하면 느슨한 채널의 안정적인 오프셋을 감소시키고, 국소 측정과의 일치를 향상시킵니다. 또한 파레토 전선 민감도 분석을 통해 바이어스 항을 포함할 때 글로벌 및 로컬 목표 간의 균형이 어떻게 변화하는지를 보여주고 있습니다.



### DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics (https://arxiv.org/abs/2606.04206)
Comments:
          ICML 2026, the project page: this https URL

- **What's New**: 이 논문에서는 변형 가능한 선형 물체(Deformable Linear Objects, DLO) 조작에 대한 새로운 접근 방식을 제안하고 있습니다. 기존 연구들은 특정 작업에 집중했으나, 본 연구는 다양한 물질의 특성을 반영한 미분 가능 시뮬레이터(Differentiable Simulator)를 도입하여 범용적인 DLO 조작을 가능하게 합니다. 이 시뮬레이터는 다양한 물질 행동을 모델링하여 조작 기술을 학습하고 평가할 수 있는 강력한 기반을 제공합니다.

- **Technical Details**: 제안하는 미분 가능 시뮬레이터 DLO-Lab은 DLO와 다른 물체 간의 상호작용을 모델링합니다. 여기에는 신축성(elasticity), 굽힘 플라스틱성(bending plasticity) 및 복잡한 상호작용을 포함한 다양한 물질 속성을 지원하는 커스터마이즈된 솔버가 포함됩니다. 이 시뮬레이션 엔진은 로봇 조작 기술을 습득하는 데 최적화되어 있으며, 여러 작업에서 효율적인 정책 학습이 가능하도록 설계되었습니다.

- **Performance Highlights**: DLO-Lab을 기반으로 한 벤치마크 과제들은 DLO 조작의 특징과 능력을 강조합니다. 다양한 정책-학습 알고리즘을 평가한 결과, 미분 기반 방법이 높은 샘플 효율성을 보이는 반면, 불연속 접촉 동역학이 있는 작업에서는 샘플링 기반 궤적 최적화 방법이 더욱 견고한 성능을 보여줍니다. 최종적으로, 시뮬레이터에서 최적화된 정책이 실제 하드웨어에 효과적으로 전이되는 것을 입증하여 시뮬레이션과 현실 간의 간극을 줄이는 데 기여했습니다.



### Distribution-Free Risk-Aware Planning and Control Under Uncertainty Using Conformal Spectral Risk Contro (https://arxiv.org/abs/2606.04185)
Comments:
          Submitted to IEEE Robotics and Automation Letters

- **What's New**: 이 논문에서는 고유의 불확실성 분포에 대한 가정을 요구하지 않고 사용자가 지정한 임계값 아래에서 위험 제어를 보장하는 위험 인식 모델 예측 제어(RA-MPC) 프레임워크를 제안합니다. 또한, 준수형 위험 제어(CRC)를 일반 스펙트럼 위험 척도로 확장하여 배포 독립적인 위험 정량화 프레임워크를 개발합니다. 이 방법론을 통해 모형 예측 제어(MPC) 프레임워크에 스펙트럼 위험 제어를 통합하여 통계적 안전 보장(guarantee)을 제공합니다.

- **Technical Details**: 제안된 RA-MPC 프레임워크에서는 위험 추정에 대한 가정을 필요로 하지 않고, 직접 스펙트럼 위험을 제어하기 위해 예측 세트를 생성합니다. 이러한 예측 세트는 진정한 불확실성과 관련된 통계적 안전성을 확보하는 데 사용됩니다. 기존 RA-MPC 방법들은 대부분 불확실성 분포의 정확한 특성을 요구하지만, 우리는 이를 통해 이러한 제한을 극복했습니다.

- **Performance Highlights**: 우리는 제안한 RA-MPC 프레임워크를 정적 및 동적 장애물 회피 시나리오에서 검증했습니다. 실험 결과, 기존 RA-MPC 프레임워크에 비해 안전성이 개선되고 해결 시간(solve time)이 단축되었음을 보여주었습니다. 이로 인해 제안된 모델의 실제 적용 가능성과 안전성을 강조할 수 있었습니다.



### Affordance2Action: Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipulation (https://arxiv.org/abs/2606.04172)
Comments:
          23 pages

- **What's New**: 이 논문은 복잡한 장면 내에서 작업 조건에 따라 기능적 부분을 정확히 정의하는 Affordance2Action (A2A)이라는 프레임워크를 소개합니다. 기존 연구에서는 개체 카테고리에 기반한 단일 지침-지역 대응을 가정하지만, 본 연구는 하나의 지침이 여러 기능적 지역과 상호 작용할 수 있는 구조를 강조합니다. A2A-Bench는 일상 장면에서 단일 및 다중 지역 지침에 대한 조사를 통해 이를 효과적으로 해결합니다.

- **Technical Details**: A2A는 대규모 자연 이미지를 활용하여 조작과 관련된 affordance를 구분하기 위해 언어 모델 필터링을 적용하며, 반복적인 마스크 수정 및 인터랙티브 분할을 통해 유효한 기능적 지역을 주석 처리하는 A2A-AffordGen을 구축합니다. A2A-GroundingModel은 이미지-지침 쌍에서 affordance 마스크를 직접 예측하여, 정밀한 조작을 위한 공간적 우선 순위를 제공합니다. 이 시스템은 다양한 작업 조건의 기능적 지역을 추적할 수 있도록 지원합니다.

- **Performance Highlights**: A2A는 일반적인 분할, VLM 기반의 기반 grounding, 그리고 affordance 증류와 같은 기존 기준에서 상당한 차이를 보여줍니다. 본 연구의 실험 결과는 과업-수준 현지화 및 하류 조작에 대한 유용한 공간적 우선 순위를 제공하는 데 있어 A2A의 효과적인 성능을 강조합니다. 모든 데이터셋과 코드는 공개되어 오픈 리서치를 촉진할 예정입니다.



### Multi-Agent Next-Best-View Optimization for Risk-Averse Planning (https://arxiv.org/abs/2606.04158)
Comments:
          8 pages, 5 figures. Submitted to IROS 2026

- **What's New**: 본 논문에서는 불확실하고 알려지지 않은 환경에서의 안전한 경로 계획을 위한 분산형 다중 에이전트 Next-Best-View (NBV) 선택 프레임워크를 제안합니다. 중앙 집중식 접근 방식의 한계를 극복하기 위해, 각 로봇은 자체적인 로컬 3D Gaussian Splatting 맵을 유지하며 팀 전체의 기대 정보 획득 (Expected Information Gain, EIG)을 극대화하는 방향으로 협력합니다. 이 과정에서 로봇들은 오직 후보 관점과 계획된 경로 설명을 교환하며, 통신량을 대폭 줄이고 공간적 확장성을 개선합니다.

- **Technical Details**: 제안된 시스템은 Consensus ADMM (C-ADMM)라는 알고리즘을 사용하여 분산된 목적을 해결합니다. 각 로봇은 평균 가치 위험 (Average Value-at-Risk, AV@R)을 통해 경로의 충돌 위험을 모델링하며, 이는 마스킹 반경을 형성하고 경로 점수를 매기는 데 사용됩니다. 3DGS(3D Gaussian Splatting) 모델을 기반으로 하는 환경 표현은 로봇이 미래 경로의 안전성을 판단할 수 있는 수단을 제공합니다.

- **Performance Highlights**: Gibson 환경에서의 실험 결과, 제안된 분산 접근법은 중앙 집중식 기반선과 유사한 매핑 품질과 경로 안전성을 달성하면서도 통신 효율성을 상당히 향상시켰습니다. 이 연구는 다양한 팀 규모에서의 평가를 통해, 위험 인지와 정보 획득을 동시에 고려한 효율적인 로봇 팀 작업을 가능하게 합니다.



### Selecting haptic guidance models in teleoperation: guidelines from a comparative user study (https://arxiv.org/abs/2606.04157)
Comments:
          EUROHAPTICS 2026 - EuroHaptics International Conference, Jul 2026, Sienna, Italy

- **What's New**: 이 연구에서는 텔레오퍼레이션(teleoperation)에서 햅틱 가이드(haptic guidance)가 운영자의 성능을 향상시키는 방법을 제시합니다. 여러 환경과 작업을 고려하여 가장 적합한 모델을 선택하기 위한 지침을 제공합니다. 상용 모델(스프링-댐퍼(spring-damper), 포텐셜 필드(potential field), 가이드 튜브(guiding tube))을 통합된 형태로 정의하였습니다.

- **Technical Details**: 이 논문에서는 스프링-댐퍼 시스템의 변형으로서 일반적인 모델들을 제시하며, 각 모델에 대한 특정 가이드를 제공합니다. 사용자는 수직 농업 작업을 중심으로 세 가지 고전 모델에 대한 사용자 연구를 수행하였고, 다양한 환경 조건 하에서 평가하였습니다. 이를 통해 모델의 성능이 환경에 따라 달라진다는 것을 확인했습니다.

- **Performance Highlights**: 연구 결과, 모든 모델이 각기 다른 상황에서 강점을 보였으며, 스프링-댐퍼는 복잡한 환경에서 우수한 성능을 보였고, 포텐셜 필드는 장애물 근처에서 위험을 나타냈습니다. 가이드 튜브는 균형 잡힌 하이브리드 선택지를 제공하였습니다. 또한, 가이드 힘의 크기가 편안함과 신뢰 점수에 직접적으로 연관 있음을 보여주는 새로운 목표 매트릭스를 제안했습니다.



### CoPark: Learning Reactive Parking via Self-Play (https://arxiv.org/abs/2606.04149)
- **What's New**: 이 논문은 안전하게 상호작용하는 여러 자동차가 주차 슬롯에 정확히 도착하는 고급 목표를 동시에 달성하는 정책 학습 문제를 다룹니다. 제안된 CoPark 학습 알고리즘은 다중 에이전트 자가 플레이(self-play) 방식으로 작동하며, 고정된 기하학적 계획을 따르는 동시에 상대의 위협을 감지하여 반응적인 교정을 학습하는 방법입니다. 이를 통해 주차 매뉴버 중 주변 차량과의 상호작용을 원활히 하면서 서브 미터 정확도를 달성합니다.

- **Technical Details**: CoPark는 고정된 행동 우선순위를 제공하는 사전 계산된 오프라인 계획과 반응적인 교정을 학습하는 잔여 정책(Residual Policy) 헤드를 결합합니다. 이 구조는 다중 에이전트 자가 플레이 과정에서 상호작용 행동을 학습할 수 있도록 설계되었습니다. 차별화된 채널에서 위협 신호를 사용하여 고정 기초와 잔여 정책 간의 우선권을 조정하고, 자동차의 행동을 제어하여 협조적인 주차를 가능하게 합니다.

- **Performance Highlights**: CoPark는 Dragon Lake Parking(DLP) 및 DeepScenario Open 3D(DSC3D)의 새로운 리액티브 주차 벤치마크에서 70-85%의 성공률과 3-6%의 충돌률을 기록하며 기존의 기법을 크게 초월합니다. 이 시스템의 결과는 reverse-yielding, mid-maneuver yielding, tight-corridor passing, queuing과 같은 진화적 상호작용 행동을 보여줍니다. 이는 기존 방법들이 처리하지 못했던 복합적인 상호작용 조건을 해결하고, 자율 주차의 새로운 가능성을 제시합니다.



### CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization (https://arxiv.org/abs/2606.04130)
Comments:
          8 pages, 15 pages of supplementary material

- **What's New**: CLAW는 행동 없는 비디오로부터 지속적인 잠재 행동 표현과 세계 모델을 동시에 학습하기 위한 완전한 자기 지도(Self-supervised) 프레임워크입니다. 이 접근 방식은 적대적 잠재 정규화(adversarial latent regularization)와 확산 기반 비디오 생성을 활용하여, 비디오에서 구조적이고 의미 있는 행동 표현을 포착합니다. 이를 통해, 이 모델은 행동 레이블이나 주석 없이도 환경의 역동성을 효과적으로 모델링할 수 있습니다.

- **Technical Details**: CLAW는 연속 잠재 행동 표현을 학습하는 Latent Action Model(LAM)과 세계 모델을 동시에 훈련합니다. 이 방법은 자가 지도(Self-supervised) 방식으로, 세계 모델이 LAM에게 정보를 제공하고, 반대로 LAM이 세계 모델을 보조합니다. 또한, 적대적 잠재 정규화 전략을 도입하여 의미 있는 잠재 행동을 촉진하고 미래 정보 유출을 방지합니다.

- **Performance Highlights**: CLAW는 관찰로부터 모방 학습(imitation learning)과 목표 지향 계획(goal-directed planning)을 지원합니다. 경험적 실험 결과, CLAW는 의미 있는 잠재 행동 표현을 생성하고, 행동 전이(action transfer)를 지원하며, 관찰로부터 계획 및 모방 학습을 가능하게 합니다. 이러한 성능은 기존 방법들에 비해 뛰어난 결과를 보여주었습니다.



### AgenticDiffusion: Agentic Diffusion-based Path Planning for Vision-Based UAV Navigation (https://arxiv.org/abs/2606.04111)
- **What's New**: 이 논문에서는 AgenticDiffusion이라는 새로운 다중 시점 UAV 내비게이션 프레임워크를 제안합니다. 이 프레임워크는 언어에 기반한 추론, 오픈 어휘 타겟 그라운딩, 비전 기반 확산 계획, 그리고 비선형 모델 예측 제어(NMPC)를 통합하여 UAV의 내비게이션을 예측합니다.

- **Technical Details**: AgenticDiffusion 프레임워크는 사용자의 자연어 지시를 기반으로 동기화된 1인칭 뷰(FPV) 및 상단 뷰(observations) 관측을 분석합니다. 이 프레임워크는 Grounding DINO를 사용하여 세부 목표를 로컬라이즈하고, 두 관측 시점에서의 시각적 특성을 평가하여 내비게이션에 가장 적합한 관점을 결정합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 40회의 실제 UAV 내비게이션 실험에서 80%의 임무 성공률을 달성했습니다. 또한, 확산 계획기는 100%의 경로 생성 성공률을 기록하여 매우 우수한 성능을 입증했습니다.



### CADET: A Modular Platform for Evaluating Distributed Cooperative Autonomy in Connected Autonomous Vehicles (https://arxiv.org/abs/2606.04072)
- **What's New**: CADET(협력 자율성 분산 실험 도구킷)는 기존의 단일 플랫폼 실행 방식을 벗어나, 자율차(Autonomous Vehicle, AV) 스택을 모듈화하여 차량, 도로 인프라(RSU), 엣지/클라우드 서버에 걸쳐 배포할 수 있도록 설계되었습니다. 이 플랫폼은 협력적 자율성을 시스템적으로 평가할 수 있는 환경을 제공하며, V2V(차량 간), V2I(차량과 인프라 간)의 협력을 통해 안전성을 극대화할 수 있는 방법을 탐구합니다.

- **Technical Details**: CADET는 AV 스택의 오프라인 테스트 및 실시간 성능 모니터링을 위한 다중 수준의 계측을 통합하며, 다양한 모델 아키텍처의 유연한 배포를 지원합니다. 이러한 설계는 최신 AI 모델과 협력 정책을 잘 지원하며, NetWaggle(Network Emulation Layer)를 통해 현실적인 네트워크 시뮬레이션을 가능하게 합니다. 이를 통해 다양한 기술적 과제를 효과적으로 해결할 수 있도록 돕습니다.

- **Performance Highlights**: CADET의 실험 결과, V2V 의도 패킷이 클라우드 기반 인식보다 높은 성능을 보여주었으며, RSU 지원 인식이 처리 용량이 포화될 때까지 안전성을 높이는 데 기여했습니다. CADET은 또한 실제 자율주행 환경에서 데이터셋 기반 실험을 지원하여 연구자들이 분산 추론 작업을 독립적으로 벤치마킹할 수 있도록 하여 연구의 전반적인 가속화를 도모합니다.



### CIPER: A Unified Framework for Cross-view Image-retrieval and Pose-estimation (https://arxiv.org/abs/2606.05011)
Comments:
          16 pages, 5 figures

- **What's New**: 본 연구에서는 cross-view geo-localization 문제를 새로운 방식으로 접근하여, 대규모 도시 이미지 검색과 정밀한 3-DoF 포즈 추정을 동시에 수행하는 통합 프레임워크를 제안합니다. 기존의 두 독립적인 접근 방식을 통합하여 반복적인 오류 전파와 비일관적인 특징 표현 문제를 해결하고자 하였습니다. 새로운 아키텍처 CIPER(Cross-view Image-retrieval and Pose-estimation transformER)는 상호 이익을 주는 특징 학습을 통해 두 가지 작업을 함께 수행합니다.

- **Technical Details**: CIPER는 공유되는 transformer encoder와 전용 pose decoder를 사용하여 이미지 검색과 포즈 추정을 동시에 처리합니다. 이 구조는 글로벌 검색 특징과 공간적 위상 단서를 분리하고, 두 방향 transformer pose decoder를 통해 지상 특징을 공간적 쿼리로 활용하여 양방향 크로스 어텐션을 수행합니다. 또한, 세트 예측 전략을 채택하여 통합된 다중 작업 목표 하에 안정적인 3-DoF 회귀를 가능하게 합니다.

- **Performance Highlights**: VIGOR, KITTI 및 Ford Multi-AV 데이터셋을 이용한 실험 결과, CIPER는 제한된 시야 및 임의의 방향 조건에서도 특히 경쟁력 있는 성능을 보였습니다. 양방향 크로스 어텐션의 정렬 및 세트 예측 전략 덕분에 기존 방법보다 훨씬 더 높은 정확도를 달성하였으며, 다양한 실세계 응용에 적합한 강력한 기준선으로 자리잡을 가능성이 높습니다.



### What Can Eye Gaze Teach Us About Real-World Cycling? Insights From the Oxford RobotCycle Projec (https://arxiv.org/abs/2606.04989)
- **What's New**: 이 논문에서는 영국 옥스포드에서 사이클링 안전에 대한 인식을 연구하며, 웨어러블 안구 추적 안경을 통해 다양한 환경과 사건 아래에서의 인식 차이를 밝혀냅니다. 특히, 자전거 도로, 자동차 도로 및 공유 버스 도로 사용 시 각기 다른 인지적 도전이 있음을 보여줍니다. 교차로 별로 안구 고정 패턴이 다르며, 이는 자전거 타는 사람의 스트레스에 대한 시사점을 제공합니다.

- **Technical Details**: 연구는 2024년과 2025년 사이 몇 차례의 RobotCycle 실험을 통해 수행되었습니다. 안구 추적 패턴은 자전거 도로, 자동차 도로 및 자전거 보호구역과 같은 경로에서의 다양한 상황을 이해하기 위해 측정되었습니다. 또한, 급정거, 보행자와의 근접 통행과 같은 사건이 있을 때와 없을 때의 안구 고정 패턴의 차이점도 분석되었습니다.

- **Performance Highlights**: 안구 추적을 통해 자전거 타는 경험에 대한 스트레스를 쉽게 분별할 수 없다는 한계가 있음을 발견했습니다. 실험에서는 각기 다른 교차로 유형에서 안구 고정 시간을 비교하였으며, 특정 사건의 존재가 자전거 이용자의 인지적 반응에 미치는 영향을 평가했습니다. 향후 연구에서는 물체 인식 및 분할 방법을 결합하여 더 정확한 스트레스 추정이 가능할 것으로 예상됩니다.



### Z-FLoc: Zero-Shot Floorplan Localization via Geometric Primitives (https://arxiv.org/abs/2606.04788)
- **What's New**: 이번 연구에서는 리트레이닝(curtain retraining) 없이 새로운 환경에서도 일반화할 수 있는 제로샷(zero-shot) 바닥 계획서(floorplan) 로컬라이제이션 방법을 제안했습니다. 이 방법은 인류가 만든 환경에서 광범위하게 존재하는 기하학적 원시(primitives)인 선(line)과 원(circle)을 활용하여 시각적 모양 변화에 영향을 받지 않는 구조적 제약을 제공합니다. 실험 결과, 이 접근법이 기존의 데이터 기반 학습 방법보다 성능이 우수하다는 것을 보여주었습니다.

- **Technical Details**: 제안된 방법은 단일 카메라의 RGB 이미지 시퀀스를 기반으로 2D 바닥 계획서(floorplan)와의 교차 모달 매칭(cross-modal matching) 문제를 다룹니다. 이를 위해 BEV(떨어진 시점의 보기, Bird’s Eye View) 맵을 재구성하고, 여기서 선과 원의 원시를 추출합니다. 후보 유사도 변환(similarity transformation)은 랜덤 샘플을 기준으로 형성하며, 각 원시 집합에서 독립적인 자세 가설을 생성합니다.

- **Performance Highlights**: 시뮬레이션된 데이터와 실제 데이터셋 모두에서 실험을 진행한 결과, 제안된 방법이 특정 환경에 맞추어 훈련된 기존의 최첨단 학습 기반 방법보다 일관되게 우수한 성능을 보였습니다. 특히, 모든 실험에서 사용된 하이퍼파라미터는 고정되어 있어, 다양한 환경에서 안정적으로 일반화될 수 있음을 입증했습니다.



### 3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training (https://arxiv.org/abs/2606.04436)
- **What's New**: 이번 논문에서는 3D 사고를 유도하는 공동 훈련 프레임워크인 3DThinkVLA를 제안합니다. 이 프레임워크는 시각-언어-행동(Vision-Language-Action, VLA) 모델이 행동 예측을 수행할 때 암묵적으로 3D 공간 추리를 가능하게 합니다. 기존의 방법들은 대부분 2D 이미지에 의존하여 3D 공간 추리와의 심각한 간극을 보완하지 못했습니다. 우리의 접근법은 3D 기하학 인식과 3D 공간 추리를 서로 분리함으로써 이러한 한계를 극복하고 있습니다.

- **Technical Details**: 3DThinkVLA 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 잠재 3D 기하학 인식 모듈은 VLM 백본을 수정하지 않고 3D 기하학 정보를 모델에 주입합니다. 2) 온라인 3D 추론 증류 모듈은 공유된 추론 앵커 토큰을 사용하여 프롬프트에 의해 유도된 추론 격차를 완화합니다. 3) 공간적으로 증강된 행동 통합 모듈은 이해된 기하학과 추론 기능을 행동 쿼리 토큰에 결합하여 예측의 Robustness를 높입니다.

- **Performance Highlights**: 본 논문에서 제안된 방법은 LIBERO, LIBERO-PLUS, SimplerEnv 및 다양한 실제 조작 작업에서 최첨단 성능을 달성했습니다. 3D 입력 없이 2D 이미지만으로도 효과적으로 3D 공간 추리를 수행하여, VLA 모델의 사전 훈련된 의미적 정렬을 완전히 보존합니다. 이는 3D 센서나 외부 모델, 명시적 텍스트 생성을 요구하지 않으면서도 성능을 극대화하는 전략입니다.



### When Freshness Is Not Enough: Distribution-Aware Age of Information for Networked LQR Contro (https://arxiv.org/abs/2606.04361)
- **What's New**: 이 논문에서는 정보의 신선함(freshness)을 측정하는 표준으로 떠오른 Age of Information (AoI)가 네트워크 제어 시스템에서 최적의 기준이 될 수 있는지를 검토합니다. 특히 평균 AoI(Mean AoI) 또는 최고 AoI(Peak AoI)가 실제 성능을 대체하는 방법이 증명되지 않았음을 강조하며, 두 가지 다른 스케줄링 정책이 같은 평균 AoI를 가질지라도 매우 다른 간격 분포를 가질 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 상태 독립적 스케줄링 정책을 사용하고, 무한 수명 LQR(Linear Quadratic Regulator) 추적 문제를 간격 분포에 대한 최적화로 환원하는 방법을 제시합니다. 특히 불안정한 다이나믹스와 지연 피드백의 영향을 받는 상황에서, 이 문제는 평균 값을 넘어서서 고차 통계 모멘트(higher-order moments) 및 지수 모멘트(exponential moments)에 의존한다는 것을 보여줍니다.

- **Performance Highlights**: 실제 차량 궤적 데이터를 사용하여 이론을 검증하였으며, 평균 AoI 단독으로는 부족하다는 것을 입증했습니다. 이론적 비용은 관찰된 성능 경향을 잘 포착함을 보여주어, 기존의 AoI 최소화 접근 접근 방식에서의 한계를 시사합니다.



### Dual Advantage Fields (https://arxiv.org/abs/2606.04188)
Comments:
          Accepted by ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning

- **What's New**: 이번 연구에서는 오프라인 목표 조건 강화 학습에서 차량을 많이 교체하는 문제를 다룬다. Dual Advantage Fields (DAF)라는 정책 추출 방법을 도입하여, 이 방법이 어떻게 전역 가치 필드를 지역적 이점 신호로 변환하는지를 설명한다. Dual goal representations을 활용하여, 목표 방향과 일치하는 행동을 평가하는 새로운 방법론을 제시한다.

- **Technical Details**: DAF는 동작 효과 모델을 학습하고 행동 선택을 위한 정량적인 평가 지표를 생산하여, 기하학적 테스트를 통해 지역적인 이점을 계산한다. 이는 상태 표현의 변형으로 나타나며, 목표의 방향과 정렬되는 행동을 선호하게 된다. 이 방법은 고정된 오프라인 데이터에서 정책을 추출하며, 별도의 목표 조건 행동 가치 함수 학습 없이 동작한다.

- **Performance Highlights**: 실험 결과, DAF는 OGBench에서 운동, 조작, 퍼즐 작업을 수행하는 데 있어 성능이 향상되었다. 지역적으로 올바른 행동이 목표로의 직접적인 이동과는 차별화되는 환경 설정에서 강력한 성과를 보였다. DAF는 오프라인 GCRL(Goal-Conditioned Reinforcement Learning)에서 일관된 성능을 제공하며, 다양한 도메인에 대해 효과적으로 작동한다.



### Semantic Constraint Synthesis for Adaptive Trajectory Optimization via Large Language Models (https://arxiv.org/abs/2606.04123)
Comments:
          7 pages, 4 figures, Presented as a short paper at IEEE CVPR 2026, AI4Space Workshop

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)을 활용하여 자연어로 표현된 임무 요구사항 및 제약조건을 실행 가능한 궤적 최적화 코드와 해당 수학적 공식으로 변환하는 새로운 프레임워크를 제안합니다. 이는 우주 임무의 증가하는 빈도와 복잡성으로 인해 효율적인 궤적 최적화 문제의 수립이 필요함을 강조합니다. LLM을 통해 임무 의도를 높은 수준에서 설명하고, 이를 정형화된 최적화 모델로 연결할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 이 연구는 궤적 최적화 문제를 해결하기 위해 자연어 임무 요구사항을 정형적 수학 공식으로 변환하는 자동화된 구조를 세 가지 단계(수학 표현 생성, 실행 가능한 코드 생성, 수치 OCP 솔버 실행)로 설계했습니다. 베이스 OCP 문제와 새로운 임무 요구사항을 결합해 LLM에 의해 최적화 문제를 자동으로 구성할 수 있도록 하며, Python을 사용하여 코드를 생성합니다. 주요 입력으로는 기본 OCP 코드, 기본 OCP 수식, 임무 요구사항 텍스트가 포함되며, LLM은 이들을 기반으로 코드 생성 및 오류 처리를 진행합니다.

- **Performance Highlights**: 제안된 프레임워크는 우주선 근접 작전 시나리오를 대상으로 시험되었으며, 세 가지 대안 프레임워크와 비교하여 성능을 평가했습니다. 실험 결과, LLM 기반의 접근 방식은 각 경우에서 높은 성공률을 보였고, 특히 수학적 제약 조건을 토대로 한 재구성에서 신뢰성을 입증하였습니다. 이 연구는 우주 차량의 궤적 설계에서 LLM의 활용 가능성을 강조하며, 복잡한 임무 요건을 간결하게 처리할 수 있는 새로운 접근법을 제시합니다.



### Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation (https://arxiv.org/abs/2606.04046)
Comments:
          Accepted at ICML 2026

- **What's New**: 본 논문에서는 SceneDiver라는 새로운 방법을 제안하여, 시각-언어 의사결정(vision-language decision making) 과제에서의 인식 한계(perceptual limitation)를 극복하고자 합니다. 기존의 시각-언어 모델(VLMs)과 시각-언어-행동 모델(VLAs)이 각각의 장점을 갖고 있지만, 시각적 환각(visual hallucinations) 문제로 인해 성능 제한을 겪고 있습니다. SceneDiver는 장기 계획 능력을 활용하여, 먼저 전체 장면 그래프(scene graph)를 구축하고 이를 통해 작업을 간단한 하위 문제로 분해하는 방식으로, 효과적으로 중요한 객체에만 집중할 수 있도록 설계되었습니다.

- **Technical Details**: SceneDiver의 중심은 거친 단계에서 세밀한 단계로 진행되는 초점(focus) 계획 수립입니다. 첫 번째 단계에서는 이미지 데이터를 구조화된 그래프 표현으로 변환하여 장면을 전반적으로 이해합니다. 두 번째 단계에서는 VLM이 각 지역 하위 장면을 탐색하여 중요한 객체를 식별하도록 합니다. 또한, 실시간 의사결정에 필요한 지연 시간을 충족하기 위해 가벼운 어댑터(adapter)를 설계하여 VLA 모델에서 효과적인 초점 능력을 추출합니다.

- **Performance Highlights**: 다양한 로봇 조작 및 방 탐색 과제를 통해 실험한 결과, SceneDiver는 조작 작업에서 10%-15%, 탐색 작업에서 최대 16%의 성능 향상을 보여주었습니다. 또한, LIBERO-plus 벤치마크에서 성공률이 9.6% 개선되었으며, 이는 의사결정의 강건성을 향상시키는데 기여했습니다. 이 모든 성능 향상과 함께 계산 효율성도 유지되었으며, 실시간 배포에 적합합니다.



New uploads on arXiv(cs.MA)

### Channel Fracture: Architectural Blind Spots in Scheduled Cross-Agent Memory Injection for Multi-Agent Orchestration Systems (https://arxiv.org/abs/2606.04896)
Comments:
          16 pages, 0 figures

- **What's New**: 다중 에이전트 AI 팀의 발전이 지식 전이의 새로운 요구를 만들어 내고 있습니다. 특히, 에이전트들이 지속적인 메모리를 사용할 때, 하향식 팀 아키텍처 내에서 정보가 상호 주입될 필요성도 증가하였습니다. 본 논문에서는 'channel fracture'라는 체계적인 실패 모드를 발견하였으며, 이는 특정 에이전트가 다른 에이전트의 자산에 접근할 수 없는 상황을 설명합니다. 이 문제를 해결하기 위해 CADVP라는 새로운 검증 프로토콜을 제안하고 두 가지 설계 원리를 명시합니다.

- **Technical Details**: Hermes Agent는 다중 에이전트 오케스트레이션을 지원하는 오픈 소스 AI 에이전트 프레임워크로, 여러 기능 도메인에 따라 구성된 프로파일로 작동합니다. 각 프로파일은 자체 메모리 저장소(SQlite 사용), 구성 파일(config.yaml) 및 게이트웨이 프로세스를 가지고 있습니다. 연구 결과, cron 작업이 메모리에 접근할 수 없게 되는 'channel fracture' 문제를 발견하였으며, 이는 특정 아키텍처 제약으로 인해 발생하게 됩니다.

- **Performance Highlights**: 우리는 CADVP v1.1을 통해 13차원의 검증 프로토콜을 제안하고, 에이전트 간의 지식 전달을 인증하는 새로운 기준을 설정합니다. 이 프로토콜은 채널 유효성을 확인하는 단계(CC-0)를 포함하여, 잘못된 긍정적 확신을 방지합니다. 실험 결과, 가장 자연스러운 채널인 cron 위임 쓰기 채널은 실패하는 반면, 우회 채널은 성공하는 모습을 보여주었습니다.



### Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems (https://arxiv.org/abs/2606.04306)
Comments:
          13 pages, 2 figures

- **What's New**: 이 논문에서는 실행이 직접적인 영향력을 미치는 작업 흐름에서 LLM(대규모 언어 모델) 기반 에이전트의 사용이 증가하고 있음을 강조하고 있습니다. 제안된 액션이 시행되기 전에 조정되어야 하는 실행 경계 문제를 다루며, OCL(조직 제어 계층)을 통해 이를 해결할 수 있는 방법을 제안합니다. OCL은 정책을 집행하고 비상사태를 관리하는 모델에 구애받지 않는 거버넌스 구조를 제공하여, 실행 전 생성된 액션을 중단하거나 수정할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: OCL은 에이전트가 생성한 제안과 플랫폼에서 시행하는 작업 사이에 간단한 인터페이스를 제공하는 방법론입니다. 이 계층은 에이전트의 제안이 실제로 플랫폼에서 수행되는 방식과 분리하여, 제안이 안전한 대안으로 수정되거나 상위 검토를 위해 경로를 지정할 수 있도록 합니다. 연구에서는 OCL이 경제적 거래를 기반으로 한 협상이 신뢰성을 개선하면서도 경제적 성과를 유지할 수 있는지 여부를 평가하고 있습니다.

- **Performance Highlights**: OCL은 다양한 LLM 백엔드를 통해 실험되었으며, 안전하지 않은 실행을 88%에서 거의 0%로 줄이고 유효한 성공률을 12%에서 96%로 증가시켰습니다. 결과는 엄격한 거버넌스가 정책 및 제약 위반에 대한 준수성과 신뢰성을 향상시키지만, 제한된 시장에서는 유연성을 줄일 수 있다는 안전과 효용 간의 트레이드오프를 보여줍니다. 이러한 발견은 LLM 에이전트 시스템이 언어 생성과 실행 가능한 액션 간의 경계에서 명시적인 거버넌스가 필요하다는 것을 시사합니다.



### Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions (https://arxiv.org/abs/2606.04197)
Comments:
          Submitted to the Journal of Artificial Societies and Social Simulation (JASSS)

- **What's New**: 이 연구는 LLM(대형 언어 모델) 에이전트가 메모리 깊이와 네트워크 구조가 합의 형성에 미치는 상호작용을 파악하는 데 중점을 두고 있습니다. 연구 결과에 따르면, 메모리가 길어질수록 분산 네트워크에서는 안정 상태에 도달하는 시간이 느려지지만 중앙 집중적인 네트워크에서는 빨라진다는 것이 밝혀졌습니다. 이는 서로 다른 네트워크 구조에서 메모리의 영향이 다르게 나타난다는 점을 강조합니다.

- **Technical Details**: 이 모델은 고정된 소셜 네트워크에서 LLM 에이전트 간의 'Naming Game'을 시뮬레이션하여 분석되었습니다. 에이전트는 마지막 M∈{2,5,10} 상호작용을 기억하며, 수렴 균형을 선택하는 과정에서 로컬 상호작용을 통해 집단적으로 하나의 관습으로 수렴할 수 있는지를 연구합니다. 각 라운드에서 두 에이전트가 선택한 규칙이 일치할 경우 성공적으로 상호작용하여 점수를 얻습니다.

- **Performance Highlights**: 연구의 결과, 중앙 집중형 네트워크에서의 빠른 정착 속도는 조각난 합의에 고착되기 쉽다는 것을 나타냅니다. 메모리를 통한 빠른 조정과 협력은 분산 네트워크에서의 일관된 규약을 저해할 수 있으며, 이는 LLM 에이전트 populations의 행동을 이해하는 데 기여합니다. 최종적으로, 메모리 깊이와 통신 구조는 함께 설계되어야 하며, 단독으로 최적화하는 것이 아니라 두 요소 간의 관계를 고려해야 한다는 실용적인 시사점을 제공합니다.



### Streaming Communication in Multi-Agent Reasoning (https://arxiv.org/abs/2606.05158)
Comments:
          project page: this https URL

- **What's New**: 이번 논문에서는 Multi-agent reasoning systems의 효율성을 크게 향상시키는 StreamMA라는 새로운 시스템을 소개합니다. StreamMA는 reasoning 단계가 생성되자마자 하류 에이전트에게 전송되는 구조를 채택하여 latency를 줄이고, 동시에 각 단계의 신뢰성을 최대한 활용합니다. 이러한 방식은 하류 에이전트가 더 신뢰할 수 있는 초기 단계에 초점을 맞출 수 있게 하여 전체적인 오류를 감소시킵니다.

- **Technical Details**: StreamMA는 전통적인 'generate-then-transfer' 패러다임 대신, 단계별로 reasoning을 스트리밍합니다. 이 시스템은 pipeline을 통해 인접한 에이전트를 연결하여 latency를 줄이는 데 기여하며, 이는 수학, 과학, 코드 등 다양한 reasoning 벤치마크에서 효과적으로 입증되었습니다. 본 연구는 stream, serial, single 프로토콜의 장점을 정량적으로 분석하며, 효과성 순위, 속도 증가 최대 한계 및 비용 비율을 도출합니다.

- **Performance Highlights**: StreamMA는 Claude Opus 4.6 및 GPT-5.4와 같은 두 가지 선도적인 LLM을 사용하여 총 여덟 개의 reasoning 벤치마크에서 뛰어난 성능을 기록했습니다. 평균 7.3 포인트 향상, 최대 22.4 포인트 향상을 달성하여 기존 기법을 능가했습니다. 또한 에이전트당 단계 수를 늘리면 효과성과 효율성이 일관되게 개선된다는 새로운 'step-level scaling law'도 발견하였습니다.



### Provably Auditable and Safe LLM Agents from Human-Authored Ontologies (https://arxiv.org/abs/2606.04903)
- **What's New**: 이번 논문에서는 선형 감사 가능성(linear auditability)이 필요한 비트리비얼 문제 도메인을 위해 LLM 에이전트 아키텍처인 Agentic Redux를 소개합니다. 우리는 적절한 도메인에서 실행할 때 Agentic Redux의 실행이 항상 의미론적으로 올바르다는 것을 증명합니다. 또한 의료 청구 규정 준수 및 보안 취약점 공개와 같은 두 가지 생산 준비 도메인을 제시합니다.

- **Technical Details**: Agentic Redux는 타입 이론(type theory) 기반의 설계로, 각 에이전트는 자신의 로컬 상태만 보고하며 글로벌 상태에 변경을 제안합니다. 메타 에이전트는 이러한 제안을 평가하여 글로벌 상태를 변경하는 유일한 주체로서, 모든 결정을 기록하는 감사 로그(audit log)를 유지합니다. 이는 시스템의 불변성을 보장하고, 잘못된 동작을 방지하는 데 기여합니다.

- **Performance Highlights**: 논문에서 제안한 에이전트 아키텍처는 규제 요구사항을 항상 준수할 수 있도록 최적화되어 있습니다. 모든 제안 및 결정은 감사 가능성이 확보된 로그에 기록되며, 이는 향후 외부 감사자가 시스템의 결정을 검토할 수 있게 합니다. Agentic Redux는 다양한 문제 도메인에 적용 가능성을 보여주며, 특히 금융 및 의료 문제 해결을 위한 효과적 방법을 제시하고 있습니다.



### R-APS: Compositional Reasoning and In-Context Meta-Learning for Constrained Design via Reflective Adversarial Pareto Search (https://arxiv.org/abs/2606.04823)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 개방형 작업에서 유창함을 보이지만, 계획하고 도구를 사용하며 장기적으로 행동해야 하는 에이전트 설정에서는 신뢰성 있는 결과를 제공하지 못하는 문제를 다루고 있습니다. 저자들은 이러한 문제를 해결하기 위한 새로운 방법인 Reflective Adversarial Pareto Search (R-APS)를 소개하며, 이는 세 가지 주요 실패를 동시에 해결하는 최초의 방법이라고 주장합니다.

- **Technical Details**: R-APS는 각각의 추론 모드에 고유한 맥락(context)을 할당하고, 세 가지 시간 척도(timescales)에서의 상호작용을 조정하는 방식으로 작동합니다. 이 방법은 계획적 조합 추론(staged compositional reasoning), 민감도 기반의 반사실적 스트레스 테스트(sensitivity-guided counterfactual stress-testing), 메타 귀납적 규칙 추출(meta-inductive rule extraction)을 통해 고유의 문맥을 개발합니다. 이는 세 개의 구조적 실패를 해결하며, 추가적인 파인 튜닝 없이 고정된 LLM에서 작동합니다.

- **Performance Highlights**: R-APS는 로봇공학, 보철학, 기계 설계 분야에서 평면 기계 합성(planar mechanism synthesis) 문제를 평가했으며, 32개의 목표 경로(target trajectories)에 대해 uniform-perturbation 기준보다 3.5배 더 긴밀한 견고성 인증을 제공했습니다. 또한, 첫 번째 수용(iteration-to-first-admission)까지 46% 더 빠른 속도를 기록하고, Enum+GA에 비해 2.1배 더 낮은 Chamfer 거리(chamfer-distance)를 달성했습니다. 이 연구는 작은 4B 사고 전문화 모델이 70B 범용 모델과 경쟁력을 가질 수 있음을 보여줍니다.



### RAMPART: Registry-based Agentic Memory with Priority-Aware Runtime Transformation (https://arxiv.org/abs/2606.04628)
- **What's New**: RAMPART는 LLM 기반 에이전트를 위한 컴파일 타임 메모리 모델과 순수 인-램( in-RAM) 블록 레지스트리입니다. 이 시스템은 프로그래머블 런타임 작업을 통해 명시적인 정책에 따라 내용을 조정하여 메모리를 최적화합니다. 다섯 가지 조합 가능 원시(primitives)인 promote, gate, write, evict, rollback은 프롬프트 토큰 비용 없이 블록에 작용하여 성능 향상을 추구합니다.

- **Technical Details**: RAMPART의 핵심 단위인 Instruction Block(IB)은 Behavior directive, Tool schema, 학습된 휴리스틱을 포함한 자연어 문자열로 구성됩니다. 각 블록은 고유한 식별자와 함께 출처 및 저자를 기록하며, 제어 가능성을 통해 블록의 삭제 여부를 결정합니다. 블록 레지스트리는 일반적으로 사용되는 데이터베이스로부터 독립적이며, 디스크 I/O가 발생하지 않아 시스템 성능을 크게 향상시킵니다.

- **Performance Highlights**: RAMPART의 제어된 프로브는 컴파일 타임 위치 조정이 작업 성공에 영향을 미친다는 것을 보여줍니다. 또한, relevance gating을 통해 프롬프트 비용을 67.8% 줄이고 성공률을 83% 회복하는 결과를 나타내었습니다. 크로스 모델 복제 실험에서 RAMPART의 성능은 다양한 모델에서 일관성을 보여주었으며, 블록 그룹화는 Mistral의 평균 통과율을 약 5배 증가시켰습니다.



### AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning (https://arxiv.org/abs/2606.04484)
Comments:
          Technical report, 27 pages

- **What's New**: 본 논문에서는 대형 언어 모델(LLM) 에이전트 강화 학습을 위한 분산형 스웜 훈련 프레임워크인 AgentJet을 소개합니다. 기존의 중앙 집중식(framework)과는 달리, AgentJet은 에이전트 롤아웃과 모델 최적화를 분리한 다중 노드 아키텍처를 채택하여 훈련 가능 모델을 호스팅하는 스웜 서버 노드와 임의의 장치에서 임의의 에이전트를 실행하는 스웜 클라이언트 노드로 구성됩니다.

- **Technical Details**: 이 프레임워크는 이질적인 다중 모델 강화 학습을 지원하여 여러 LLM을 두뇌로 사용하는 이질적인 다중 에이전트 팀의 훈련을 가능하게 합니다. 그리고 격리된 에이전트 런타임을 통한 다중 작업(cocktail training) 수행, 외부 환경 실패로부터 훈련 과정을 방해받지 않도록 하는 내결함(fault-tolerant) 실행, 훈련 중 에이전트를 수정할 수 있도록 하는 라이브 코드(iteration)가 가능합니다. 또한, AgentJet은 시간선 병합(context tracking module) 기능을 통해 다중 모델, 다중 턴, 다중 에이전트 환경에서 효율적인 RL을 지원합니다.

- **Performance Highlights**: AgentJet의 주요 성능 특징으로는 1.5-10배의 훈련 속도 향상(training speedup)을 가능하게 하는 중복된 컨텍스트를 통합하는 것입니다. 추가적으로, AgentJet은 연구 주제를 입력으로 받아 대규모 클러스터에서 자동으로 장기간(multi-day) RL 연구를 수행할 수 있는 자동화된 연구 시스템을 도입하여, 연구자가 개입하지 않고도 RL 연구의 중요한 탐색(workflows)을 재현할 수 있습니다.



### When Freshness Is Not Enough: Distribution-Aware Age of Information for Networked LQR Contro (https://arxiv.org/abs/2606.04361)
- **What's New**: 이 논문에서는 정보의 신선함(freshness)을 측정하는 표준으로 떠오른 Age of Information (AoI)가 네트워크 제어 시스템에서 최적의 기준이 될 수 있는지를 검토합니다. 특히 평균 AoI(Mean AoI) 또는 최고 AoI(Peak AoI)가 실제 성능을 대체하는 방법이 증명되지 않았음을 강조하며, 두 가지 다른 스케줄링 정책이 같은 평균 AoI를 가질지라도 매우 다른 간격 분포를 가질 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 상태 독립적 스케줄링 정책을 사용하고, 무한 수명 LQR(Linear Quadratic Regulator) 추적 문제를 간격 분포에 대한 최적화로 환원하는 방법을 제시합니다. 특히 불안정한 다이나믹스와 지연 피드백의 영향을 받는 상황에서, 이 문제는 평균 값을 넘어서서 고차 통계 모멘트(higher-order moments) 및 지수 모멘트(exponential moments)에 의존한다는 것을 보여줍니다.

- **Performance Highlights**: 실제 차량 궤적 데이터를 사용하여 이론을 검증하였으며, 평균 AoI 단독으로는 부족하다는 것을 입증했습니다. 이론적 비용은 관찰된 성능 경향을 잘 포착함을 보여주어, 기존의 AoI 최소화 접근 접근 방식에서의 한계를 시사합니다.



### What Makes Majority Illusion Easy to Detect? (https://arxiv.org/abs/2606.04260)
- **What's New**: 이 논문은 사회 네트워크 내에서 다수의 환상을 탐지하는 문제를 다룹니다. 다수의 환상이란 사회에서 소수 의견이 지배적인 것으로 잘못 인식되는 현상입니다. 저자들은 q-Majority Illusion 문제를 연구하며, 어떤 사회 네트워크가 다수의 환상을 허용하는지 탐색합니다. 이를 통해 다양한 사회 네트워크의 구조적 특성이 이 문제의 해결 가능성에 미치는 영향을 분석합니다.

- **Technical Details**: 연구자들은 주로 그래프 이론을 통해 문제를 정형화하고, 다수의 환상을 탐지하기 위한 새로운 매개변수를 소개합니다. 특히, vertex integrity와 같은 구조적 매개변수를 사용하여 문제를 효율적으로 해결할 수 있는 알고리즘을 제안합니다. 이 알고리즘은 기존 기술보다 더 나은 성능을 보유하고 있으며, vertex cover number라는 기존 매개변수에 대해서도 개선된 성능을 보여줍니다. 또한, 다양한 그래프 클래스에 따라 계산 부담을 평가하고 그에 따른 알고리즘 복잡성을 분석합니다.

- **Performance Highlights**: 이 알고리즘을 통해 저자들은 다양한 사회 네트워크의 구조에 따라 다수의 환상을 탐지하는 데 있어 계산 효율성을 크게 향상시킬 수 있음을 입증했습니다. 특히, 경량 그래프에서 vertex distance 매개변수 사용에 대한 강력한 계산 불가능성을 확인하였습니다. 이는 다수의 환상을 탐지하는 것이 특정 조건 하에서 어떻게 더 용이할 수 있는지를 보여주며, 대규모 사회 네트워크에서도 효율적인 방법론을 제시합니다.



### Token Budgets: An Empirical Catalog of 63 LLM-Agent Budget-Overrun Incidents, with an Affine-Typed Rust Mitigation as a Case Study (https://arxiv.org/abs/2606.04056)
Comments:
          26 pages. Artifact (catalog CSV, Rust crate, formal proofs): this https URL

- **What's New**: 이 논문은 LLM-agent의 예산 초과 문제를 다룬 첫 번째 연구로, 2023년부터 2026년까지 21개 오케스트레이션 프레임워크에서 발생한 63건의 생산 사고 사례를 문서화하고 있습니다. 저자들은 이를 통해 예산 초과의 재발성을 입증하며, 각 사건에 대해 GitHub 이슈와 관련된 금전적 손실 정보를 제공하고 있습니다. 또한, 이 연구는 예산 관리 문제의 다양한 측면을 조명하고, 예방을 위한 구조적 부족 사항도 정리하였습니다.

- **Technical Details**: 이 논문에서 제안된 방법은 Rust에서 실행 가능한 affine ownership를 통해 예산 제어를 실현합니다. 저자들은 토큰 예산(token budgets)이라는 1,180줄의 Rust 크레이트를 개발하여 클론, 이중 지출(double-spending), 위임 후 예산 사용을 컴파일 오류로 처리합니다. 예산 초과 문제를 방지하기 위해, 작성 시의 무결성을 보장하는 방법을 사용하여 operator의 실수를 정적으로 방지하고 있습니다.

- **Performance Highlights**: 테스트 결과는 이 방법이 5개의 런타임, 3개의 제공자 및 다양한 조건에서 동작하면서 예산 초과가 전혀 없음을 보고하였습니다. 값비싼 런타임 비용이 요구되지 않는 한, 여러 작업을 처리할 때 시간 지연 없이 작동하는 것을 확인했습니다. 이 방법은 모든 구현에서 컴파일 타임 무결성이 보장되며, 운영 중에 오류가 발생하더라도 이전의 오류를 방지하는 데 의미 있는 방안을 제시합니다.



### Assistax: A Multi-Agent Hardware-Accelerated Reinforcement Learning Benchmark for Assistive Robotics (https://arxiv.org/abs/2507.21638)
Comments:
          Accepted at the Reinforcement Learning Conference 2026

- **What's New**: Assistax는 보조 로봇 작업에서 발생하는 복잡한 문제를 해결하기 위해 설계된 오픈 소스 벤치마크입니다. 이 벤치마크는 JAX의 하드웨어 가속을 이용하여 물리 기반 시뮬레이션에서 학습 속도를 크게 증가시켰습니다. Assistax는 로봇과 인간 환자 사이의 상호작용을 다중 에이전트 강화 학습(multi-agent RL)을 통해 모델링하며, 이는 로봇의 제로샷(zero-shot) 조정 능력을 검증하는 데 사용됩니다.

- **Technical Details**: Assistax는 JAX와 MuJoCo MJX를 기반으로 하는 다섯 가지 보조 로봇 작업을 포함하는 구현을 제공합니다. MARL 알고리즘들을 조정하여 로봇이 다양한 선호를 가진 상대와 협력적으로 작업을 수행할 수 있도록 실험합니다. 이 시스템은 Ad-Hoc Teamwork (AHT) 문제를 해결할 수 있는 학습 파이프라인과 미세 조정된 기준선들을 제공합니다.

- **Performance Highlights**: Assistax는 벤치마크 환경 중 가장 빠른 성능을 자랑하며, 물리 기반 시뮬레이션의 경우 최대 412배의 속도 향상을 이끌어냅니다. 인기 있는 연속 제어 RL 및 MARL 알고리즘에 대한 광범위한 평가를 통해 신뢰할 수 있는 기준선을 확립하고 있습니다. 이러한 높은 성능은 RL 연구를 보조 로봇 분야로 발전시키는 데 기여할 것입니다.



