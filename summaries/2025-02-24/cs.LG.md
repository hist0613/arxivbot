New uploads on arXiv(cs.CL)

### Privacy Ripple Effects from Adding or Removing Personal Information in Language Model Training (https://arxiv.org/abs/2502.15680)
Comments:
          23 pages, 26 figures

- **What's New**: 이번 연구는 개인 식별 정보(PII)의 메모리화가 기계 학습 모델 학습 과정에서 어떻게 변화하는지를 다룹니다. 연구팀은 헬스케어와 같은 특수 응용 프로그램에 사용할 경우 PII가 필요할 수 있기 때문에 LLM (Large Language Models) 훈련에서 PII를 완전히 제거하는 것이 어렵다는 점을 강조합니다. 또한, PII의 추가와 제거가 모델의 메모리화에 미치는 영향을 분석하였습니다.

- **Technical Details**: 연구는 세 가지 주요 현상을 기재하였으며, 첫째, 'assisted memorization'이라는 개념을 통해, 유사한 PII가 나중에 훈련될 경우 이전에 본 시퀀스가 메모리화될 수 있음을 발견했습니다. 둘째, 새로운 PII를 데이터에 추가하는 것이 기존 PII의 메모리화에 극적으로 영향을 미칠 수 있음을 확인했습니다. 셋째, PII를 제거하면 다른 PII의 메모리화가 증가할 수 있다는 점을 발견하였습니다.

- **Performance Highlights**: 실험을 통해 LLM의 PII 메모리화의 영향을 실증적으로 조사하였으며, 개별 사용자의 데이터를 제거하기 위한 방법이 다른 개인의 PII 메모리화 리스크에 부정적인 영향을 미칠 수 있음을 밝혔습니다. 이는 모델 제작자들이 모델 학습 시 이러한 개인 정보 보호 리스크를 고려해야 함을 시사합니다.



### FLEKE: Federated Locate-then-Edit Knowledge Editing (https://arxiv.org/abs/2502.15677)
- **What's New**: 본 연구는 Federated Locate-then-Edit Knowledge Editing (FLEKE)라는 새로운 작업을 제안하여 여러 클라이언트가 개인 정보 보호를 보장하면서도 협력적으로 Knowledge Editing (LEKE)을 수행할 수 있도록 합니다. 기존의 LEKE 방법들은 단일 사용자 설정에 의존하여 다중 클라이언트 환경에서는 비효율적이었습니다. FLEKE는 다수의 클라이언트가 중복된 계산을 줄이고, 서로의 지식을 최적화하면서 독립적으로 업데이트할 수 있는 가능성을 제시합니다.

- **Technical Details**: FLEKE는 두 단계의 프레임워크인 FedEdit를 사용하여 Mediator Knowledge Vectors (MKVs)를 선택하고 재사용하는 방법을 최적화합니다. 첫 번째 단계에서 클라이언트는 로컬에서 LEKE를 적용하여 MKVs를 생성하고 중앙 서버에 업로드합니다. 두 번째 단계에서는 서버에 저장된 MKVs를 코사인 유사성을 기반으로 검색하여 재편집할 수 있도록 하여 중복 계산을 최소화합니다.

- **Performance Highlights**: 실험 결과 FedEdit는 비연합 환경의 최신 방법 성능의 96% 이상을 유지하면서도 FedAvg 기반의 기준보다 약 두 배 우수한 성과를 기록했습니다. 또한, FLEKE 작업에서 MEMIT이 PMET보다 더욱 일관된 성능을 보였습니다. 연구 결과는 ML 모델의 효율적인 지식 업데이트에 대한 중요한 기여를 명확히 보여줍니다.



### Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing (https://arxiv.org/abs/2502.15666)
Comments:
          17 pages, 17 figures

- **What's New**: 대형 언어 모델(LLMs)의 텍스트 생성 사용 증가로 인해, AI 생성 콘텐츠 탐지에 대한 우려가 커지고 있습니다. 그러나 간과된 문제는 AI 도구를 사용해 인간이 작성한 콘텐츠를 미세하게 다듬은 AI-폴리시드 텍스트입니다. 최소한으로 다듬어진 텍스트도 AI 생성으로 분류되어야 하는지에 대한 중요한 질문이 제기됩니다.

- **Technical Details**: 본 연구에서는 AI-폴리시드 텍스트 평가(AI-Polished-Text Evaluation, APT-Eval) 데이터셋을 사용해 11종의 최신 AI 텍스트 탐지기를 체계적으로 평가합니다. 이 데이터셋에는 다양한 AI 개입 수준에서 다듬어진 11.7K 샘플이 포함되어 있습니다. 연구 결과, 탐지기들은 최소한으로 다듬어진 텍스트조차 AI 생성으로 잘못 분류하며, AI 개입 정도를 구별하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: 탐지기들은 또한 기존 및 더 작은 모델에 대해 편향된 결과를 보이며, 이러한 한계는 보다 세분화된 탐지 방법론의 필요성을 강조합니다. 실제로, 잘못된 분류는 표절 혐의와 AI 콘텐츠의 확산에 대한 잘못된 주장을 초래할 수 있습니다.



### Machine-generated text detection prevents language model collaps (https://arxiv.org/abs/2502.15654)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)의 훈련에서 생성된 데이터의 기원(인간 또는 합성)이 불확실할 때 모델 붕괴(model collapse)를 방지하기 위한 새로운 방법론을 제안합니다. 이를 위해 기계 생성 텍스트 탐지기에서 중요 가중치를 사용하여 데이터 분포를 재표본화(resampling)하는 방법을 설계하였습니다. 이는 모델의 성능 저하를 줄이고, 또한 인간 작성 데이터가 충분할 경우 성능을 향상시키는 결과를 보여줍니다.

- **Technical Details**: 본 연구에서는 디코딩 전략(decoding strategy)이 모델 붕괴의 정도에 미치는 영향을 분석하고, 재귀적 훈련(recursive training) 동안 생성된 데이터의 특성과 인간 참조(human references)와의 유사성을 살펴봅니다. 또한, 모델 붕괴를 평가하기 위해 세 가지 관점—작업 성능(task performance), 모델 생성 질(model generation quality), 그리고 인간 텍스트와의 의미적 유사성(semantic similarity)—에서 평가를 진행했습니다. 이러한 평가를 통해, 우리는 디코딩 전략이 모델 붕괴에 미치는 심각한 영향을 강조합니다.

- **Performance Highlights**: 제안된 방법론은 두 개의 LLM 변형(GPT-2와 SmolLM2)에서 검증되어, 인공지능 생성 텍스트 탐지기가 제공하는 확률적 추정치를 바탕으로 훈련 데이터의 재표본화가 성공적으로 모델 붕괴를 방지함을 확인했습니다. 결과적으로, 해당 방법을 통해 훈련 데이터셋에 인간 작성 데이터가 충분히 포함되어 있을 때, 모델 성능이 개선되는 것을 확인할 수 있었습니다. 이러한 실험은 텍스트 생성 작업에서 생성된 데이터의 품질 및 다양성을 보장하는 데 기여합니다.



### Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models (https://arxiv.org/abs/2502.15639)
Comments:
          34 pages

- **What's New**: 이번 연구에서는 다국어 대형 언어 모델(mLLMs)에서 언어 간 정렬(cross-lingual alignment)이 효과적으로 향상될 수 있는 방법에 대해 분석합니다. 데이터 효율적인 방법으로서 모델 개입(model interventions)을 통해 언어 모델의 출력 방향을 조정하는 기법인 '전문가 찾기(finding experts)'를 사용합니다. 연구 결과, 이 개입이 대형 언어 모델의 임베딩 공간(embedding space)을 변형시키고, 결과적으로 언어 간 정렬이 개선됨을 확인했습니다.

- **Technical Details**: 연구에서는 Aya-8B, PolyLM-13B, Bloom-7B와 같은 세 가지 오픈소스 다국어 대형 언어 모델을 선택하여, 전문가 뉴런(expert neurons)을 식별하고 개입하는 방식을 사용합니다. Flores200 데이터셋을 이용하여 특정 목표 언어에 해당하는 전문가 뉴런을 찾아내며, 이에 대한 조작 후 언어 간 임베딩 공간과 변화를 분석합니다. 이 과정에서 제안된 개입이 언어 모델의 난이도(perplexity)에 미치는 영향도 관찰됩니다.

- **Performance Highlights**: 연구 결과, 개입 이후 목표 언어의 생성 확률이 전반적으로 증가하며, 이는 개입의 성공적인 결과로 해석됩니다. 특히, 언어 간 임베딩의 거리가 줄어드는 현상이 나타났으며, 이는 교차 언어 검색(cross-lingual retrieval) 성능에서 최대 2배의 정확도 개선으로 이어졌습니다. 이러한 성과는 기존 모델들의 하위 작업(performance on downstream tasks)에서도 긍정적인 영향을 미치는 것으로 보입니다.



### Extraction multi-étiquettes de relations en utilisant des couches de Transformer (https://arxiv.org/abs/2502.15619)
Comments:
          in French language

- **What's New**: 본 기사에서는 프랑스어 텍스트에서 멀티 레이블 관계 추출을 위한 딥 러닝 아키텍처인 BTransformer18 모델을 소개합니다. 이 접근법은 BERT 계열의 사전 학습된 언어 모델의 맥락 표현 능력과 Transformer 인코더의 강력한 장기 의존성 캡처 기능을 결합합니다. 실험은 TextMine'25 챌린지 데이터셋에서 수행되었으며, CamemBERT-Large를 사용할 때 매크로 F1 점수 0.654를 기록하여 FlauBERT-Large보다 우수한 성능을 달성했습니다.

- **Technical Details**: 우리의 접근 방식은 사전 학습된 언어 모델의 파인 튜닝(logic of fine-tuning)으로 구성되어 있으며, 일반적인 body와 관계 추출에 특화된 head로 나뉩니다. 모델에서 body는 프랑스어에 최적화된 CamemBERT-Large로 구현되며, 이 모델은 Transformer 아키텍처 기반의 숨겨진 계층을 포함하여 주목(attention) 메커니즘을 통해 관계를 추출합니다. 각 입력 텍스트의 토큰에 대한 맥락 임베딩(contextual embeddings)을 생성하는 언어 모델이 사용되며, 이 과정은 L개의 Transformer 인코더 층을 통해 장기 의존성을 포착하는 구조를 형성합니다.

- **Performance Highlights**: 모델의 성능을 평가하기 위해, 우리는 TextMine'25 데이터셋을 사용하여 실험을 수행하였습니다. CamemBERT-Large를 사용할 때 성능 개선이 두드러져, 최고 매크로 F1 점수 0.654에 도달하였습니다. 이는 FlauBERT-Large 기반 모델을 초월하는 성과로, 자동으로 복잡한 관계를 추출하는 데 있어 효과적인 접근법임을 보여줍니다.



### Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing (https://arxiv.org/abs/2502.15618)
Comments:
          ICLR 2025

- **What's New**: 이번 논문에서는 Probe Pruning (PP)이라는 새로운 프레임워크를 소개합니다. 이는 대규모 언어 모델(LLM)에서 온라인으로 동적으로 구조적 프루닝을 수행할 수 있으며, 각 배치의 중요한 가중치를 효율적으로 식별할 수 있는 전략을 제공합니다. PP는 1.5%의 FLOPs만 사용하여 LLM의 구조적 프루닝 효율성을 크게 개선할 수 있음을 보여줍니다.

- **Technical Details**: PP는 세 가지 주요 단계로 구성되며, 각 단계는 프로빙(probing), 기록 기반 프루닝(history-informed pruning), 전체 추론(full inference)입니다. 프로빙 단계에서는 잔여 중요성(residual importance)에 따라 숨겨진 상태의 소규모 세트를 선택하고, 이후 기록 기반 프루닝 단계에서는 선정된 프로빙 상태를 과거 상태와 통합합니다. 최종적으로 통합된 정보를 바탕으로 중요도 점수를 사용하여 가중치를 구조적으로 프루닝합니다.

- **Performance Highlights**: LLaMA-2-7B 모델에 대한 평가 결과, PP는 기존에 비해 40% 프루닝 비율에서 성능 저하율을 2.56배 낮추는 성과를 내었습니다. 실험을 통해 PP가 최소한의 프로빙을 통해도 모델의 성능을 유지할 수 있으며, 다양한 입력 배치에서 동적 아울라이어를 효과적으로 처리할 수 있음을 확인하였습니다. 이는 실제 응용에서 리소스를 절약하면서도 높은 성능을 유지하는 데 기여할 것입니다.



### Pastiche Novel Generation Creating: Fan Fiction You Love in Your Favorite Author's Sty (https://arxiv.org/abs/2502.15616)
- **What's New**: 이 논문에서는 Pastiche Novel Generation이라는 새로운 작업을 소개하고 있습니다. 이 작업은 생성된 소설이 원작의 독특한 특징을 모방하도록 요구하며, 캐릭터 프로필 이해, 가능한 줄거리 전개 예측, 생동감 있는 언어로 구체적인 세부 사항 작성이 포함됩니다. 이를 위해 WriterAgent라는 소설 생성 시스템을 제안하여, 문학적 패스티시의 핵심 요소를 마스터하도록 설계되었습니다.

- **Technical Details**: WriterAgent는 커리큘럼 학습 패러다임을 통해 훈련되며, 낮은 수준의 스타일 마스터리에서 높은 수준의 서사적 일관성으로 발전합니다. 주요 작업으로는 언어 스타일 학습, 캐릭터 모델링, 줄거리 계획 및 스타일리시한 글쓰기가 포함되어, 포괄적인 서사적 통제를 보장합니다. WriterAgent는 WriterLoRA 프레임워크를 활용하며, 이는 계층적이고 누적적인 작업 특화 모듈로 구성되어 있습니다.

- **Performance Highlights**: 다언어 고전인 해리포터(Harry Potter)와 홍루몽(Dream of the Red Chamber)에서 WriterAgent의 평가 결과를 보여주었습니다. 이 시스템은 목표 작가의 설정, 캐릭터 역학, 글쓰기 스타일을 포착하는 데 있어서 기존 모델보다 우수성을 입증하며, 일관되고 충실한 서사를 생성하는 데 성공하였습니다.



### LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models (https://arxiv.org/abs/2502.15612)
Comments:
          8 pages, 10 figures in the main paper

- **What's New**: 이 논문에서는 Mamba-1 및 Mamba-2를 위한 새로운 토큰 수준 분해 방법인 LaTIM을 소개합니다. 이 방법은 SSM(상태 공간 모델)계산을 재구성하여 토큰별 분석을 가능하게 하며, 이는 Mamba 아키텍처에Attention 기반 해석 가능성 기법을 적응시킬 수 있게 합니다. LaTIM 방식은 Mamba의 토큰 대 토큰 상호작용 패턴을 밝혀내는 데 유용하며, 투명한 모델을 위한 토대가 됩니다.

- **Technical Details**: SSM은 시퀀스 모델링을 위한 효율적인 접근 방식으로, 최근 Mamba 아키텍처가 언어 모델링 및 기타 모달리티에서 우수한 성능을 입증하였습니다. Mamba는 Attention 메커니즘을 사용하지 않으면서도 구조적인 재귀 메커니즘을 활용해 효율적인 시퀀스 처리를 구현합니다. LaTIM은 이러한 Mamba 아키텍처의 해석 가능성을 높이기 위해, 토큰별 기여도를 명확히 분해합니다.

- **Performance Highlights**: Mamba의 해석 가능성을 향상시키는데 기여하는 LaTIM의 효과는 다양한 작업(기계 번역, 복사, 검색 기반 생성 등)에 걸쳐 평가되었습니다. 특히, 잘 정의된 대각선 Attention 패턴이 특징인 복사 작업에서도 우수한 성능이 나타났습니다. 또한, 소스와 타깃 간의 정밀한 정렬이 필요한 기계 번역 작업에서도 LaTIM의 유효성이 입증되었습니다.



### On the Robustness of Transformers against Context Hijacking for Linear Classification (https://arxiv.org/abs/2502.15609)
- **What's New**: 이번 논문에서는 Transformer 기반의 Large Language Models (LLMs)의 context hijacking 현상을 다루고 있습니다. 이 현상은 사실상 올바른 정보가 포함된 context가 LLM의 예측을 방해하는 문제로, 모델의 강인성에 중대한 이슈를 제기합니다. 연구자들은 최근의 Linear Transformers 발전을 바탕으로 이 현상을 이론적으로 분석하였으며, 이러한 분석은 Transformer 아키텍처에 대한 깊은 이해를 제공합니다.

- **Technical Details**: 이 논문은 context hijacking에 대한 robustness (강인성)을 분석하기 위해 다층 linear transformer 모델을 활용했습니다. 연구자들은 hijacking 예제를 반대 라벨을 가진 query-answer 쌍으로 설정하고 multi-step gradient descent를 이용하여 최적의 학습률과 초기화를 관찰했습니다. 그 결과, 깊은 Transformer는 더 섬세한 최적화 단계를 수행할 수 있어 hijacking 예제의 영향을 줄일 수 있다는 것을 입증했습니다.

- **Performance Highlights**: 실험 결과, 더 깊은 Transformer 구조가 더 높은 강인성을 보임을 확인하였습니다. 이는 더 많은 prepended context 예제가 사용될 때 모델의 예측이 바뀌기 어려워진다는 것을 의미합니다. 본 연구는 Linear Problems에 대한 Gradient Descent 방법을 사용하는 타 분야에서도 독립적인 관심을 가질 수 있는 결과를 제시합니다.



### Do Multilingual LLMs Think In English? (https://arxiv.org/abs/2502.15603)
Comments:
          Main paper 9 pages; including appendix 48 pages

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 다국어 작업을 수행하더라도 가장 중요한 결정을 영어에 가까운 표현 공간에서 내린다는 사실을 보여줍니다. 연구진은 프랑스어, 독일어, 네덜란드어, 중국어 문장에 대한 내부 표현을 탐구하며, LLM이 의미적으로 중요한 단어에 대해서 먼저 영어에 가까운 표현을 생성한 후 그것을 목표 언어로 번역하는 과정을 따릅니다. 이 결과는 다국어 LLM이 영어에 의해 크게 형성된 방식으로 이유를 공개하지 않고 작동함을 시사합니다.

- **Technical Details**: 이 연구에서는 4개의 오픈 소스 모델(Llama-3.1-70B, Gemma-2-27b, Aya-23-35B, Mixtral-8x22B)의 성능을 비교 분석하며, 이 모델들이 내부 표현 공간에서 어떻게 작동하는지를 상세히 설명합니다. 로그잇 렌즈(logit lens)와 인과 추적(causal tracing), 그리고 스티어링 벡터(steering vectors)의 세 가지 해석 방법으로 모델의 내부 표현을 조사합니다. 특히, 영어에서 생성된 스티어링 벡터를 사용했을 때 비영어 문장의 결과를 더 효과적으로 조정할 수 있다는 점이 중요합니다.

- **Performance Highlights**: 이 연구는 LLM들이 영어 표현 공간에서 결정을 내리기 때문에 비영어 문장 생성에서 성능이 저하된다는 점을 강조합니다. 특히, 모델의 언어 범위에 따라 성능 차이를 분석하면서, 결과적으로 LLM의 영어 중심 행동이 다양한 언어 설정에서의 모델의 공정성, 신뢰성, 그리고 강인성에 영향을 미친다고 설명합니다. 이번 연구는 LLM의 다국어 처리에서의 한계를 이해하는 데 중요한 통찰력을 제공합니다.



### Robust Bias Detection in MLMs and its Application to Human Trait Ratings (https://arxiv.org/abs/2502.15600)
Comments:
          To appear at Findings of NAACL 2025

- **What's New**: 이 논문은 MLMs(Masked Language Models)에서 인구 통계적 특성에 대한 바이어스(Frame Bias)를 평가하기 위한 체계적인 통계적 접근 방식을 제안합니다. 기존의 일률적인 템플릿을 사용하는 연구가 고유의 변동성 및 바이어스 정량화를 무시하는 반면, 본 연구에서는 혼합 모델(mixed models)을 활용하여 랜덤 효과를 고려하고 바이어스를 통계적 효과 크기(statistical effect sizes)로 정량화합니다. 또한, 성별 바이어스를 성격(personality)과 특성(character) 측면에서 탐구하며, 다양한 MLM의 특성을 비교합니다.

- **Technical Details**: 본 논문은 템플릿 기반 접근 방식을 사용하여 MLM의 바이어스를 평가합니다. 템플릿은 인구 통계적 속성 단어와 대상 단어를 포함하는 문장 구조로, 성별 및 인간의 특성을 분석합니다. 각 템플릿에서 파생된 연결 문장들의 의존성을 평가하기 위해, 우리는 랜덤 효과를 고려하는 혼합 효과 모델을 채택하고, 문장 바이어스의 추정치에 대해 공통성을 바탕으로 가중치를 부여합니다. 효과 크기를 사용하는 것의 중요성을 강조하여, 관찰된 바이어스의 진정한 차이를 판별할 수 있도록 합니다.

- **Performance Highlights**: ALBERT는 이진 성별에 대해 바이어스가 없으나 비이진적인 경우에는 가장 높은 바이어스를 보이는 반면, RoBERTa-large은 이진 성별에서는 가장 높은 바이어스를 나타내지만 비이진한 경우에는 거의 바이어스가 없는 것으로 나타났습니다. 또한, 심리학 연구 결과와 배치되는 MLM의 바이어스 경향이 관찰되며, 성격의 여러 차원에 대해 남성과 여성 간의 차이는 미세하다는 결과가 도출되었습니다. 최종적으로, 본 연구는 MLM이 성별 바이어스를 적절하게 식별할 수 있는지를 평가하며, 자동화된 결정에서의 바이어스 감소의 필요성을 강조합니다.



### SafeInt: Shielding Large Language Models from Jailbreak Attacks via Safety-Aware Representation Intervention (https://arxiv.org/abs/2502.15594)
- **What's New**: 이번 연구는 Jailbreak 공격에 대한 방어책으로, 안전성을 고려한 representation intervention 기법인 SafeInt를 제안합니다. 이는 기존의 방어 기법들이 효과성과 효율성을 동시에 충족하지 못하는 문제를 해결하고자 하며, LLM의 representation을 동적으로 조정하여 해로운 쿼리로부터 보호합니다. SafeInt는 Jailbreak 샘플의 representation 분포를 조정하여 이를 안전하지 않은 샘플과 맞추는 새로운 방법론을 제시합니다.

- **Technical Details**: SafeInt는 LLM의 중간 층에서 매개변수화된 개입을 통해 조정된 representation을 사용하고, 이후 층에서 분류 확률과 대조 학습을 포함하여 representation 정렬을 수행합니다. 이는 Jailbreak과 무관한 representation에 대해서는 안정성을 유지하기 위해 그 재구성을 수행합니다. 연구는 Qwen-7B-Chat, Llama-2-7B-Chat, Llama-3-8B-Instruct, Vicuna-7B-v1.5와 같은 네 가지 LLM에서 이루어졌습니다.

- **Performance Highlights**: 연구 결과, SafeInt는 여섯 가지 Jailbreak 공격에 대한 방어 성능에서 모든 기존 기법을 초월하며, 유용성(utility)도 상당 부분 유지하고 있음을 보여주었습니다. 또한, Adaptive 공격에 대한 효과성도 검증하여 실시간 공격에도 잘 대응할 수 있다는 결과를 도출하였습니다. 전반적으로 SafeInt는 Jailbreak 공격에 대한 방어의 새로운 기준을 제시합니다.



### Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning (https://arxiv.org/abs/2502.15592)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 긴 문맥(long-context) 모델링을 다루며, 특히 지시문 조정(instruction tuning)과 같은 중요한 요소를 분석합니다. 또한, 짧은 문맥에 대해 지시문을 조정한 모델이 긴 문맥에 성공적으로 일반화할 수 있음을 발견했습니다. 이를 바탕으로 고품질 지시-답변 쌍을 위한 확장된 배경 문맥을 생성하는 "context synthesis"라는 새로운 데이터 합성 프레임워크를 제안합니다.

- **Technical Details**: 기존의 모델들은 긴 문맥을 다루기 위해 포지션(position) 및 주의(attention) 문제를 해결하는 데 초점을 맞추었습니다. 그러나 지시문 조정의 중요성은 간과되었으며, 논문은 이러한 문제를 해결하기 위해 저자 특수 실험을 통해 얻은 세 가지 주요 발견(지시문 품질, 문맥 조합, 문맥 길이)을 기반으로 합니다. 최종적으로 이 논문에서는 LLaMA2-7B-64K과 LLaMA3.1-8B-128K 모델을 활용하여 실험을 진행합니다.

- **Performance Highlights**: 실험 결과, 제안된 context synthesis 접근 방식이 이전의 지시문 합성 방법보다 유의미하게 우수한 성능을 보였으며, 인간이 주석한 긴 문맥 지시 데이터와 비슷한 성능을 달성했습니다. 지시문 조정이 포함된 문맥을 이용한 모델들은 새로운 문서 수준의 벤치마크 작업에서도 뛰어난 일반화를 보였습니다. 이를 통해 지시문 품질과 컨텍스트의 중요성이 강조되었습니다.



### LightThinker: Thinking Step-by-Step Compression (https://arxiv.org/abs/2502.15589)
- **What's New**: 이번 논문에서는 LightThinker라는 새로운 방법을 제안합니다. 이 방법은 대형 언어 모델(LLMs)이 추론 중 중간 사고를 동적으로 압축할 수 있게 해줍니다. 사람의 인지 과정에서 영감을 받아, 경량화된 표현으로 불필요한 추론 단계를 제거함으로써 메모리 사용량과 연산 비용을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: LightThinker는 LLM이 언제 어떻게 압축을 수행할지 학습하도록 훈련시킵니다. 데이터 구성, 숨겨진 상태(hidden states) 매핑, 전문적인 주의 마스크(attention masks)를 사용하여 압축 로직을 조정하고, Dependency (Dep) 메트릭을 도입하여 압축 정도를 정량화합니다. 이 메트릭은 생성된 각 토큰이 얼마나 많은 역사적 토큰에 의존하는지를 측정합니다.

- **Performance Highlights**: 실험 결과, LightThinker는 Qwen 모델에서 피크 토큰 사용량을 70% 줄이고, 추론 시간을 26% 감소시키며, 정확도를 크게 저하시키지 않고(1% 감소) 효율성을 높였습니다. 이는 복잡한 추론 작업에서 LLM의 효율성을 향상시킬 수 있는 새로운 방향을 제시합니다.



### Chats-Grid: An Iterative Retrieval Q&A Optimization Scheme Leveraging Large Model and Retrieval Enhancement Generation in smart grid (https://arxiv.org/abs/2502.15583)
Comments:
          12 pages, 10 figures

- **What's New**: 본 논문은 스마트 그리드 환경에 최적화된 반복 검색 기반 질문-답변(Q&A) 프레임워크인 Chats-Grid를 제안합니다. 기존의 검색 증강 생성(RAG) Q&A 시스템들이 겪고 있는 다양한 문제, 특히 검색 품질과 관련성 문제를 해결하는 데 중점을 두고 있습니다. Chats-Grid는 여러 단계로 구성된 최적화를 통해 실시간 데이터 처리와 관련된 문제를 개선합니다.

- **Technical Details**: Chats-Grid는 쿼리 확장을 통해 다양한 데이터 소스, 즉 센서 판독값, 계량기 기록 및 제어 시스템 매개변수 등을 포괄적으로 다룹니다. 검색 과정에서는 BM25 희소 검색과 BAAI 일반 임베딩(BGE) 밀집 검색 방법을 결합하여 대량의 이질적인 데이터셋을 효과적으로 처리합니다. 마지막으로, 고급 언어 모델을 활용해 문서의 관련성을 평가하고, 부적절한 결과를 필터링하며, 맥락의 정확성을 기준으로 문서를 재정렬하여 정확하고 맥락 인식이 가능한 답변을 생성합니다.

- **Performance Highlights**: 실험 결과에 따르면 Chats-Grid는 기존 최첨단 방법 대비 충실도, 맥락 회상, 관련성 및 정확성에서 각각 2.37%, 2.19%, 3.58%의 성능 향상을 보였습니다. 이 프레임워크는 스마트 그리드의 운영 효율성을 높이고 사용자 상호작용을 개선하여 보다 견고하고 적응 가능한 스마트 그리드 인프라를 지원하는 데 기여합니다. 또한, 설명된 방법론은 포괄적인 시뮬레이션과 사례 연구를 통해 그 효과성을 입증하였습니다.



### Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders (https://arxiv.org/abs/2502.15576)
Comments:
          Pre-print. 20 pages, 5 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 내부 상태를 이해하기 위한 새로운 접근 방식을 제안합니다. 기존의 희소 오토인코더(sparse autoencoder, SAE) 기술을 활용하여 LLM의 학습된 특성을 해석하고, 그에 대한 의미를 명확히 하는 방법을 모색합니다. 특히, 기존 설명 방법들의 빈도 편향(frequency bias) 문제를 다루며, 정해진 어휘 집합(fixed vocabulary set)을 사용하여 설명의 정확성을 높이고자 합니다.

- **Technical Details**: 초기에는 LLM의 숨겨진 표현을 이해하기 위해 다양한 기법이 적용되었습니다. 이 연구에서는 희소 오토인코더를 사용하여 LLM의 고유한 의미 특성을 설명할 수 있는 방법을 설명합니다. 비과거 행렬 분해(mathematical decomposition) 기법이 한계가 있는 반면, SAE는 수많은 희소 특성 벡터를 학습하고 이를 통해 복잡한 숨겨진 공간을 재구성하는 데 효과적임을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 설명 방법보다 더 의미 있는 담론 레벨 설명(discourse-level explanations)을 제공하고 있음을 나타냅니다. 또한, 이러한 설명은 LLM의 동작을 효과적으로 조정하여 jailbreak 공격을 방어하는 데 유효합니다. 이는 LLM 행동을 조정하는 데 있어 설명의 가치가 중요하다는 점을 강조합니다.



### A Survey of QUD Models for Discourse Processing (https://arxiv.org/abs/2502.15573)
Comments:
          accepted to the main conference of NAACL2025

- **What's New**: 이 논문은 QUD(Question Under Discussion) 프레임워크의 최근 발전과 다양한 모델들이 담고 있는 내용에 대한 종합적인 조사 결과를 제시하고 있습니다. 특히 QUD는 담화 처리(discourse processing)에서 사용되는 혁신적인 접근법으로, NLP 작업을 질문-답변형으로 전환하는 경향과 맞물려 있습니다. 이러한 연구들은 RST, PDTB 및 SDRT와 같은 기존의 담화 프레임워크와의 관계를 탐구합니다.

- **Technical Details**: 논문에서는 QUD 프레임워크에 따라 세 가지 담화 처리 모델인 QUD-tree 접근법, 기대 기반 접근법, 종속 기반 접근법을 식별하고 이들의 이론적 배경을 논의합니다. QUD-tree 접근법은 QUD의 구조를 모델링하기 위해 트리 구조를 사용하는 이론적 제안에 기반하고 있으며, 기대 기반 접근법은 문맥 내에서 질문을 유도하는 요소를 인식하는 데 초점을 맞춥니다. 이러한 접근법들은 기본적으로 언어 학적 이론에 뿌리를 두고 있어 모델의 제약 조건과 표현의 특성을 이해하는 데 어려움이 있습니다.

- **Performance Highlights**: QUD 프레임워크가 담화 모델링에 적용될 수 있는 가능성은, 대규모 언어 모델(LLMs)의 발전으로 인해 비용 효과적인 옵션으로 여겨진다는 점입니다. 또한, 자유 형식의 질문을 사용한 담화 주석(annotation) 접근법은 종종 전문가가 사전 정의한 추상적이고 모호한 담화 관계와 비교하여 상대적으로 더 간단하게 느껴질 수 있습니다. 이러한 점에서, QUD는 담화 처리 작업에 있어 특히 저자와 독자 간의 상호작용을 복잡하게 만드는 과제를 제시합니다.



### DReSD: Dense Retrieval for Speculative Decoding (https://arxiv.org/abs/2502.15572)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 Speculative Decoding (SD) 방식을 통해 대규모 언어 모델(LLM)의 생성 속도를 향상시키는 방법을 제안합니다. 특히, 비모수적 데이터 저장소에서 다음 몇 개의 토큰을 검색하여 제안하는 retrieval 기반 SD에 주목합니다. 기존의 Sparse Retrieval (REST) 방식의 한계를 해결하기 위해, 새로운 Dense Retrieval for Speculative Decoding (DReSD) 프레임워크를 소개합니다.

- **Technical Details**: DReSD는 문맥화된 토큰 임베딩을 사용하여 근사 최근접 이웃 검색을 통해 토큰 시퀀스를 효율적으로 검색합니다. 이는 SD의 문맥을 더욱 잘 이해하고 의미적으로 관련성이 높은 토큰 시퀀스를 제공하여, 생성 과정의 효율성을 높입니다. 실험 결과, 기존 Sparse Retrieval에 비해 DReSD는 수용률(acceptance rates)이 87% 증가하며, 수용된 토큰의 길이가 65% 늘어나고, 생성 속도는 19% 빨라졌습니다.

- **Performance Highlights**: DReSD는 실험을 통해 Dense Retrieval의 세 가지 핵심 요소를 파악하였고, 최적 구성을 통해 생성 속도를 최대 4.64배까지 가속화할 수 있음을 보여주었습니다. 이는 LLM 통합이 용이한 새롭고 효과적인 SD 프레임워크를 제공하며, 더 나은 성능을 실현하기 위해 중요한 기초 데이터입니다.



### PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning (https://arxiv.org/abs/2502.15543)
Comments:
          20 pages, 7 figures, 7 tables

- **What's New**: 이번 논문에서는 ParametrIc Pruning-based Knowledge-Augmented Generation (PIP-KAG)이라는 새로운 접근법을 제안하고 있습니다. 이 방법은 LLMs(대규모 언어 모델)의 내부 지식을 가지치기(pruning)하여 외부 지식 활용을 개선하는 데 중점을 두고 있습니다. 또한, CoConflictQA 벤치마크를 구축하여 LLM의 질문 응답 시 맥락 충실도를 평가하는 새로운 기준을 마련했습니다.

- **Technical Details**: PIP-KAG는 LLM의 내부 지식을 제거하기 위해 신경 활성화(neuron activation) 기반의 파라미터 가지치기(pruning) 방법을 사용합니다. 이 접근법은 지식 증강 후 비활성화된 파라미터를 식별하여 제거하게 됩니다. 이후, 플러그 앤 플레이(plugin-and-play) 방식의 KAG 적응 모듈을 설치하여 외부 지식의 활용도를 높입니다.

- **Performance Highlights**: 실험 결과, PIP-KAG는 CoConflictQA에서 지식 갈등을 크게 줄이고, 맥락 충실도(context fidelity)를 향상시키는 데 성공했습니다. 특히, PIP-KAG는 LLM의 파라미터를 13% 감소시켜 파라미터 효율성을 높였습니다. 이러한 성과는 KAG 프레임워크 내에서 파라미터 효율적인 LLM 구축에 중요한 통찰을 제공합니다.



### SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instrucion Following Evaluation for Social Agents (https://arxiv.org/abs/2502.15538)
Comments:
          26 pages, 5 figures, 23 tables

- **What's New**: 이 논문은 인간의 사회적 전략을 사회 에이전트에 통합하는 연구의 결핍을 극복하고자 합니다. 저자들은 SOTOPIA-Ω 프레임워크를 제안하여 언어 에이전트의 사회적 능력을 강화하고, 협상 이론에서 영감을 얻은 다단계 추론 전략을 포함시킵니다. 또한, 사회적 지시를 따르는(Social Instruction Following, S-IF) 개념과 이를 평가하기 위한 두 가지 새로운 메트릭을 도입하였습니다.

- **Technical Details**: SOTOPIA-Ω 프레임워크는 전문 에이전트에 다단계 추론 전략과 두 가지 간단한 직접적인 전략을 동적으로 주입합니다. 이를 통해 고품질의 사회적 대화 훈련 데이터셋을 자동으로 구축할 수 있습니다. 제안된 새로운 S-IF 평가 메트릭은 사회적 능력과 보완적으로 작용하여, 에이전트의 성능을 더욱 높일 수 있습니다.

- **Performance Highlights**: 7B 모델들이 고품질 데이터셋을 기반으로 학습하여, GPT-4와 같은 전문가 에이전트를 초월하여 사회적 목표를 달성하는 데 있어 뛰어난 성과를 보였습니다. 실험 결과, 동적 구성의 이점을 확인할 수 있었고, 이는 에이전트가 오랜 시간 동안 침체되는 상황을 극복하는 데 큰 도움이 됨을 보여주었습니다.



### Scale-Distribution Decoupling: Enabling Stable and Effective Training of Large Language Models (https://arxiv.org/abs/2502.15499)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 훈련 안정성을 개선하기 위해 Scale-Distribution Decoupling (SDD)이라는 새로운 접근 방식을 제안합니다. SDD는 완전 연결 계층의 가중치 행렬의 스케일(scale)과 분포(distribution)를 명확히 분리함으로써 훈련을 안정화합니다. 기존의 기술들과는 달리, SDD는 활성화에 대한 정규화(nomalization) 메커니즘을 적용하고 학습 가능한 스케일링 벡터( scaling vector)를 도입하여 잘 조건화된 그래디언트를 유지합니다.

- **Technical Details**: SDD는 가중치 행렬의 스케일(scale)과 분포(distribution)을 분리하여 훈련 안정성을 증진합니다. 이는 활성화 정상화(activation normalization) 및 독립적인 스케일 제어를 통해 이루어집니다. SDD의 수정된 형식은 norm(V⋅x)와 α를 통해 분포 특성을 포착하고 스케일을 독립적으로 조정하여 그래디언트 폭발(gradient explosion) 및 소실(vanishing)을 방지합니다.

- **Performance Highlights**: 실험 결과, SDD는 다양한 LLM 아키텍처에서 훈련 안정성을 개선하며, 특히 불안정한 Post-Norm Transformers에서 성능을 높였습니다. 또한 SDD는 훈련의 수렴(convergence) 안정성 및 효율성을 높여 대규모 사전 학습(pre-training) 작업에 매우 실용적입니다. 전반적으로 SDD는 최소한의 메모리 오버헤드와 경량화를 유지하면서 실질적인 솔루션을 제공합니다.



### ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models (https://arxiv.org/abs/2502.15487)
Comments:
          Submitted to ACL 2025

- **What's New**: 이번 논문에서는 ExpliCa라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 대형 언어 모델(Large Language Models, LLMs)의 명시적 인과적(reasoning causal) 사고를 평가하는 데 도움을 줍니다. ExpliCa는 인과적(causal) 및 시간적(temporal) 관계를 서로 다른 언어적 순서로 통합하고 언어적 연결어(linguistic connectives)를 통해 명시적으로 표현한 점이 특징입니다.

- **Technical Details**: ExpliCa 데이터셋은 사람들의 수집된 허용성 평가 인간 평가 지표로 보강되어 있습니다. 연구팀은 LLM을 ExpliCa 데이터셋으로 평가하면서 prompting 및 perplexity 기반 지표를 사용하였습니다. 연구에서는 상업적 및 오픈 소스 LLM 7개를 테스트하였고, 그 중 어느 모델도 0.80 이상의 정확도에 도달하는 데 어려움을 겪었습니다.

- **Performance Highlights**: 모델은 인과관계(causal relations)와 시간관계(temporal relations)를 혼동하는 경향이 있으며, 이벤트의 언어적 순서에 따라서도 성능이 강하게 영향을 받는 것으로 나타났습니다. 마지막으로, perplexity 기반 점수와 prompting 성능은 모델의 크기(model size)에 따라 다르게 나타났습니다.



### Enhancing RWKV-based Language Models for Long-Sequence Text Generation (https://arxiv.org/abs/2502.15485)
Comments:
          8 pages, 2 tables, 3 figures

- **What's New**: 이 논문에서는 RWKV 기반 언어 생성 모델을 개선하여 긴 시퀀스 텍스트 처리를 향상시키는 방법을 제안합니다. 적응형 토큰 이동(adaptive token shift) 및 게이팅 메커니즘(gating mechanism)을 도입하여 텍스트 생성에서 장기 의존성을 더 잘 캡처할 수 있습니다. 여러 실험을 통해 기본 RWKV 모델과 개선된 모델의 성능을 비교하였고, 명백한 성능 향상이 확인되었습니다.

- **Technical Details**: RWKV 모델은 순환(recurrent) 및 Transformer의 장점을 결합한 하이브리드 아키텍처입니다. 본 논문에서 제안하는 적응형 토큰 이동 메커니즘은 과거의 상태를 동적으로 조정하며, 게이팅 메커니즘은 이 정보를 어떤 비율로 포함할지를 결정합니다. 각 시간 단계에서 shifted hidden state를 계산하고, 해당 상태가 현재 hidden state에 추가되는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과, 개선된 모델은 BLEU 및 ROUGE 점수가 크게 향상되었고, 특히 ROUGE-1 및 ROUGE-L 점수에서 장기 의존성을 더욱 잘 캡처하며 생성 품질이 향상된 것으로 나타났습니다. 이 결과는 모델의 훈련 안정성을 위한 레이어 정규화(layer normalization)와 정보 통합을 위한 적응형 게이트의 중요성을 강조합니다.



### When Compression Meets Model Compression: Memory-Efficient Double Compression for Large Language Models (https://arxiv.org/abs/2502.15443)
- **What's New**: 이 논문은 저메모리 디바이스에서의 LLM(대형 언어 모델) 배포 시 발생하는 메모리 요구사항을 해결하기 위한 새로운 방법론을 제안합니다. 특히, 양자화(Quantization) 이후의 LLM을 추가로 압축하는 프레임워크를 도입하여 약 2.2배의 압축 비율을 달성합니다. 압축 인식 양자화(Compression-aware Quantization)와 가지치기(Pruning) 기법을 통해 모델 파라미터의 압축 가능성을 높이고 메모리 사용량과 대기 시간 간의 균형을 조절하는 접근 방식을 제안합니다.

- **Technical Details**: 압축 인식 양자화(Compression-aware Quantization) 기법은 모델의 가중치 분포를 조정하여 데이터의 압축성을 증가시키는 방법으로, 데이터의 불균일성(uniformity)을 활용합니다. 또한, 양자화 데이터에서 상관관계가 높은 데이터 분포를 분석하여 압축 효율성을 높이는 전략을 제공합니다. 실제 환경에서의 추론(inference)에서 발생할 수 있는 빈번한 복원(decompression) 작업의 오버헤드를 줄이기 위해, 속도 적응형 방법(Speed-adaptive Method)을 구현하여 메모리 아키텍처에 기반한 전체 추론 속도를 분석합니다.

- **Performance Highlights**: 실험 결과, 압축된 LLM은 메모리 사용량이 40% 감소하면서도 정확도와 추론 속도의 손실이 최소화되었습니다. 특히, 압축 모델은 약 1%의 정확도 하락을 보이며, 높은 압축 비율(CR)을 달성하였습니다. 이 방법론은 LLM을 메모리가 제한된 디바이스에서 성공적으로 배포할 수 있는 유망한 솔루션을 제공합니다.



### Mixup Model Merge: Enhancing Model Merging Performance through Randomized Linear Interpolation (https://arxiv.org/abs/2502.15434)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 Mixup 데이터 증강 기법에서 영감을 얻은 Mixup Model Merge (M$^3$)라는 새로운 모델 병합 방법을 제안합니다. 이 방법은 두 개의 대형 언어 모델(LLMs)의 매개변수를 임의로 생성된 선형 보간 비율을 통해 결합하여 더 유연하고 포괄적인 매개변수 공간 탐색을 가능하게 합니다. M$^3$는 기존의 고정된 비율로 매개변수를 병합하는 방식의 한계를 극복하고 모델 병합의 가능성을 극대화합니다.

- **Technical Details**: M$^3$는 매개변수 간의 기여 비율을 동적으로 조정하여 매개변수 병합 과정을 더 유연하게 만듭니다. 이 방법은 두 개의 미세 조정된 LLM 간의 매개변수 융합 비율을 랜덤으로 생성된 비율 λ로 제어하며, 이 λ는 베타(Beta) 분포에서 샘플링됩니다. 이로 인해 M$^3$는 모델 병합의 매개변수 공간을 더 깊이 탐색할 수 있습니다.

- **Performance Highlights**: M$^3$는 세 가지 미세 조정된 LLM에 대해 수행된 실험에서 다양한 작업에서 성능을 크게 향상시켰습니다. 또한, OOD(out-of-distribution) 및 적대적 공격에 대한 견고함을 강화하고, DARE와 같은 희소화(sparsification) 기법과 결합할 때 더 우수한 결과를 보여줍니다. 전반적으로, M$^3$는 추가적인 계산 자원 없이 효율적인 해결책을 제공하며, 통합된 모델의 성능을 크게 개선합니다.



### Pub-Guard-LLM: Detecting Fraudulent Biomedical Articles with Reliable Explanations (https://arxiv.org/abs/2502.15429)
Comments:
          long paper under review

- **What's New**: 이번 연구에서는 생물 의학 저널에서의 사기 탐지를 위해 특별히 설계된 최초의 대규모 언어 모델 기반 시스템인 Pub-Guard-LLM을 제안합니다. 이 시스템은 세 가지 응용 모드(바닐라 추론, 검색 증대 생성, 다중 에이전트 논쟁)를 제공하여 텍스트 기반의 예측 설명 기능을 포함합니다. 또한, 연구자들이 활용할 수 있도록 11,000개 이상의 실제 생물 의학 논문을 포함한 오픈 소스 벤치마크인 PubMed Retraction도 소개합니다.

- **Technical Details**: Pub-Guard-LLM 시스템은 텍스트 예측 시 더 높은 신뢰성을 제공하며, 이를 위해 고유한 세 가지 응용 모드를 사용합니다. 이 시스템은 다양한 기준을 기반으로 하는 평가 방법으로 사기 탐지 성능을 측정하며, 사기 논문을 정확하게 탐지하는 것이 가능합니다. 또한, 시스템의 성능 향상과 설명 가능성을 동시에 강화하여 과학적 무결성을 보호하는 데 기여합니다.

- **Performance Highlights**: Pub-Guard-LLM은 여러 벤치마크와 비교했을 때 일관되게 우수한 성능을 보였으며, 기존 시스템에 비해 더 관련성 높은 설명을 제공합니다. 특히, 세 가지 응용 모드는 예측의 정밀도와 재현율, 설명의 신뢰도를 조절할 수 있어 다양한 사용자의 필요에 맞춰 효과적으로 활용할 수 있습니다. 이러한 다재다능한 기능은 사기 논문 탐지에 널리 적용될 수 있는 가능성을 제시합니다.



### Evaluating Multimodal Generative AI with Korean Educational Standards (https://arxiv.org/abs/2502.15422)
Comments:
          18 pages; To appear at NAACL 2025 Main Conference (Project page: this https URL )

- **What's New**: 이번 논문은 한국 교육 시험을 활용하여 멀티모달 생성 AI 시스템을 평가하기 위한 새로운 벤치마크인 KoNET(Korean National Educational Test Benchmark)을 소개합니다. KoNET는 초중고 및 대학 입학 자격 시험인 KoEGED, KoMGED, KoHGED, KoCSAT의 네 가지 시험으로 구성되어 있으며, 이들 시험은 낮은 자원 언어인 한국어에 대한 AI의 성능을 분석할 수 있는 기회를 제공합니다. KoNET를 통해 다양한 모델의 성능을 비교하고, 특히 인간 오답률 데이터와 함께 AI의 능력을 면밀히 평가할 수 있습니다.

- **Technical Details**: KoNET는 한국의 국가 교육 시험에서 문제를 변환하여 구조화된 멀티모달 VQA 형식으로 구성되어 있습니다. 각 시험은 질문 난이도에 대한 세부 분석을 제공하며, KoCSAT는 응시자의 오답률 데이터를 포함하여 AI 모델의 행동을 인간 성능과 직접 비교할 수 있도록 합니다. 더불어, 다양한 오픈소스 및 클로즈드 소스 AI 모델이 KoNET에서 테스트되며, Chain-of-Thought (CoT) 접근법과 OCR API를 활용하여 이미지 기반 문제를 처리합니다.

- **Performance Highlights**: 실험에서는 18개의 오픈소스 LLMs, 20개의 오픈소스 MLLMs, 4개의 클로즈드 소스 LLMs 및 4개의 클로즈드 소스 MLLMs를 포함하여 다양한 모델을 평가했습니다. KoNET는 정확한 기준을 마련하고, AI 성능을 이해하기 위한 통찰력을 제공함으로써 한국 교육 시장에서 AI 기반 교육 기술의 응용 가능성을 높입니다. 또한, 데이터 세트와 코드는 모두 공개되어 있어서 연구자들이 자유롭게 접근할 수 있도록 합니다.



### Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking (https://arxiv.org/abs/2502.15419)
Comments:
          15 pages, 1 figure, 18 tables

- **What's New**: 이번 논문에서는 스페인어, 독일어, 영어와 같은 저자원 언어를 지원하는 220만 클레임-소스 쌍으로 구성된 최초의 대규모 다국어 사실 확인 데이터세트인 MultiSynFact를 소개합니다. 논문은 Wikipedia의 외부 지식을 활용하여 이 데이터세트를 생성하는 파이프라인을 설명합니다. 또한, MultiSynFact를 다국어 사실 확인 및 데이터세트 생성 연구에 활용할 수 있도록 사용자 친화적인 오픈 소스 프레임워크를 제공하는 점이 특징입니다.

- **Technical Details**: 저자들은 Wikipedia에서 정보 문장을 추출하고, LLMs(대형 언어 모델)를 사용하여 Supports, Refutes, Not-info(결정하기에 충분한 정보 부족)로 분류되는 클레임을 생성하는 3단계 파이프라인을 설계하였습니다. 이 과정에는 생성된 클레임의 언어적 및 의미적 일치를 보장하는 엄격한 검증 단계가 포함되어 있습니다. 최종적으로 우리는 220만 개의 소스-클레임 쌍을 포함하는 MultiSynFact를 생성하였으며, 이는 다양한 언어로 확장이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 우리의 합성 데이터를 훈련에 포함한 모델이 기존 단일 언어 및 다국어 데이터세트에서 훈련한 모델보다 일반화 성능이 향상되었습니다. 특히 스페인어 및 독일어에서 매크로 F1 점수가 유의미하게 증가했습니다. 데이터와 전체 구현은 GitHub를 통해 오픈 소스로 제공될 예정입니다.



### MHQA: A Diverse, Knowledge Intensive Mental Health Question Answering Challenge for Language Models (https://arxiv.org/abs/2502.15418)
- **What's New**: 정신 건강 문제는 세계적으로 증가하는 이슈로, 본 연구는 MHQA(정신 건강 질문 응답)라는 새로운 다중 선택 데이터세트를 제안합니다. 기존의 정신 건강 관련 데이터세트는 주로 텍스트 분류에 집중해 왔으나, MHQA는 불안, 우울증, 외상, 강박적/강제적 문제를 포함한 네 가지 주요 도메인에 대한 질문 응답 작업을 제공합니다. 이 데이터세트는 PubMed 초록을 근거로 하여 개발되었으며, 2,475개의 전문가 검증된 QA 쌍과 약 56,100개의 외부 의료 참고자료를 통해 의사결정이 이루어진 QA 쌍을 포함하고 있습니다.

- **Technical Details**: MHQA 데이터세트는 58,600개 QA 쌍으로 구성되어 있으며, 각 질문은 네 개의 선택지와 한 개의 정답을 포함하고 있습니다. 데이터세트는 불안, 우울증, 외상, 강박 및 강제적 문제와 관련된 질문을 포괄하여, 각각의 질문은 사실적, 진단적, 예후적, 예방적 질문 유형으로 분류됩니다. 우리는 GPT-4o-mini 모델을 활용하여 PubMed초록에서 정보를 식별하여 QA 쌍으로 전환하는 강력한 파이프라인을 개발했습니다.

- **Performance Highlights**: 논문에서는 다양한 언어 모델에 대해 MHQA-Gold를 기준으로 한 성능 평가를 세부적으로 다루고 있습니다. 각 모델의 F1 점수를 보고했으며, Few-shot 및 Supervised Fine-tuning 실험도 수행했습니다. 연구 결과, 모델별로 특정 도메인과 질문 유형에 따른 성과를 논의하면서, MHQA 데이터세트가 NLP 모델을 통해 정신 건강 QA의 발전과 테스트를 촉진할 수 있음을 강조하였습니다.



### Textual-to-Visual Iterative Self-Verification for Slide Generation (https://arxiv.org/abs/2502.15412)
Comments:
          Work in progress

- **What's New**: 자동화된 프레젠테이션 슬라이드 생성은 시간을 소모하는 작업으로, 현재 연구에서는 이를 위한 혁신적인 접근 방식을 소개하고 있습니다. 이 방식은 콘텐츠 생성(content generation)과 레이아웃 생성(layout generation)으로 작업을 세분화하여, 학술 슬라이드를 만들 때의 일반적인 절차와 일치하도록 구성되었습니다. 본 논문은 특히 LLM(large language model)을 사용하여 문맥과 섹션 정보를 활용해 더 일관된 슬라이드 생성을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법에서는 텍스트에서 시각적 레이아웃으로의 변환 과정을 통해, 복잡한 텍스트 레이아웃을 직관적이고 비주얼한 형식으로 변환합니다. 이는 LLM 기반의 Reviewer + Refiner 워크플로우를 통해 수행되며, 시각적 참조를 통해 레이아웃의 정확성을 높이는 자기 검증(self-verification) 프로세스가 포함됩니다. 초기 레이아웃은 텍스트 형식으로 생성되지만, 이로 인해 발생하는 오류를 시각적으로 확인하고 수정할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 여기서 제안된 접근 방식은 기존 방법들보다 슬라이드의 정렬(alignment), 논리적 흐름(logical flow), 시각적 매력(visual appeal), 그리고 가독성(readability) 등에서 유의미하게 향상된 성능을 보입니다. 실험 결과는 자동화된 학술 슬라이드 생성의 효과성과 잠재력을 체계적으로 입증하며, 과학적 커뮤니케이션을 더욱 효율적으로 할 수 있는 기반을 제공합니다. 이를 통해 연구자들은 더 많은 시간과 자원을 핵심 연구에 집중할 수 있게 될 것입니다.



### HiFi-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings (https://arxiv.org/abs/2502.15411)
- **What's New**: 이번 논문에서는 미국 증권 거래 위원회(SEC)의 요구에 따라 금융 보고를 위한 iXBRL 형식의 복잡한 태그 체계를 활용하여, 비구조적 금융 텍스트에서 KPI(Key Performance Indicator)의 수치를 효과적으로 추출할 수 있는 HiFi-KPI 데이터셋을 소개합니다. HiFi-KPI는 약 218,126개의 라벨로 구성된 계층 구조를 가지고 있으며, 1.8M 개의 문단과 5M 개의 엔터티를 포함하고 있습니다. 또한, HiFi-KPI Lite라는 전문가가 매핑한 라벨을 가진 소규모 데이터셋도 제공하여 LLM(Large Language Models)에서 쉽게 사용할 수 있도록 합니다.

- **Technical Details**: HiFi-KPI 데이터셋은 SEC의 10-K 및 10-Q 보고서에서 수집된 문서들을 기반으로 하며, 2017년 1월 1일부터 2024년 6월 1일 사이의 모든 관련 보고서가 포함되었습니다. 왕복적인 접근을 통해 iXBRL 태그와 상관관계가 있는 기간이나 수치적 값들을 유지하며, 이를 통해 태그 간의 관계가 있는 맥락을 보존할 수 있습니다. 논문에서는 encoder 기반 접근법과 대규모 언어 모델을 이용한 구조적 추출 방법에 대한 기준선을 제공합니다.

- **Performance Highlights**: HiFi-KPI 데이터셋은 다수의 다운스트림 작업을 지원하며, 텍스트 분류, 시퀀스 레이블링, 구조적 정보 추출 등에서 뛰어난 성능을 발휘합니다. 기존 데이터셋들과 비교하여, HiFi-KPI는 2.77개의 태그가 문장당 포함되어 있어 정보의 충분함을 최대화하고 있습니다. 이 데이터셋은 다양한 회사의 보고서를 구조적으로 비교하고 분석하는 데 유용할 것으로 기대됩니다.



### Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning (https://arxiv.org/abs/2502.15401)
- **What's New**: 본 연구는 문제 해결 논리를 기반으로 한 커리큘럼(Iclumn) ICL 방법론을 제안합니다. 이 방법은 대조군 데이터셋(BREAK)을 이용하여 예제의 선정과 순서를 최적화하고, 문제 해결 단계를 기준으로 난이도를 평가합니다. 기존 간단한 통계 정보에 의존한 접근 방식에 비해 내재적인 예제 간 연결성을 더욱 효과적으로 반영합니다.

- **Technical Details**: 제안된 방법론은 문제 해결 논리(Problem-Solving Logic)를 기반으로 하여, LLM이 문제를 해결하는 데 필요한 단계와 그 순서를 정의합니다. QDMR(Question Decomposition Meaning Representation) 기법을 활용하여 복잡한 문제를 여러 개의 하위 질문으로 분해하고, 이를 통해 수집된 데이터를 바탕으로 예제의 유사성을 분석합니다. 커리큘럼 학습 원칙에 따라 쉬운 예제에서 어려운 예제로 나열되는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과, 본 방법론은 다수의 벤치마크에서 기존 ICL 방법들보다 성능 및 효율성을 향상시키는 것으로 나타났습니다. 평균 성능이 향상되었고, 복잡한 추론 작업에 대한 LLM의 능력도 유의미하게 증가했습니다. 이러한 결과는 다양한 데이터셋에서 일관되게 확인되었습니다.



### Evaluating Social Biases in LLM Reasoning (https://arxiv.org/abs/2502.15361)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)에서의 편견(bias) 문제를 체계적으로 평가하고 분석했습니다. 이전의 연구들이 주로 수학이나 코딩 작업에 집중했던 것과 달리, 이 연구는 LLM의 추론 과정에서 발생하는 사회적 편견의 증폭을 탐구합니다. 이러한 연구는 편견이 포함된 논리적 주장 형성이 야기할 수 있는 부정적인 결과를 강조합니다.

- **Technical Details**: 연구는 BBQ 데이터셋을 활용하여 LLM의 추론 단계에서 발생하는 편견을 평가했습니다. 저자들은 두 가지 개수(8B, 32B)의 DeepSeek-R1 변형을 조사하고, LLM-as-a-judge 방식으로 각 추론 단계의 편견 강도를 스코어링했습니다. 연구를 통해 잘못된 응답에서 사회적 고정관념의 언급 빈도가 상당히 높음을 발견했습니다.

- **Performance Highlights**: 결과적으로, 고정관념이 없는 추론 패턴이 개선된 모델 성능과 강하게 연관되어 있다는 사실이 밝혀졌습니다. 본 연구는 LLM이 정확성을 향상시킬 수 있음에도 불구하고 많은 경우 편견이 증폭될 수 있음을 보여줍니다. 특히 애매한 맥락에서 편견을 더 강화하는 경향이 있는 것으로 나타났습니다.



### AttentionEngine: A Versatile Framework for Efficient Attention Mechanisms on Diverse Hardware Platforms (https://arxiv.org/abs/2502.15349)
Comments:
          15 pages

- **What's New**: 이번 논문에서는 AttentionEngine이라는 포괄적인 프레임워크를 소개합니다. 이 프레임워크는 다양한 하드웨어 플랫폼에서 Attention 메커니즘의 최적화를 지원하며, 수동 조작을 최소화하여 성능 향상을 도모합니다. AttentionEngine은 모듈화된 연산으로 Attention 계산을 분해하여 다양한 알고리즘 요구 사항에 유연하게 적응할 수 있도록 합니다.

- **Technical Details**: AttentionEngine은 두 가지 기본 연산, 즉 relevance scoring과 aggregation으로 Attention 메커니즘을 추상화합니다. 이를 통해 상관성 점수를 계산하고, 사용자 정의 기능을 추가하여 사용자가 원하는 주의 변형을 디자인할 수 있는 템플릿을 제공합니다. 이 프레임워크는 입력 구성 및 하드웨어 제약에 맞춰 자동으로 최적화를 수행하여 다양한 Attention 변형과 하드웨어 플랫폼을 지원합니다.

- **Performance Highlights**: 실험 결과, AttentionEngine은 기존의 수작업 최적화된 커널과 동급의 성능을 발휘하며, 지원되지 않는 구성에서는 최대 10.4배의 속도 향상을 자랑합니다. 또한, AttentionEngine는 사용자 정의 Attention 메커니즘의 설계 및 최적화에 대한 유례없는 유연성을 제공하여 확장 가능하고 일반화 가능한 Attention 계산의 중요한 진전을 나타냅니다.



### Constructing a Norm for Children's Scientific Drawing: Distribution Features Based on Semantic Similarity of Large Language Models (https://arxiv.org/abs/2502.15348)
- **What's New**: 이번 연구는 아동의 과학 그림을 분석하기 위해 Large Language Model (LLM)을 활용하고, 1420개의 그림을 다루어 일관된 그림 표현이 존재하는지를 탐구합니다. 기존 연구의 문제점을 해결하고, 아동의 그림 연구에 대한 기초 데이터를 제공하는 것을 목표로 하고 있습니다. 연구의 결과로는 대부분의 그림이 0.8 이상의 의미적 유사성을 보여 일관성이 있음을 확인했습니다.

- **Technical Details**: 연구에서는 word2vec 알고리즘을 사용하여 아동의 그림 간 의미적 유사성을 계산하였으며, 9가지 과학 주제에 대한 그림을 분석했습니다. Kendall rank correlation coefficient를 사용하여 샘플 크기(Sample Size), 추상 정도(Abstract Degree), 초점(Focus Points)가 그림에 미치는 영향을 조사하였고, 단어 빈도 통계를 통해 아동이 수업에서 배운 내용을 재현하여 추상 주제를 표현하는 방식을 탐구했습니다.

- **Performance Highlights**: 이번 연구는 아동의 과학 그림 표현의 일관성을 밝혀내어, 후속 연구를 위한 기준을 제시합니다. 그림의 의미적 유사성이 높게 나타났으며, 정확성과는 별개로 일관성 편향(Consistency Bias)을 발견하였습니다. 이러한 결과는 아동의 사고 방식 이해와 교육 분야에 중요한 통찰을 제공합니다.



### Tokenization is Sensitive to Language Variation (https://arxiv.org/abs/2502.15343)
- **What's New**: 이번 연구는 언어 변화가 LLM(대형 언어 모델)의 성능에 미치는 영향을 조사합니다. 다양한 토크나이저 설정에 따라 모델 성능이 어떻게 달라지는지를 비교하며, 특정 작업(semantic tasks와 form-based tasks)에 대한 적합성을 강조합니다. 연구진은 또한 새로운 방법론을 도입해 토크나이저의 영향을 측정하며, 기존의 효율성 측정 방법보다 개선된 결과를 보여줍니다.

- **Technical Details**: 이번 연구에서는 BERT base 모델을 Byte-Pair Encoding(BPE) 알고리즘으로 사전 훈련하여, 모델 성능에 영향을 미치는 주요 알고리즘 설계 선택을 조사했습니다. 토크나이저 설정에는 적합한 말뭉치(fitting corpus), 사전 토크나이저(pre-tokenizer) 및 어휘 크기(vocabulary size) 등이 포함됩니다. 이 연구를 통해 사전 토크나이저의 영향력이 성능 향상에 가장 큰 영향을 미친다는 것을 발견했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 토크나이저의 성능이 작업의 요구 사항(언어 변화에 대한 내구성 요구 작업과 민감성 요구 작업)에 따라 달라진다고 설명합니다. 로지스틱 회귀를 사용한 분석에서, 기존의 지표들보다 BERT의 성능과 높은 상관관계를 가지며, 이는 더욱 효율적인 토크나이저 구축을 위한 기초 자료로 활용될 수 있습니다.



### Stepwise Informativeness Search for Improving LLM Reasoning (https://arxiv.org/abs/2502.15335)
Comments:
          Preprint

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 다단계 추론 능력을 향상시키기 위해 새로운 접근방식을 제안합니다. 구체적으로, 저자들은 LLM이 정보를 덜 활용한 이전 단계에서 정보를 적극적으로 참조하고 불필요한 중복 정보를 최소화하는 방식으로, 정확하고 간결한 단계별 설명을 생성하도록 유도합니다. 이를 위해 'stepwise informativeness search'라는 프레임워크는 두 가지 선택 휴리스틱을 기반으로 작동합니다.

- **Technical Details**: 제안된 프레임워크는 'stepwise beam search' 접근 방식을 사용하여 다단계 추론 문제를 해결합니다. 각 단계에서 모델은 병렬로 추론 단계를 생성하며, 이를 통해 시간 소모가 큰 Monte Carlo Tree Search 방법에 비해 효율적으로 작업을 수행할 수 있습니다. 선택 과정에서는 'grounding-guided selection'과 'novelty-guided selection' 같은 두 가지 휴리스틱을 고려하여 더 유용한 후보를 선택하게 됩니다.

- **Performance Highlights**: 실험 결과는 다양한 다단계 추론 데이터셋에서 제안된 접근 방법이 낮은 오류율로 향상된 설명 품질과 높은 추론 정확도를 달성했음을 보여줍니다. 이 프레임워크는 모델의 고유한 출력과 주의 점수를 활용하여 단계 검색을 더욱 효과적으로 유도할 수 있도록 설계되었으며, 도메인에 구애받지 않는 적용 가능성을 지니므로 광범위하게 활용될 수 있습니다.



### Detecting Future-related Contexts of Entity Mentions (https://arxiv.org/abs/2502.15332)
- **What's New**: 이 논문은 자동으로 엔티티가 미래 문맥에서 언급되었는지를 식별하는 기술에 대해 다루고 있으며, 이를 위해 새로운 데이터셋인 19,540개의 문장을 소개합니다. 해당 데이터셋은 Wikipedia에서 수집한 인기 있는 엔티티를 기반으로 하여 미래 관련 및 비미래 관련 문맥으로 구성됩니다. 이러한 작업은 정보 처리에서 자동화된 시간 분석에 대한 증가하는 요구를 해결하기 위한 것으로, 기존 연구의 한계를 보완하고자 합니다.

- **Technical Details**: 논문에서는 엔티티 중심 텍스트에서 미래 언급을 감지하고 분류하는 접근 방식을 제시합니다. 여기서는 기존의 명시적인 시간 표현에만 집중했던 전통적 방법 대신, 문맥적 단서를 기반으로 미래 지향적 콘텐츠를 식별하는 데 중점을 두었습니다. 데이터셋 구축 과정에서 BERT 모델을 활용하여 미래 관련 문장을 분류하고, 다양한 기계 학습 모델과 최신 Large Language Model의 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 전통적인 기계 학습 방법 및 transformer 기반 모델 모두를 통해 미래 관련 문맥을 구별하는 성능이 확인되었습니다. 특히, BERT와 같은 최첨단 모델들은 미래 지향적 내용의 식별에서 우수한 성능을 보였으며, LLMs의 활용 가능성도 탐구되었습니다. 이러한 성과는 미래 예측 및 관련 정보 검색 기술의 발전에 기여할 것으로 예상됩니다.



### Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inferenc (https://arxiv.org/abs/2502.15294)
- **What's New**: 이 논문은 대형 언어 모델(LLM)에서의 대화 데이터 분석을 통해 대화의 라운드 수가 증가할수록 KV 캐시가 GPU 메모리에 저장되어 효율성을 저해하는 문제를 발견합니다. 이를 해결하기 위해 'Round Attention'이라는 새로운 라운드 수준의 주의 메커니즘을 제안합니다. 이 메커니즘은 가장 관련성 높은 라운드의 KV 캐시만을 회상하고 계산하여 효율성을 높입니다.

- **Technical Details**: 제안된 라운드 주의 메커니즘은 대화 라운드 수가 많아짐에 따라 발생하는 메모리 사용 문제를 해결하기 위해, 특정 라운드에 국한된 KV 캐시를 활용합니다. 이는 모델의 성능을 유지하면서도 메모리 사용량을 약 55% 줄이는 데 성공합니다. 실험은 모델의 효율성을 높이기 위해 다양한 대화 데이터 세트를 사용하여 진행되었습니다.

- **Performance Highlights**: 이 연구 결과는 Round Attention이 메모리 비용을 크게 줄여줄 뿐만 아니라 대형 언어 모델의 성능을 유지시킨다는 것을 보여줍니다. 이는 특히 긴 텍스트와 복잡한 작업을 처리하는 데 있어, LLM을 사용하는 시스템의 효율성을 크게 향상시킬 수 있는 잠재력이 있습니다.



### Analyzing the Inner Workings of Transformers in Compositional Generalization (https://arxiv.org/abs/2502.15277)
Comments:
          Accepted to NAACL 2025 main

- **What's New**: 이번 연구는 신경 모델의 조합 일반화(compositional generalization) 능력을 평가하는 전통적인 방법의 한계를 지적하고, Transformer 모델의 내부 메커니즘을 분석하여 어떤 구문(syntactic feature) 특성에 의존하는지를 조사합니다. 기존의 연구들이 모델의 출력(output)에 중점을 두었음을 감안할 때, 본 연구는 신경망 내부의 서브네트워크(subnetwork)와 구문 특징에 대한 인과(in causal analysis) 분석을 통해 더 깊이 이해할 수 있는 기회를 제공합니다. 이를 통해, 모델의 조합적인 규칙을 이해하고 내재된 능력을 향상시키는 방법을 모색하고자 합니다.

- **Technical Details**: 연구 방법론은 크게 세 가지 단계로 나뉘며, 기본 모델 학습, 서브네트워크 탐색, 인과 분석으로 구성됩니다. 학습 데이터셋은 훈련(train), 기준(in-distribution test), 외부(out-of-distribution generalization) 일반화 세트로 구성되어 있으며, 이 일반화 세트는 훈련 세트에 없는 구문 구조를 포함합니다. 연구팀은 Transformer 모델을 처음부터 학습하고, 머신 번역(machine translation)과 의미 해석(semantic parsing)이라는 두 가지 과제를 통해 조합 일반화 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 모델과 그 서브네트워크는 구문 구조를 활용하여 일반화 성능에서 우수한 결과를 보였습니다. 특히 서브네트워크는 전체 모델에서 더 나은 일반화 성능을 보였으나, 흥미롭게도 비조합(non-compositional) 알고리즘을 추가적으로 활용한다는 사실을 발견하였습니다. 또한, 서브네트워크는 훈련 초기 단계에서 비조합 솔루션을 학습하며, 훈련 과정에서 일반화 성능이 점진적으로 향상된다는 것을 알 수 있었습니다.



### A Training-free LLM-based Approach to General Chinese Character Error Correction (https://arxiv.org/abs/2502.15266)
Comments:
          25 pages, 12 figures

- **What's New**: 이번 논문에서는 중국어 철자 교정(Chinese spelling correction, CSC)의 한계를 극복하고자 모든 유형의 문자 오류를 고려하는 일반 중국어 문자 오류 수정(General Chinese Character Error Correction, C2EC) 작업을 제시합니다. 기존에는 주로 입력 실수로 인한 문자 치환 오류에 초점을 맞췄지만, 이번 연구는 누락된 문자 및 중복 문자 오류와 같은 다른 일반적인 오류 유형도 포괄합니다. 이로써 CSC 작업의 실용성을 크게 향상시키는 기초가 될 것입니다.

- **Technical Details**: C2EC 벤치마크는 CCTC와 Lemon 데이터셋의 데이터를 통합하고 수동으로 검증하여 구축됐습니다. 우리는 길이 변화 처리를 위해 Levenshtein 거리(Levenshtein distance)를 사용하여 훈련 없이 프롬프트 없는 CSC 방법을 C2EC로 확장하고, 성능 향상을 위해 추가적인 프롬프트 기반 대형 언어 모델(large language model, LLM)을 활용하였습니다. 이러한 접근 방식은 다양한 문자 오류를 동시에 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 우리 방법은 14B 파라미터의 LLM이 특정한 CSC와 C2EC 작업에서 약 50배 더 큰 모델과 동등한 성능을 보이는 것으로 확인되었습니다. 이는 어떠한 미세 조정(fine-tuning)도 없이 이루어졌다는 점에서 상당한 발전을 보여줍니다. 결과적으로, 제안된 방법은 실용적인 문자 오류 수정에서 중요한 기여를 할 것으로 기대됩니다.



### Retrieval-Augmented Speech Recognition Approach for Domain Challenges (https://arxiv.org/abs/2502.15264)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 LLM 기반의 음성 인식 시스템을 제안합니다. 기존의 음성 인식 모델이 훈련 단계에서 도메인 전용 데이터에 의존하는 것과 달리, 제안된 접근법은 추론 단계에서 도메인 전용 텍스트 데이터를 활용합니다. 이를 통해 도메인 불일치 문제를 해결하고, 성능을 향상시키는 결과를 보여줍니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 구성요소, 즉 콘텐츠 임베딩 데이터베이스 준비, 도메인 특정 콘텐츠 검색, 그리고 LLM 강화 음성 인식으로 이루어져 있습니다. 훈련 및 추론 단계에서 도메인 특정 문서가 검색되며, 검색된 문서는 LLM 디코더에 통합되어 추론의 정확도를 높입니다. 또한, 음성 인식 모델은 두 가지 단계로 나누어 음성 인코더 최적화와 LLM 디코더 최적화를 수행합니다.

- **Performance Highlights**: 우리의 실험 결과, CSJ 데이터베이스를 기반으로 한 제안된 시스템은 도메인 외 시험에서 19.6%의 상대적 성능 향상을 보여주었으며, 기존의 평가 세트에서 최첨단 결과를 기록했습니다. 특히, 이러한 성능 향상은 일반적인 훈련 데이터의 일부만을 사용했음에도 불구하고 달성되었습니다.



### Corrections Meet Explanations: A Unified Framework for Explainable Grammatical Error Correction (https://arxiv.org/abs/2502.15261)
Comments:
          19 pages, 2 figures, and 9 tables

- **What's New**: 이 논문에서는 언어 학습자를 위한 설명 가능한 문법 오류 수정 시스템(EXGEC)을 소개하고 있습니다. 기존 연구들이 교정 과정과 설명 간의 관계를 무시하는 반면, EXGEC는 이 두 가지 작업이 서로 강화될 수 있도록 설계되었습니다. 또한, 데이터셋 EXPECT에서 발견된 많은 노이즈 문제를 해결하기 위해, 수정된 EXPECT-denoised 데이터셋을 제안하여 보다 객관적인 학습 프레임워크를 제공하고 있습니다.

- **Technical Details**: EXGEC는 교정(correction) 및 설명(explanation) 작업을 통합하는 통합된 다중 작업 프레임워크로, 이를 통해 상호 작용(interaction)을 강화합니다. 논문은 EXPECT 데이터셋을 기반으로 하여, 심층 신경망 모델을 활용해 문법 오류를 감지하고, 증거 단어(extractive rationales)를 추출하며, 오류를 분류하는 방식을 구체적으로 다룹니다. 예를 들어, 선행 설명(pre-explaining) 모델과 후행 설명(post-explaining) 모델의 성능 차이를 분석하여, 예측 순서가 작업 성능에 미치는 영향을 밝혀냈습니다.

- **Performance Highlights**: EXGEC는 BART, T5 및 Llama3와 같은 여러 자연어 처리(NLP) 모델에서 교정 및 설명 작업 모두에서 단일 작업 기준선(single-task baselines)을 초과하는 성능을 보여주었습니다. 실험 결과, EXGEC 모델은 기존의 단일 작업 모델보다 교정 및 설명 작업에서 더 나은 성능을 보이며, 서로의 긍정적 상호 작용을 입증했습니다. 이는 EXGEC가 다양한 예측 순서에서 이 두 가지 작업의 관계를 효과적으로 탐구할 수 있음을 시사합니다.



### LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design (https://arxiv.org/abs/2502.15260)
Comments:
          Accepted by DATE 2025

- **What's New**: Mamba와 같은 상태 공간 모델(SSMs)은 최근에 주목받고 있으며, 이 논문에서는 이러한 모델의 처리 속도를 높이기 위한 새로운 접근법인 LightMamba를 제안합니다. 기존의 Transformer 기반 대형 언어 모델(LLMs)과 비교하여 Mamba는 시퀀스 길이에 따라 선형 계산 복잡도를 요구하며, 우수한 성능을 보입니다. 하지만 Mamba의 활성화 아웃라이어 분포가 산재해 있어 기존의 LLM 가속기가 효율적이지 않은 문제가 있습니다.

- **Technical Details**: LightMamba는 양자화 알고리즘과 FPGA 가속기 아키텍처를 공동 설계하여 Mamba 추론을 효과적으로 가속화하도록 설계되었습니다. 여기서는 회전 지원 양자화(rotation-assisted quantization)와 2 제곱의 SSM 양자화(power-of-two SSM quantization) 방법을 통해 대부분의 계산을 4비트로 줄이는 FPGA 친화적인 후속 훈련 양자화(PTQ) 알고리즘을 제안합니다. 또한, Mamba 연산을 부분적으로 언롤(partially unroll)하는 FPGA 가속기를 설계하여 효율성과 하드웨어 비용의 균형을 맞추었습니다.

- **Performance Highlights**: LightMamba는 Xilinx Versal VCK190 FPGA에서 구현되어 GPU 기준 대비 4.65배에서 6.06배 높은 에너지 효율성을 기록했습니다. Alveo U280 FPGA에서 평가했을 때, LightMamba는 초당 93개 토큰을 생성하여 GPU 기준의 1.43배 성능을 자랑했습니다. 이러한 성과는 특히 양자화와 하드웨어 효율성을 극대화하는 새로운 접근법 덕분입니다.



### Understand User Opinions of Large Language Models via LLM-Powered In-the-Moment User Experience Interviews (https://arxiv.org/abs/2502.15226)
- **What's New**: 이 논문에서는 CLUE( Contextualized LLM-powered User Experience understanding)를 소개하며, LLM과의 상호작용 이후 사용자 경험 인터뷰를 자동으로 진행하여 사용자 의견을 수집하는 프레임워크를 제시합니다. CLUE-Interviewer는 사용자가 LLM과 대화한 직후에 반응을 기록하여, 사용자 의견의 깊이 있는 인사이트를 수집하는 새로운 방법론을 제공합니다. 이 방법은 수천명의 사용자를 대상으로 하여, 다양한 주제에 대한 대화와 인터뷰 세션을 기록하였습니다.

- **Technical Details**: CLUE는 두 가지 주요 구성 요소인 CLUE-Interviewer와 CLUE-Insighter로 구성됩니다. CLUE-Interviewer는 사용자와 LLM의 대화 후 반응을 수집하기 위해 반구조화된 UX 인터뷰를 자동으로 수행합니다. 이러한 인터뷰는 미리 정의된 차원에 대한 깊은 통찰을 유도하며, 사용자의 응답에 따라 주제를 탐색할 수 있는 유연성을 가지고 있습니다.

- **Performance Highlights**: 사용자 연구 결과, CLUE-Interviewer는 다양한 주제에 대한 시의적절한 사용자 의견을 효과적으로 수집하는 것으로 나타났습니다. 사용자들은 LLM에 대해 전반적으로 보수적인 의견을 보였지만, 특정 차원에 대해 높은 평가를 하였습니다. 또한, 사용자들은 시각 및 멀티미디어 기능, 신선한 정보 접근, 개인화된 응답 등 새로운 기능 요청을 통해 향후 LLM 개발을 위한 동기와 증거를 제공하였습니다.



### ESPnet-SpeechLM: An Open Speech Language Model Toolk (https://arxiv.org/abs/2502.15218)
- **What's New**: ESPnet-SpeechLM은 음성 언어 모델(SpeechLM) 및 음성 기반 에이전트 응용 프로그램의 개발을 민주화하기 위해 설계된 오픈 툴킷입니다. 이 툴킷은 음성 처리 작업을 보편적인 시퀀스 모델링 문제로 체계화하고 데이터 전처리, 사전 학습, 추론, 작업 평가의 일관된 워크플로우를 포함합니다. 사용자는 작업 템플릿을 쉽게 정의하고 주요 설정을 구성할 수 있으며, 이를 통해 원활한 SpeechLM 개발을 돕습니다.

- **Technical Details**: ESPnet-SpeechLM은 모든 음성 작업을 시퀀스 모델링 문제로 통합하며, 사용자 정의 작업 템플릿을 정의한 후 주요 매개변수를 설정하여 자동화된 파이프라인을 지원합니다. 이 툴킷은 데이터 토큰화 방법, 모델 아키텍처, 동적 다중 작업 등 다양한 디자인 선택을 지원하는 모듈형 워크플로우를 제공합니다. 또한 HuggingFace와 호환되는 인터페이스를 제공하여 데이터셋과 모델 공유를 돕습니다.

- **Performance Highlights**: ESPnet-SpeechLM의 유연성과 효율성을 보여주기 위해 여러 가지 사용 사례를 제공하며, 200k 시간 이상의 음성-텍스트 쌍 데이터셋에서 경쟁력 있는 SpeechLM 기반의 자동 음성 인식(ASR) 및 텍스트-음성 변환(TTS) 시스템을 구축할 수 있습니다. 또한 240억 개의 텍스트 토큰 또는 오디오 프레임을 활용하여 ASR, TTS, TextLM 및 AudioLM 작업을 사전 학습한 17억 개 매개변수를 가진 다중 작업 SpeechLM을 구축하는 방법을 상세히 설명합니다.



### Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing (https://arxiv.org/abs/2502.15208)
Comments:
          9 pages

- **What's New**: 이 논문은 동적 시스템 이론을 통해 대형 언어 모델(LLM)의 반복적 변환 과정과 이들이 안정적인 구성으로 수렴하는 현상을 분석합니다. LLM의 연속적인 패러프레이징(successive paraphrasing)을 실험하며, 이러한 과정이 다양한 언어적 표현을 탐색하기보다는 제한된 주기적 상태로 수렴한다는 사실을 발견하였습니다. 이 연구는 LLM의 생성 능력에 내재된 제약을 강조하고 있으며, 이러한 동적 시스템의 관점이 LLM의 표현 잠재력을 이해하는 데 중요한 역할을 한다는 점을 제안하고 있습니다.

- **Technical Details**: 동적 시스템 이론은 시스템의 상태와 진화를 규명하는 수학적 프레임워크를 제공합니다. 파라프레이징 생성을 이론적으로 다루면서 LLM의 패러프레이징 기능을 반복적 변환으로 보고, 이러한 변환이 주기성 및 수렴성과 같은 관찰된 현상과 어떻게 연결되는지를 설명합니다. 이를 통해 LLM이 출력 텍스트의 상태를 어떻게 변화시키는지, 그 결과로 발생하는 주기적 상태의 패턴을 이해하려고 합니다.

- **Performance Highlights**: 연구 결과, LLM의 패러프레이징은 예측할 수 있는 형태의 반복적인 결과로 수렴하는 경향을 갖습니다. 분석을 통해, LLM이 점점 좁은 해법 집합에 점점 더 확신을 가지며 이러한 주기적 사이클에 수렴한다는 사실이 밝혀졌습니다. 마지막으로, 주기적 사이클을 방해하는 간단한 방법을 제안하며, 이를 통해 모델이 보다 적절한 언어적 변형을 재도입하고 안정적이지만 제한된 주기를 피할 수 있음을 보여줍니다.



### TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding (https://arxiv.org/abs/2502.15197)
Comments:
          15 pages, 10 figures, 5 tables

- **What's New**: 본 논문에서는 다중 요청 설정에서 배치 추측 디코딩(batch speculative decoding)의 전체 처리량을 최적화하는 새로운 방법인 TETRIS를 제안합니다. 기 기존 방법들과 달리, TETRIS는 병렬 검증에서 승인될 가능성이 높은 드래프트 토큰(draft tokens)을 적극적으로 선택하여, 거부되는 토큰을 줄이고 컴퓨팅 리소스를 효율적으로 활용합니다. 이러한 접근은 특히 제한된 추론 용량을 가진 서비스 제공자에게 빠른 추론을 가능하게 합니다.

- **Technical Details**: TETRIS는 각 요청의 토큰이 검증될 때 수용될 가능성이 높은 드래프트 토큰을 그리드 방식으로 선택하는 방식으로, 이는 전체 처리량을 최대화하는 데 중점을 둡니다. 이 방법론은 토큰의 수용 가능성을 고려하여 최적의 드래프트 토큰 선택을 동적으로 결정하며, 이는 컴퓨터 자원의 활용을 더욱 개선합니다. TETRIS는 이론적으로 각 디코딩 단계에서 최적 처리량을 달성한다고 증명합니다.

- **Performance Highlights**: TETRIS는 기존의 추측 디코딩에 비해 일관되게 더 높은 수용률을 기록하며 제한된 추론 용량의 활용도를 더욱 효과적으로 증가시킵니다. 실험 결과, TETRIS는 표준 추측 디코딩 및 동적 드래프트 윈도우를 이용한 기존 방법들에 비해 전체 처리량과 지연(latency)가 개선되었습니다. 이는 TETRIS가 실제 모델 서비스 배포에서 추론 속도를 개선할 수 있는 잠재력을 지니고 있음을 나타냅니다.



### Scale-Free Graph-Language Models (https://arxiv.org/abs/2502.15189)
- **What's New**: 이번 논문은 그래프-언어 모델(GLMs)을 새롭게 발전시켜 그래프 생성(graph generation)과 텍스트 임베딩(text embedding)을 통합한 단일 프레임워크를 제안합니다. 특히, 실제 엣지 분포의 고유한 특성인 스케일-프리 속성을 사용해 그래프 생성을 이루며, k-최근접 이웃(kNN) 그래프를 통한 근사화에 중점을 둡니다. 이러한 접근 방식은 LM의 미세 조정(finetuning)을 위한 보조 감독(supervision)을 제공하는 그래프 기반의 유사 라벨러(pseudo-labeler)를 개발하는 데 도움을 줍니다.

- **Technical Details**: 제안된 모델인 스케일-프리 그래프 언어 모델(SFGL)은 그래프 생성 및 텍스트 임베딩 문제를 동시에 해결합니다. 연구팀은 LMs와 GNNs(그래프 신경망)의 시너지를 활용하여 실제 스케일-프리 구조를 기반으로 한 그래프 생성과 텍스트 임베딩의 통합을 시도했습니다. 이 모델은 잘 정의된 구조적 선행 사전(structural prior)를 통해 학습의 복잡성과 편향을 줄입니다.

- **Performance Highlights**: 제안된 SFGL 모델은 다양한 데이터세트에서 실험을 통해 KNN 그래프의 스케일-프리 구조 근사화의 유효성을 입증하였습니다. 이 모델은 처리 성능을 향상시키기 위해 반복적인 훈련(iterative training)을 실시할 수 있으며, 학습된 LMs는 더 풍부한 의미 정보를 담은 텍스트 임베딩을 생성합니다. GLMs의 잠재력을 강조한 이 연구는 스케일-프리 구조에 기반한 접근 방식을 통해 그래프 기반 반감독 학습(semi-supervised learning)의 신규 가능성을 제시합니다.



### mStyleDistance: Multilingual Style Embeddings and their Evaluation (https://arxiv.org/abs/2502.15168)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2410.12757

- **What's New**: 이 논문에서는 영어 스타일 임베딩만 사용 가능했던 기존의 한계를 극복하고, 다국어 스타일 임베딩 모델인 Multilingual StyleDistance(mStyleDistance)를 새로운 방법으로 제안합니다. 이 모델은 합성 데이터(synthetic data)와 대조 학습(contrastive learning)을 기반으로 하며, 아랍어, 독일어, 스페인어 등 9개 언어의 데이터를 활용하여 훈련되었습니다. mStyleDistance는 이러한 다국어 스타일 임베딩이 다양한 언어와 특징에 대해 잘 일반화됨을 보여줍니다.

- **Technical Details**: mStyleDistance 모델은 자연어 처리에서의 스타일 전이를 위해 설계되었으며, 다양한 스타일 특징을 다룬 합성 데이터셋인 mSynthStel을 사용하여 훈련되었습니다. 이 데이터셋은 각 언어에서 40개의 스타일 특징을 반영하며, 긍정적(positive) 및 부정적(negative) 예제를 생성하기 위해 GPT-4를 활용합니다. 모델의 품질을 평가하기 위해 다국어 및 교차 언어 STEL-or-Content 평가 벤치마크를 제공하며, 이는 같은 스타일의 문장이 의미적 내용과 무관하게 임베딩 공간에서 가깝게 위치할 수 있는지를 측정합니다.

- **Performance Highlights**: mStyleDistance 모델은 기존 모델들보다 더 높은 성능을 보이며, 다국어 스타일 벤치마크에서 우수한 결과를 도출합니다. 특히, 저자는 다국어 저자 검증 작업에도 이 모델을 적용하여, 다국어에서도 효과적으로 작동함을 보여주었습니다. 연구자들은 이 모델과 데이터를 공개하여 향후 연구에서 활용될 수 있도록 기여하고 있습니다.



### Extreme Speech Classification in the Era of LLMs: Exploring Open-Source and Proprietary Models (https://arxiv.org/abs/2502.15155)
Comments:
          Accepted to 7th International Conference on information systems and management science (ISMS), 2024

- **What's New**: 최근 소셜 미디어 플랫폼의 사용자 수가 증가함에 따라 온라인에서 극단적 발언(extreme speech)의 확산이 증가하고 있습니다. 이 연구는 기존 언어 모델이 중립적 텍스트와 비중립적 텍스트를 구분할 수 있는 능력을 보여주지만, 다양한 극단적 발언의 유형을 분류하는 것은 여전히 도전 과제로 남아 있다는 점을 강조합니다. 특히, 극단적 발언 분류는 사회-문화적 맥락을 깊이 이해해야 하는 복잡한 작업입니다.

- **Technical Details**: 이 연구에서는 Maronikolakis et al. (2022)의 극단적 발언 데이터셋의 인도 하위 집합을 활용하여 LLMs(대형 언어 모델)를 사용하는 효과적인 분류 프레임워크를 개발합니다. 오픈소스 Llama 모델과 클로즈드 소스 OpenAI 모델을 비교 평가한 결과, 사전 학습된 LLMs는 중간 정도의 효율성을 보였지만, 도메인 특정 데이터로의 파인 튜닝(fine-tuning)을 통해 성능을 크게 향상시킬 수 있음을 발견했습니다.

- **Performance Highlights**: GPT 기반 모델은 제로샷(zero-shot) 설정에서 Llama 모델보다 성능이 우수했으나, 파인 튜닝 후 성능 격차가 사라졌습니다. 이는 LLM이 언어적 및 맥락적 뉘앙스를 잘 반영할 수 있는 적응성을 지니고 있음을 보여줍니다. 이 연구는 극단적 발언 분류의 자동화 시스템 개발에 대한 가능성을 제시하고 있습니다.



### Investigating the Adaptive Robustness with Knowledge Conflicts in LLM-based Multi-Agent Systems (https://arxiv.org/abs/2502.15153)
Comments:
          Working in progress

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인해 이들은 단순한 텍스트 생성기를 넘어 다중 에이전트 시스템(MAS)에서 협업과 도구 사용이 가능한 자율 에이전트로 진화하게 되었습니다. 하지만 LLM 기반 MAS의 견고성은 특히 지식 충돌에 직면했을 때 아직 명확히 밝혀지지 않았습니다. 본 논문에서는 경미하거나 작업 중심의 지식 충돌에 직면할 때의 MAS의 견고성을 조사하기 위해 네 가지 포괄적인 메트릭을 설계했습니다.

- **Technical Details**: 먼저 이질적인 에이전트가 도입한 경미한 지식 충돌이 시스템의 견고성에 미치는 영향을 분석한 결과, 이는 의사 결정의 협업성을 개선하는 것으로 나타났습니다. 다음으로 작업 중심의 지식 충돌을 합성하여 이를 한 에이전트에 반영한 결과, 상반된 지식 충돌이 MAS의 견고성에 미치는 영향은 미미하거나 전혀 없음을 발견했습니다. 또한 MAS는 지식 충돌에 대한 의존도를 줄여 대안적인 해결 경로를 채택함으로써 자체 수리(Self-repairing) 능력을 나타낸다는 점이 관찰되었습니다.

- **Performance Highlights**: 아블레이션 연구를 통해 지식 충돌의 수, 에이전트 수, 상호작용 횟수에 대한 실험을 수행했습니다. 그 결과, MAS의 자체 수리 능력에는 본질적인 한계가 있으며, 모든 결과는 다양한 요인에 걸쳐 일관되게 유지되는 것을 확인했습니다. 결론적으로, 지식 충돌은 단순한 장애물이 아니라 LLM 기반 MAS의 적응적 견고성을 이끄는 중요한 원동력임을 밝히며, 에이전트 간의 브레인스토밍을 촉진하기 위해 MAS에 지식 충돌을 적절히 도입할 것을 제안합니다.



### Latent Factor Models Meets Instructions:Goal-conditioned Latent Factor Discovery without Task Supervision (https://arxiv.org/abs/2502.15147)
Comments:
          NAACL 2025

- **What's New**: 이 논문에서는 Instruct-LF라는 새로운 시스템을 제안합니다. 이 시스템은 사용자 지시사항을 바탕으로 목표 지향적인 잠재 요인(latent factor) 발견을 위한 프레임워크로, LLM의 지시 따르기 능력과 통계 모델을 통합합니다. Instruct-LF는 노이즈가 많은 대규모 데이터셋에서도 효과적으로 작동하며, 기존 방법들이 해결하지 못했던 문제를 다룹니다.

- **Technical Details**: Instruct-LF는 두 가지 주요 단계로 구성됩니다: 목표 지향적인 데이터 변환(goal-oriented data transformation) 단계와 잠재 요인 발견(latent factor discovery) 단계입니다. 첫 번째 단계에서는 LLM을 사용하여 목표 관련 속성(goal-related properties)을 생성하고, 이를 통해 입력된 비구조화 데이터셋을 데이터-속성 행렬(data-property matrix)로 변환합니다. 두 번째 단계에서는 상관관계가 있는 속성을 클러스터(clusters)로 그룹화하여 보다 고차원적인 추상 개념을 나타냅니다.

- **Performance Highlights**: 실험 결과 Instruct-LF는 영화 추천, 텍스트 기반 네비게이션, 법률 문서 분류 등에서 높은 성능을 보였습니다. 이 시스템은 기존의 최선 모델보다 5-52% 향상된 성능을 보였고, 인간 평가에서 평균적으로 1.8배 더 자주 선택되었습니다. 이러한 결과들은 Instruct-LF가 사용자 목표에 부합하는 유용한 패턴을 효과적으로 발견할 수 있음을 나타냅니다.



### Do LLMs Make Mistakes Like Students? Exploring Natural Alignment between Language Models and Human Error Patterns (https://arxiv.org/abs/2502.15140)
- **What's New**: 이 연구는 대형 언어 모델(LLM)이 다중 선택 질문(MCQ)에서 학생들이 자주 선택하는 오답(분산항)에 대한 학습 패턴을 얼마나 잘 반영하는지를 탐구합니다. 이는 LLM의 생성 확률과 학생 응답 분포 간의 관계를 조사하기 위한 첫 번째 실증적 연구로, 선택된 자료는 실세계에서의 학생 응답을 포함합니다. 연구자는 오답 선택이 LLM의 생성 가능성과 어떤 연관이 있는지를 분석하여 교육적 맥락에서 LLM의 역할에 대한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 3,202개의 MCQ를 포함한 데이터셋을 활용하여 LLM이 학생들이 오답을 선택하는 패턴을 얼마나 잘 반영하는지를 조사합니다. LLM의 생성 가능성에 대한 정량적 관계를 파악하기 위해 새로운 정렬 점수를 도입하였습니다. 연구는 소형 및 대형 LLM(예: LLaMA와 Qwen)의 오답 선택 경향이 유사하다는 주장을 포함하고 있으며, 이는 LLM의 규모와 무관하게 존재하는 경향으로 나타났습니다.

- **Performance Highlights**: 실험 결과, LLM이 학생들이 선택한 분산항에 대해 중간 정도의 상관관계를 보였으며(Pearson r = 0.28-0.37), 특히 대형 모델이 더 강한 일치를 보였습니다. 또한, LLM이 자주 선택된 오답을 선택하는 경향을 나타내면서, 이는 교육 평가의 설계에 중요한 의미를 갖습니다. 연구 결과는 더 작은 언어 모델도 충분히 유용하게 활용될 수 있음을 보여주며, 인간 전문가들과 협력하여 더 나은 평가 도구를 개발할 수 있는 가능성을 제시합니다.



### Chain-of-Rank: Enhancing Large Language Models for Domain-Specific RAG in Edge Devic (https://arxiv.org/abs/2502.15134)
Comments:
          NAACL 2025 (Findings)

- **What's New**: 이 논문에서는 Retrieval-augmented generation (RAG)와 Large Language Models (LLMs)의 통합이 중요한 진전을 이루었음을 강조합니다. 특히, 도메인 특화 RAG를 통해 LLM이 특정 도메인에서의 학습을 통해 더 정확하게 동작할 수 있도록 하는 방법을 제안하고 있습니다. 이를 통해 자원이 제한된 환경에서도 LLM이 신뢰성 있는 작업을 수행할 수 있도록 하고자 합니다.

- **Technical Details**: 주요 기술적 내용으로는 Chain of Rank (CoR) 방법이 제안되었습니다. CoR는 이전의 복잡한 추론과정에서 벗어나 입력 외부 문서의 신뢰도를 단순히 순위화하여 계산 복잡성을 줄입니다. 이는 LLM이 기계적으로 중요한 정보에 집중할 수 있게 하여 최종 답변의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 논문에서 제안한 CoR 방법은 벤치마크에서 최첨단(SOTA) 성능을 달성했습니다. 이는 계산 리소스가 제한된 환경에서도 도메인 특정 작업을 효과적으로 수행할 수 있도록 해주며, 리소스 효율성과 정확도의 균형을 잘 이루고 있습니다.



### CoT-ICL Lab: A Petri Dish for Studying Chain-of-Thought Learning from In-Context Demonstrations (https://arxiv.org/abs/2502.15132)
Comments:
          22 pages, 27 figures, 3 tables

- **What's New**: CoT-ICL Lab는 언어 모델에서 체인-오브-스Thought(Chain-of-Thought, CoT)와 인-컨텍스트 학습(In-Context Learning, ICL)을 연구하기 위한 새로운 프레임워크이자 방법론입니다. 이 연구는 토큰화된 합성 데이터 세트를 생성하고, 인-컨텍스트 예제의 복잡성에 대한 세밀한 제어를 제공합니다. 기존의 연구와는 달리, CoT-ICL Lab는 입력과 체인 토큰을 이산 토큰 공간에서 처리하여 자연어 처리와 밀접하게 연결됩니다.

- **Technical Details**: CoT-ICL Lab는 인과 구조(causal structure)와 토큰 처리 함수(token processing functions)를 분리하여 복잡한 문제를 처리할 수 있는 유연성을 제공합니다. 이 프레임워크는 방향 비순환 그래프(Directed Acyclic Graph, DAG)를 사용하여 체인 생성을 조정하고, 다양한 수준의 토큰 변환을 위해 다층 퍼셉트론(Multi-Layer Perceptron, MLP)을 활용합니다. 또한, 연구자들이 복잡한 문제를 더 잘 이해할 수 있도록 다양한 구성 요소를 조정할 수 있도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, CoT가 있는 인-컨텍스트 학습은 모델 성능의 전환을 가속화하는 데 기여함을 보여주었습니다. 특히, 모델의 깊이가 적은 예제에서 CoT의 활용에 중요한 것으로 나타났고, 더 많은 예제가 얕은 모델이 깊은 모델의 성능을 따라가는 데 도움이 되었습니다. 이를 통해 CoT-ICL Lab가 언어 모델에서 ICL과 CoT에 대한 이론적 및 경험적 통찰력을 제공하는 단순하면서도 강력한 테스트베드로 작용함을 강조했습니다.



### Unveiling Reasoning Thresholds in Language Models: Scaling, Fine-Tuning, and Interpretability through Attention Maps (https://arxiv.org/abs/2502.15120)
- **What's New**: 이번 연구는 다양한 크기와 훈련 데이터를 가진 decoder-only transformer 기반 언어 모델의 in-context learning (ICL) 능력을 조사합니다. 특히, 16억 개 이상의 매개변수를 가진 모델이 Commonsense Reasoning과 Deductive Reasoning 같은 작업에서 상당히 향상된 성능을 나타내는 중요한 매개변수 임계값을 식별했습니다. 이러한 연구는 CoT (chain-of-thought) prompting 기법에서 중요한 발전을 보여주며, 모델의 크기가 성능에 미치는 영향을 강조합니다.

- **Technical Details**: 모델의 추론 능력은 다양한 크기의 transformer 구조에서 분석되었으며, 특히 10억 개 이하의 매개변수를 가진 모델을 중심으로 새로운 발견이 이루어졌습니다. 연구에서는 CommonsenseQA 및 PrOntoQA-OOD 데이터셋을 사용하여 모델의 성능을 평가하였으며, 1.6억 개의 매개변수를 가진 모델이 더 나은 성공률을 보이는 것을 정량적으로 증명하였습니다. Fine-tuning을 통한 업무별 예시로 저성능 모델의 추론 성능을 향상시킬 수 있다는 점도 강조되었습니다.

- **Performance Highlights**: 연구 결과, 16억 개의 매개변수를 초과하는 모델들이 더 긴 추론 체인이 필요한 작업에서 특히 뛰어난 성능을 보여준다고 나타났습니다. 모델의 attention map 분석을 통해 올바른 CoT를 생성할 수 있는 모델은 후속 토큰과 올바른 품사의 높은 token-level attention 점수를 보여줍니다. 이 연구는 매개변수 임계값을 식별하고 sub-threshold 모델의 성능을 개선하는 방법을 제안하며, 다양한 크기의 모델 간 성능 차이에 대한 통찰력을 제공합니다.



### Social Genome: Grounded Social Reasoning Abilities of Multimodal Models (https://arxiv.org/abs/2502.15109)
Comments:
          Under Review, 22 pages

- **What's New**: 이번 논문에서는 멀티모달 모델의 기초적이고 구체적인 사회적 추론 능력을 평가하기 위한 새로운 벤치마크인 Social Genome을 소개합니다. Social Genome은 272개의 상호작용 비디오와 1,486개의 인간 주석이 포함된 추론 경로(reasoning traces)를 포함하며, 이들 경로는 5,777개의 추론 단계로 구성되어 있습니다. 이 벤치마크는 시각적 단서, 언어적 단서, 음성 단서 및 외부 지식을 참조하여 사회적 정보를 해석하는 데 필요한 능력을 평가합니다.

- **Technical Details**: Social Genome은 비디오의 시각적, 언어적, 음성적 단서와 외부 지식을 참고하여 사회적 추론을 수행하는 알고리즘 개발을 위한 중요한 기초 자료입니다. 각 추론 단계는 정보의 참조 방식에 따라 태그가 지정되어 있으며, 이는 11,000개 이상의 개체, 5,000개 이상의 멀티모달 단서 및 2,900개 이상의 외부 지식을 포함합니다. 이 논문은 모델이 생성한 사회적 추론 경로의 의미론적 및 구조적 측면을 평가할 수 있는 메트릭(metrics)을 정의합니다.

- **Performance Highlights**: 실험 결과, 최신 모델들이 사회적 추론에서의 차이를 여전히 보이고 있으며, 제로샷(zero-shot) 및 인컨텍스트 학습(in-context learning) 설정에서 성능이 부족하다는 것을 알 수 있었습니다. Social Genome 벤치마크를 통해 연구자들은 멀티모달 모델의 기초적인 사회적 추론 능력을 향상시키기 위한 갭과 기회를 식별할 수 있었습니다. 이러한 발견들은 AI 시스템의 사회적 상호작용 처리 능력을 향상시키는 데 기여할 것으로 예상됩니다.



### LUME: LLM Unlearning with Multitask Evaluations (https://arxiv.org/abs/2502.15097)
- **What's New**: 최근의 데이터 보호 규제 및 개인 정보 요구에 따라 LLMs(대형 언어 모델)의 유효한 unlearning 알고리즘에 대한 필요성이 커지고 있습니다. 이 논문에서는 LUME(LLM Unlearning with Multitask Evaluations)라는 새로운 멀티태스크 unlearning 벤치마크를 개발하여 창작물과 민감한 정보를 포함하는 데이터 세트를 다룹니다. 다음으로, 1B 및 7B 매개변수를 가진 두 개의 세부 튜닝된 모델을 공개하여 unlearning 알고리즘의 효율성을 평가합니다.

- **Technical Details**: LUME 벤치마크에는 세 가지 작업이 포함됩니다: (1) 합성 창작 단편 소설, (2) 개인 정보가 포함된 합성 전기, (3) 공개 전기. unlearning의 효과성을 측정하기 위해 기억화, 개인 정보 유출, 모델 유틸리티 테스트와 같은 정교한 메트릭을 사용합니다. 이 방법은 프라이버시와 유틸리티를 모두 고려하면서 특정 정보 세트를 기억하지 못하도록 하는 것을 목표로 합니다.

- **Performance Highlights**: 여러 최신 unlearning 알고리즘을 평가한 결과, 민감한 정보를 효과적으로 제거하더라도 모델의 유틸리티가 크게 저하되는 경향이 있음을 발견했습니다. 제안된 벤치마크는 공개적으로 사용 가능하며, 관련 데이터 및 정보에 대한 유효한 unlearning 솔루션 개발에 기여할 것으로 기대됩니다. LUME는 LLM의 유용성을 테스트하기 위한 중요한 자원으로 자리 잡을 것으로 보입니다.



### Judging It, Washing It: Scoring and Greenwashing Corporate Climate Disclosures using Large Language Models (https://arxiv.org/abs/2502.15094)
Comments:
          16 pages, 12 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 활용하여 기업의 기후 관련 공시를 평가하고, 자가 청정화를 위한 방법을 분석합니다. 새로운 점은 LLM-as-a-Judge (LLMJ) 방법론을 통해 기업의 배출 감소 목표와 진행 상황에 대한 보고서 평가에서 효과적인 두 가지 점수를 도출했다는 것입니다. 또한, 기업이 LLM을 사용하여 잘못된 정보 없이도 공시를 향상시킬 수 있는 방법을 조사했습니다. LLM의 응답을 평가하기 위한 점수 시스템의 견고성을 검증한 점도 주목할 만합니다.

- **Technical Details**: 연구에서는 1,410개의 CDP 보고서를 기반으로 LLMJ 방법론을 적용하였습니다. LLMJ의 두 가지 점수 시스템, 즉 참조 기반의 숫자 평가(numercial rating)와 쌍 비교(pairwise comparison)를 통해 높은 성과를 낸 기업과 그 외의 기업을 구분하는 데 효과적임을 확인했습니다. 본 연구에서는 OpenAI의 GPT-4o-mini-2024-07-18를 사용하여 각 문서에 대한 정확성, 구체성, 완전성 및 명료성을 기준으로 평가했습니다. 이 과정에서 LLM이 작성한 응답이 공정하게 점수화될 수 있도록 비관련 정보나 응답 길이에 대한 고려를 배제했습니다.

- **Performance Highlights**: LLMJ의 결과로, 쌍 비교 시스템이 특히 LLM으로 청정화된 응답에 대해 높은 견고성을 보였습니다. 실제로 LLM은 불확실성을 피하고, 사실무근 정보 없이도 맥락에 맞는 설명을 생성할 수 있음을 보여주었습니다. 이러한 기술적 접근은 기업들이 기후 공시에서 투명성을 높이는 데 기여할 수 있음을 시사합니다. 최종적으로, 이 연구는 기후 변화 대응 분야에서 LLM의 활용 가능성과 함께 청정화 방지 방안을 제시함으로써 기후 공시의 질적 향상을 도모합니다.



### Optimizing Singular Spectrum for Large Language Model Compression (https://arxiv.org/abs/2502.15092)
- **What's New**: 이번 연구에서는 SoCo(Singular spectrum optimization for large language model Compression)라는 새로운 압축 프레임워크를 소개합니다. 이 프레임워크는 기존의 SVD(singular value decomposition) 방법을 개선하여, 데이터 기반으로 분해된 컴포넌트의 중요성을 재조정할 수 있습니다. 이전의 방법들이 단순히 특이값을 중요 점수로 간주했던 것과는 달리, SoCo는 실제 다운스트림(task) 성능에 맞춰 학습합니다.

- **Technical Details**: SoCo는 학습 가능한 대각 행렬을 사용하여 특이 스펙트럼(singular spectrum)의 중요 점수를 할당합니다. 세 단계로 구성된 훈련 과정을 통해 초기 거친 압축에서 세밀한 희소화(sparsification)로 점차적으로 점수를 정제합니다. 이 방식은 모델 압축과 성능 유지를 효과적으로 조화롭게 합니다.

- **Performance Highlights**: 여러 LLMs와 벤치마크에서의 실험 결과는 SoCo가 현재 최첨단의 모델 압축 기법을 초월하는 성능을 보여주었음을 나타냅니다. 특히, SoCo는 고정된 특이값 순서에 의존하기보다는 희소화된 중요 점수에 따라 적응적으로 컴포넌트를 가지치기(prune)합니다. 남은 컴포넌트가 증대된 중요 점수를 바탕으로 가지치기된 부분의 손실을 보완할 수 있습니다.



### Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans (https://arxiv.org/abs/2502.15090)
- **What's New**: 이 연구에서는 현대의 대형 언어 모델(LLMs)이 인간의 표현과 어떻게 잘 일치하는지를 조사하는 새로운 접근 방식을 제시합니다. 활성화 조작(activation steering)에서의 방법을 채택하여 특정 개념(예: '고양이')에 대해 책임이 있는 뉴런을 식별하고 이에 따른 활성화 패턴을 분석합니다.

- **Technical Details**: 연구의 결과는 LLM의 표현이 행동 데이터를 통해 유추된 인간의 표현과 밀접하게 일치함을 보여줍니다. 이 정렬은 이전 연구에서 인간과 모델 간의 정렬에 초점을 맞춘 단어 임베딩(word embeddings)보다 더 뛰어납니다. 또한, LLM이 개념을 해석 가능한 인간 중심의 계층적 관계(e.g., '동물'-'개')로 조직하는 방식을 보여줍니다.

- **Performance Highlights**: 이번 접근 방식은 LLM이 개념을 표현하는 방식을 더 세부적으로 관찰할 수 있게 해줍니다. 연구 결과는 LLM이 인류의 이해를 반영하여 개념을 정리하는 능력이 뛰어난 것을 강조하고 있습니다.



### Is Safety Standard Same for Everyone? User-Specific Safety Evaluation of Large Language Models (https://arxiv.org/abs/2502.15086)
Comments:
          Under review

- **What's New**: 이번 연구는 사용자 맞춤 안전(user-specific safety)이라는 새로운 개념을 도입하고, 이를 평가하기 위한 최초의 벤치마크 데이터셋 U-SafeBench를 제안합니다. 현재 사용되고 있는 LLM(Large Language Model)들이 사용자 맞춤 안전 기준을 고려할 때 안전성을 보장하지 못한다는 결과가 드러났습니다. 이는 LLM의 안전 사용과 관련한 연구에서 간과된 중요한 부분을 해결하기 위한 필요성을 강조합니다.

- **Technical Details**: U-SafeBench는 150명 이상의 사용자 프로필과 1,900개 이상의 실제 사용자 지시를 포함하여 LLM의 사용자 맞춤 안전성을 평가하기 위해 설계되었습니다. 또한, 연구진은 사용자 맞춤 안전성을 평가하기 위한 평가 프로토콜을 개발했으며, 이는 LLM을 사용하는 다양한 실제 사용 사례에서의 안전성을 분석합니다. 여기서 사용자 맞춤 안전이란 일반 사용자의 지시에 대한 응답이 특정 그룹 사용자에게 위험할 수 있는 상황을 반영합니다.

- **Performance Highlights**: 실험 결과, 현재 18개의 LLM은 사용자 맞춤 안전 성능이 평균 18.6%로 상당히 낮다는 것을 보여주었습니다. 이는 일반적인 안전 기준에 비해 매우 낮은 수치로, 사용자 맞춤 안전이 LLM의 실질적인 사용 시 중요한 고려사항임을 나타냅니다. 또한, 연구진은 체인 오브 사고(chains of thought) 접근 방식을 통해 이러한 사용자 맞춤 안전 취약성을 개선할 수 있는 간단하고 효과적인 방법을 제안했습니다.



### Rare Disease Differential Diagnosis with Large Language Models at Scale: From Abdominal Actinomycosis to Wilson's Diseas (https://arxiv.org/abs/2502.15069)
- **What's New**: 이번 논문에서는 RareScale이라는 새로운 시스템을 제안했습니다. 이 시스템은 LLM(대형 언어 모델)과 전문가 시스템을 결합하여 희귀 질환에 대한 진단 정확도를 높이고자 합니다. 이를 통해 의료 환경에서 희귀 질환 감별 진단을 개선하고자 하며, 575개의 희귀 질환을 대상으로 한 결과를 보고합니다.

- **Technical Details**: RareScale은 전문가 시스템과 LLM을 동시에 활용하여 희귀 질환에 대한 대화 시뮬레이션을 진행합니다. 이러한 시뮬레이션 데이터는 희귀 질환 후보 예측 모델 트레이닝에 사용되며, 이 모델에서 생성된 후보들은 블랙박스 LLM에 추가 입력으로 제공됩니다. 이를 기반으로 할 경우, RareScale은 흔한 진단과 희귀 진단 간의 균형을 맞출 수 있도록 설계되었습니다.

- **Performance Highlights**: RareScale의 접근 방식은 기존 블랙박스 LLM의 성능을 17% 이상 향상시켰으며, Top-5 정확도가 56.8%에서 74.1%로 증가했습니다. 이러한 성과는 희귀 질환의 감별 진단 과정에서 LLM과 전문가 지식을 통합함으로써 달성된 결과입니다. 궁극적으로 RareScale은 희귀 질환을 더 효과적으로 식별할 수 있는 가능성을 보여줍니다.



### Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation (https://arxiv.org/abs/2502.15040)
Comments:
          GenAI4Health - AAAI '25

- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 향상을 위해 Visual RAG (V-RAG) 프레임워크를 제안합니다. V-RAG는 텍스트와 시각 데이터 모두를 활용하여 정보의 정확성을 향상시키기 위해 노력합니다. 특히, 의료 분야에 중점을 두어 MIMIC-CXR 및 Multicare 데이터셋에서 성능을 검증하였고, 환자 진단의 정확도를 높이는 데 기여하고 있습니다.

- **Technical Details**: V-RAG는 텍스트 기반 RAG의 한계를 극복하고 이미지를 포함한 다중 데이터를 활용하여 더 정확한 응답을 제공합니다. 이를 위해 오픈북(Open Book) 성능을 강화하며, 제시된 멀티 이미지 처리 기술을 통해 모델의 이해도를 향상시킵니다. 결국, V-RAG는 단일 이미지 데이터로 훈련된 Model이 다중 이미지 데이터 또한 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: V-RAG를 적용한 결과, 의료 이미지 및 X-ray 보고서의 생성에서 hallucination 문제를 보다 효과적으로 해결할 수 있음을 보여주었습니다. 시험 결과, RadGraph-F1 Score의 향상 또한 확인되었으며, 이는 진단의 정확성 및 신뢰성을 높이는 데 기여하고 있습니다. 이런 성과는 빈번한 의료 개체뿐만 아니라 드문 개체에 대해서도 긍정적인 효과를 입증합니다.



### InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback (https://arxiv.org/abs/2502.15027)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문에서는 Large Multimodal Models (LMMs)의 인간 피드백과의 상호작용 지능을 평가하기 위한 새로운 프레임워크인 InterFeedback를 설계하였습니다. 이는 LMM과 데이터셋에 적용 가능하며, InterFeedback-Bench를 통해 다양한 오픈소스 LMM을 평가합니다. 추가적으로, OpenAI-o1 및 Claude-3.5-Sonnet와 같은 주요 모델의 상호작용 성능을 수동으로 테스트하기 위해 새롭게 수집한 120개의 사례를 포함한 InterFeedback-Human 데이터셋도 제공됩니다.

- **Technical Details**: InterFeedback 프레임워크는 LMM이 상호작용적으로 문제를 해결할 수 있도록 설계되었으며, 문제 해결 능력과 피드백 해석 능력을 평가하는 기준점인 InterFeedback-Bench를 도입합니다. 이 연구에서는 MMMU-Pro와 MathVerse라는 두 가지 도전적인 데이터셋을 활용하여 다양한 LMM의 상호작용 지능을 평가하였습니다. 최종적으로, LMM이 피드백을 통해 성능을 개선할 수 있는지에 대한 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, OpenAI-o1 등 최첨단 LMM조차도 인간의 피드백을 통해 50% 미만으로 결과를 수정하는 데 그쳤습니다. LMM은 높은 품질의 피드백을 요구하며, 저품질 피드백은 성능을 오히려 저하시킬 수 있다는 사실을 발견하였습니다. 이러한 결과는 LMM의 피드백 해석 및 반영 능력을 향상시킬 필요성을 강조합니다.



### A Meta-Evaluation of Style and Attribute Transfer Metrics (https://arxiv.org/abs/2502.15022)
- **What's New**: 이 논문은 스타일 및 속성 전이에 대한 평가 메트릭이 콘텐츠 보존(content preservation) 중심으로 대규모 연구를 수행한 점이 새롭습니다. 기존의 스타일 전이 평가가 스타일 강도(style strength), 콘텐츠 보존(Content Preservation), 그리고 유창성(fluency)이라는 세 가지 차원으로 나뉘어 있는 반면, 이 연구는 보다 체계적이고 효율적인 평가 기준을 제안합니다.

- **Technical Details**: 이 연구에서는 다양한 스타일 전이에 대한 콘텐츠 보존 평가를 위해 8개의 평가 메트릭을 벤치마킹(benchmark)하고, 스타일 변화에 잘 정렬된 새로운 테스트 세트를 구축하였습니다. 특히 스타일 전이를 다룰 때 콘텐츠 보존 메트릭이 스타일 변화에 조건부로 설정되어야 한다고 주장하며, 스타일 변화에 따라 다음 토큰의 가능성(likelihood)을 활용한 제로샷 평가(zero-shot evaluation) 방법을 제안합니다.

- **Performance Highlights**: 연구 결과, 기존의 유사도 메트릭들은 스타일 변화가 클수록 콘텐츠 보존을 낮게 평가하는 경향이 있음을 밝혀냈습니다. 또한, 기존의 메트릭들은 인간 평가와의 상관관계를 세밀하게 고려하지 않아, 실제 적합한 평가를 제공하지 못하는 문제가 있음을 지적합니다. 이 논문은 콘텐츠 보존 메트릭의 평가 방법론에 대한 새로운 관점을 제공하며, 향후 스타일 전이 방법을 공정하게 평가할 수 있도록 기여할 것으로 기대됩니다.



### Using tournaments to calculate AUROC for zero-shot classification with LLMs (https://arxiv.org/abs/2502.15018)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)이 수행하는 제로샷 분류(zero-shot classification) 작업을 쌍 비교(pairwise comparison) 작업으로 변환하여 상대적인 순위를 얻는 방법을 제안합니다. 이를 통해 기존의 분류 작업에서 의사 결정 경계(decision boundary)를 유연하게 조정할 수 없는 문제를 해결하려고 합니다. 이 접근법은 Elo 등급 시스템을 사용하여 데이터셋 내 개체(instance)의 점수를 매기고, 궁극적으로 신뢰도 순위를 도출하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 초기 Elo 등급을 통해 데이터셋의 모든 개체에 대해 순위를 매기고, 쌍 비교를 통해 이 순위를 지속적으로 업데이트하는 과정을 포함합니다. 이 과정은 매 라운드마다 LLM 호출을 수반하며, 수집된 결과는 ROC(Receiver-Operating Characteristic) 또는 PR(Precision-Recall) 곡선을 생성하는 데 사용됩니다. 또한, 효율성을 위해 무작위, 그래프 기반, 스위스 시스템 등의 다양한 스케줄링 알고리즘을 평가합니다.

- **Performance Highlights**: 이 연구에서 제안된 방법은 CoLA와 CliniFact라는 두 개의 다양한 데이터셋에서 F1 점수 측면에서 향상된 분류 성능을 보여줍니다. 특히 무작위 스케줄링이 예상 외로 효과적이었으며, 스위스 시스템과 새로운 그래프 기반 방법도 대부분의 설정에서 우수한 성능을 보였습니다. 이러한 결과는 제로샷 LLM의 적용 가능성을 넓히는 데 기여할 것으로 기대됩니다.



### Obliviate: Efficient Unmemorization for Protecting Intellectual Property in Large Language Models (https://arxiv.org/abs/2502.15010)
- **What's New**: 최근 AI 기업들과 콘텐츠 제작자 사이의 저작권 협약으로 인해 언어 모델의 저작권 콘텐츠 재현 능력에 대한 정밀한 제어 필요성이 강조되었습니다. 우리는 Obliviate라는 새로운 포스트 트레이닝(post-training) 기법을 제안하며, 이 방법은 특정 텍스트의 정확한 복제를 선택적으로 방지하면서 의미론적 이해를 유지합니다. Obliviate는 기억된 시퀀스에서 특정 토큰을 선택하고 모델의 확률 분포를 수정하여 정확한 복제를 방지하는 방식으로 작동합니다.

- **Technical Details**: Obliviate는 사전 훈련된 언어 모델에서 특정 시퀀스를 비기억하도록 선택적으로 수정하는 기술입니다. 이 기술은 Kullback-Leibler (KL) 발산 손실을 사용하여 목표 토큰이 재현될 확률을 줄이면서 유창성과 일관성을 유지하도록 조정합니다. 비목표 토큰에 대해서도 KL 발산을 적용하여 상위 k 토큰 분포의 일관성을 유지하여 모델의 전반적인 성능을 보존합니다.

- **Performance Highlights**: Obliviate는 여러 대형 언어 모델(LLaMA-3.1 8B, Qwen-2.5-7B 등)에 대해 평가된 결과, 100배의 저작권 콘텐츠의 Verbatim memorization 감소를 보여주면서도 표준 벤치마크에서 성능 저하가 1% 이내로 유지되었습니다. 이 결과는 Obliviate가 선택적으로 목표 콘텐츠를 비기억하면서 모델 성능을 유지하는 데 효과적임을 강조합니다.



### Contextualizing Search Queries In-Context Learning for Conversational Rewriting with LLMs (https://arxiv.org/abs/2502.15009)
- **What's New**: 본 논문에서는 대화형 쿼리 리라이트(conversational query rewriting)의 필요성을 강조하며, 기존의 감독 학습 방법들이 제한된 데이터 환경에서 문제가 발생함을 지적합니다. 이를 해결하기 위해 Prompt-Guided In-Context Learning이라는 새로운 접근 방식을 제안하고, 이 방법이 현재의 강력한 모델들을 초월할 수 있음을 실험을 통해 입증합니다.

- **Technical Details**: 이 방법은 Large Language Models (LLMs)의 in-context learning 능력을 활용하여, 최소한의 레이블 데이터로도 대화형 쿼리를 효과적으로 재작성하도록 설계된 프롬프트를 사용합니다. 제안된 접근 방식은 입력 및 출력 형식, 작업 설명 및 몇 가지 예시를 포함한 프롬프트를 사용하여, 훈련 완료된 LLM이 명시적 미세 조정 없이도 컨텍스트가 독립적인 쿼리를 생성할 수 있도록 합니다.

- **Performance Highlights**: TREC와 Taskmaster-1 데이터셋을 통해 진행된 광범위한 실험 결과, 이 방법은 BLEU, ROUGE-L, 성공률(Success Rate), MRR(most-relavant retrieval) 등 다양한 평가 지표에서 기존의 감독 모델 및 대조적 공동 훈련 방법을 능가하는 것으로 나타났습니다. 인지 예시의 중요성을 확인하는 탈락 연구(ablation study)를 통해, 제안된 방법의 효과성을 뒷받침하는 결과를 도출하였습니다.



### LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers (https://arxiv.org/abs/2502.15007)
Comments:
          accepted to NAACL 2025

- **What's New**: 이번 논문에서는 Large Language Models(LLMs)가 맥락 정보를 인코딩하고 저장하는 방법을 정량화하는 새로운 방법을 소개합니다. 연구팀은 종종 사소하게 여겨지는 토큰들, 예를 들어 정관사, 구두점 등이 비정상적으로 높은 맥락 정보를 지닌다는 사실을 발견했습니다. 이러한 발견은 LLM에서 덜 중요하게 보이는 토큰들이 맥락 유지를 위해 얼마나 중요한지를 강조하고 있습니다.

- **Technical Details**: LLM-Microscope라는 오픈소스 툴킷은 토큰 수준의 비선형성(nonlinearity), 맥락 기억(contextual memory), 중간 층의 기여도 분석을 시각화할 수 있는 기능을 제공합니다. 이 프레임워크는 LLM의 내부 행동을 분석하기 위한 종합적인 방법론을 제공하며, 각 층의 임베딩 간의 비선형 전환을 단일 선형 매핑으로 근사할 수 있는 정도를 정량화합니다. 연구의 주요 분석 방법론으로는 맥락화 평가(contextualization assessment)와 토큰 수준 비선형성 측정이 포함됩니다.

- **Performance Highlights**: 연구진은 맥락 정보가 포함된 토큰을 제거하면 MMLU와 BABILong-4k와 같은 특정 작업에서 성능이 일관되게 저하된다는 점을 강조했습니다. 이러한 성능 저하는 강력한 언어 모델인 GPT-4o에 의해 불필요하다고 판단된 토큰들을 제거할 때도 지속적으로 나타났습니다. LLM-Microscope는 연구자들이 다양한 언어 모델의 내부 구조를 쉽게 분석할 수 있게 해줍니다.



### A Socratic RAG Approach to Connect Natural Language Queries on Research Topics with Knowledge Organization Systems (https://arxiv.org/abs/2502.15005)
Comments:
          6 pages, 2 figures, AAAI 2025 Workshop on A Translational Institute for Knowledge Axiomatization

- **What's New**: 이 논문에서는 자연어 질의(Natural Language Queries)를 연구 주제와 기계가 해석할 수 있는 의미적 엔티티(Semantic Entities)로 매핑하는 Retrieval Augmented Generation (RAG) 에이전트를 제안합니다. 이 접근법은 Socratic Dialogue와 RAG를 결합하여 사용자의 직관적인 연구 주제 이해를 구조화된 Knowledge Organization Systems (KOS)와 일치시킵니다. 제안된 방법은 복잡한 학술 분류를 더 쉽게 접근할 수 있도록 합니다.

- **Technical Details**: 논문에서는 기존 연구 주제를 정리하는 문제와 관련된 연구들을 간단히 개관한 후, CollabNext의 간단한 설명을 제공합니다. CollabNext는 사람들이 어느 주제에서 활동하는지를 탐색할 수 있도록 설계된 지식 그래프입니다. 이 애플리케이션은 공개적으로 접근할 수 있는 강력하고 구조화된 데이터에 의존하며, 연구 주제에서 사람과 조직 사이의 관계를 맺고 있습니다.

- **Performance Highlights**: CollabNext는 HBCUs(Historically Black Colleges and Universities)와 신진 연구자들의 가시성을 높이기 위해 의도적으로 설계되었습니다. 이 프로젝트는 연구 분류에서 발생하는 혼잡함과 이름 충돌 문제를 해결하기 위해 기존의 공개적인 의미적 구조를 활용합니다. 다양한 연구 영역에 대한 인사이트를 제공하여, 잠재적으로 넓은 범위로 개인 및 연구 공동체에 기여할 수 있는 가능성이 있습니다.



### Beyond No: Quantifying AI Over-Refusal and Emotional Attachment Boundaries (https://arxiv.org/abs/2502.14975)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 감정 경계 처리 능력을 평가하기 위한 오픈 소스 벤치마크와 평가 프레임워크를 제시합니다. 6개 언어로 구성된 1156개의 프롬프트를 사용하여 GPT-4o, Claude-3.5 Sonnet, Mistral-large의 세 가지 선두 모델의 성능을 분석하였습니다. 이를 통해 각 모델이 어떻게 감정 경계를 유지하는지를 정량적으로 평가하였습니다.

- **Technical Details**: 이 프레임워크는 직접 거부(direct refusal), 사과(apology), 설명(explanation), 돌리기(deflection), 인식(acknowledgment), 경계 설정(boundary setting), 감정 인식(emotional awareness) 등 7개의 주요 패턴을 기준으로 응답을 정량화합니다. 연구 결과, Claude-3.5가 최고 점수인 8.69/10을 기록하였으며, 평균 86.51 단어의 긴 응답을 생성하는 것을 확인했습니다. 또한 영어와 비영어 응답 간 성능의 격차가 상당함을 발견했습니다.

- **Performance Highlights**: 영어의 평균 점수는 25.62인 반면, 비영어 상호작용의 평균 점수는 0.22 미만으로 극심한 차이를 보였습니다. 영어 응답의 거부율은 43.20%로 비영어의 1% 미만에 비해 현저히 높았습니다. Mistral 모델은 돌리기 전략을 선호하는 경향을 보였으며, 모든 모델에서 공감 점수가 지속적으로 낮았습니다. 향후 연구에서는 보다 정교한 점수 산출 방법과 언어 범위 확대, 문화적 차이를 탐구하는 방향으로 나아가야 할 것입니다.



### Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding (https://arxiv.org/abs/2502.14969)
- **What's New**: 이 논문은 다양한 자연어 출력 형식을 요구하는 작업에 대해 일반-purpose language models (언어 모델)의 구조화된 출력을 탐구합니다. 연구팀은 다섯 가지 토큰 포맷(token formats)과 네 가지 NLP 벤치마크를 사용하여 서로 다른 모델들이 어떻게 다르게 성능을 발휘하는지를 분석했습니다. 고유한 결과로, 숫자 형식에서 가장 높은 정확성을 기록하며, 선행 공백을 포함하는 토큰을 사용하는 것이 성능을 5%-10% 향상시킬 수 있다고 밝혔습니다.

- **Technical Details**: 연구에서는 네 가지 사전 훈련된 언어 모델 가족을 다섯 가지 토큰 세트를 비교하여 grammar-constrained decoding (문법 제한 해독)에서의 성능을 평가했습니다. 기존 방법론이 그러하듯이, GCD를 통해 특정 문법에 따라 토큰을 선택적으로 샘플링하여 구조화된 출력을 보장하는 것이 주된 초점입니다. 또한, ill-formed subword token representations (형식이 잘못된 하위 단어 토큰 표현)이 이들 토큰 간의 선호도에 영향을 미친다는 가설을 제시하였습니다.

- **Performance Highlights**: 모든 모델이 [0,1] 구간 내의 숫자만을 생성하는 지시를 받을 때 가장 높은 성능을 발휘했습니다. 특히, 작은 모델에서는 포맷 기반 차이가 두드러지며, 선행 공백을 포함한 토큰 사용이 성능 향상에 기여합니다. 연구 결과는 일반적인 언어 모델을 사용하는 사용자들이 더 나은 성능을 확보하기 위한 구체적인 전략을 구현하는 데 유용할 것입니다.



### Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection (https://arxiv.org/abs/2502.14932)
- **What's New**: 이 논문은 지식 그래프(knowledge graph)와 대형 언어 모델(large language model, LLM)의 통합을 통해 추론 과정을 향상시키는 방법을 제시합니다. 특히, 기존 방법들이 이진 판단을 사용해 정보를 처리하는 한계를 극복하기 위해 Active self-Reflection 프레임워크를 도입하여, 구조화된 그래프에 기반한 반복 추론(end-to-end training approach)을 가능하게 합니다. 이는 LLMs가 구조화된 지식을 이해하는 방식을 새롭게 제시합니다.

- **Technical Details**: 프레임워크 내에서, 모델은 특수 토큰(special tokens)을 활용해 지식 탐색의 필요성을 능동적으로 결정하고, 검색된 지식에 대해 반영(reflective)하는 비판을 수행합니다. ArG 프레임워크에 의해 반복적인 추론 과정이 진행되며, 모델은 각 단계에서 검색이 필요한지를 평가하고, 검색된 지식의 관련성을 평가해 합리성 점수를 부여합니다. 결과적으로, 이 과정은 추론 트리(reasoning tree)로 표현되며, 각 단계에서 후보가 생성되고, 이들을 기반으로 내려가는 경로(path)가 확장됩니다.

- **Performance Highlights**: 제안된 모델은 기존 지식 그래프 추론 작업에서 우수한 성능을 보이며, 더 나아가 높은 해석 가능성(interpretability)을 제공합니다. 이는 모델이 구조화된 지식에 대한 이해도를 깊이 탐구할 수 있게 하며, 기존의 방법들과 비교해 더욱 뛰어난 결과를 도출해냅니다. 이 연구는 기존 방법론의 한계를 극복하는 데 중요한 기여를 하며, 지식 그래프와 LLM 통합의 새로운 가능성을 보여줍니다.



### A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language? (https://arxiv.org/abs/2502.14924)
- **What's New**: 이 연구는 언어의 정보-이론적 복잡성과 대형 언어 모델(LLM)에서의 프랙탈 구조를 비교하여 이러한 모델들이 자연 언어의 프랙탈 특성을 얼마나 잘 재현하는지를 조사합니다. 특히, 프랙탈 매개변수가 LLM의 출력에서 광범위하게 변동하는 반면, 자연 언어에서는 좁은 범위에 속한다는 점을 강조합니다. LLM이 특정한 조건(예: 온도 설정과 프롬프트 방법)에서 이러한 특성을 구현하지 못할 수 있다는 점도 발견하였습니다.

- **Technical Details**: 언어는 자기 유사성(self-similarity)과 장기 의존성(long-range dependence, LRD)을 따라 구조화되어 있으며, 이러한 특성은 Hurst 지수와 Hölder 지수로 정량화됩니다. LLM이 자연 언어의 특성을 모방하기 위해서는 토큰 수준에서 잘 보정되어야 하지만, 추론 과정에서의 오류는 격렬한 프랙탈 구조를 왜곡할 수 있습니다. 연구는 240,000개 이상의 LLM 생성 문서를 포함한 데이터셋을 구축하여, 다양한 프롬프트와 온도 설정에 따른 LLM의 출력을 분석했습니다.

- **Performance Highlights**: 이 연구에서는 LLM의 프랙탈 특성이 자연 언어보다 확연히 다르다는 것을 보여주었습니다. 특히, 프랙탈 매개변수를 통해 LLM 생성 텍스트를 식별하는 데 유용할 수 있으며, 주요 모델 구조에도 이러한 결과가 보편적으로 적용된다는 점이 중요한 발견입니다. 결과적으로 이 연구는 LLM의 합성 텍스트 생성을 평가하고 탐지하는 데 필요한 통찰을 제공합니다.



### AI Thinking as a Meaning-Centered Framework: Reimagining Language Technologies Through Community Agency (https://arxiv.org/abs/2502.14923)
Comments:
          LT4All 2025. Language Technologies for All - 2025. Advancing Humanism through Language Technologies. Paris (FR), UNESCO Headquarters, 24-26 February 2025

- **What's New**: 이번 논문은 기술 발전이 지역사회와 함께 이루어져야 한다는 새로운 방안을 제시합니다. AI Thinking 프레임워크를 통해 언어와 문화의 보존을 중심으로 기술 개발을 진행할 수 있도록 합니다. 이는 모든 커뮤니티의 언어적 지식 표현에서 지역사회의 통제를 유지할 수 있는 다층적 기술 생태계를 설계하는 것을 포함합니다.

- **Technical Details**: 논문에서 제안하는 AI Thinking 프레임워크는 언어 기술 개발의 중심에 의미를 두고, 인지적, 사회적, 문화적 차원 간의 복잡한 상호작용을 유지합니다. 이 구조는 커뮤니티 지식 시스템과의 통합을 지원하며, 기존 언어 기술의 한계를 극복하는 것을 목표로 합니다. 또한, 기존 모델들이 필요로 하는 방대한 디지털 언어 데이터 부족 문제를 해결하는 동시에, 문화적 맥락을 유지하는 방법론을 제시합니다.

- **Performance Highlights**: AI Thinking 프레임워크는 언어 기술의 통합적인 접근을 통해 지역사회의 기관들을 강화하며, 참여 및 통제 권한을 보장하는 것을 목표로 합니다. 이 논문에서는 현대 기술이 문화적 관점에서 어떻게 의미를 보존할 수 있는지를 논의하며, 향후 연구 방향 및 기술적 요건을 다루고 있습니다. 이러한 접근은 언어의 복잡성을 잃지 않으면서도 현대의 기술적 도전에 대응할 수 있는 가능성을 보여줍니다.



### SIFT: Grounding LLM Reasoning in Contexts via Stickers (https://arxiv.org/abs/2502.14922)
- **What's New**: 이 논문은 대형 언어 모델(LLM)들이 맥락을 잘못 해석하는 문제가 reasoning 과정에서 중요한 이슈가 될 수 있음을 밝힙니다. 특히, "10 dollars per kilo"에서 LLM이 'per'를 '각각'이라는 의미로 이해하지 못해 계산 오류를 발생할 수 있음을 예로 듭니다. 이를 해결하기 위해 새로운 포스트 트레이닝 접근법인 **Stick to the Facts (SIFT)**를 제안하며, 이는 LLM의 reasoning을 맥락에 기반하여 보다 정확하게 수행할 수 있도록 돕습니다.

- **Technical Details**: SIFT의 핵심 요소는 *Sticker*로, 모델 자체에 의해 생성되어 맥락 내 핵심 정보를 강조합니다. SIFT는 입력 쿼리를 통해 핵심 사실을 요약하고, Sticker 기반의 두 가지 예측을 생성합니다. 만약 두 예측이 다르다면, Sticker는 *forward* 최적화 및 *inverse* 생성 과정을 통해 점진적으로 refinment됩니다. 이는 보다 신뢰할 수 있는 reasoning 결과를 도출하기 위한 방법입니다.

- **Performance Highlights**: SIFT는 다양한 모델과 벤치마크에서 일관된 성과 개선을 보였습니다. 예를 들어 DeepSeek-R1 모델의 경우 AIME2024에서 pass@1 정확도를 78.33%에서 **85.67%**로 향상시켜 오픈 소스 커뮤니티에서 새로운 최첨단 결과를 수립했습니다. 또한, Llama3.2-3B-Instruct와 같은 소형 모델에서도 약 1.03%에서 7.34%의 정확도 향상을 보여주며, SIFT의 효과를 입증합니다.



### The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Tex (https://arxiv.org/abs/2502.14921)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)로 생성된 합성 데이터(synthetic data)로부터 훈련 샘플(training samples)의 정보가 얼마나 유출될 수 있는지를 분석합니다. 합성 데이터 생성 파이프라인에서 정보 흐름(information flow)의 미세한 부분을 간과할 경우 사생활(privacy)에 대한 잘못된 인식을 초래할 수 있습니다. 저자들은 특히 적대자가 미세 조정된 모델(fine-tuned model)에 접근하지 못하고 합성 데이터만 가지고 있는 경우에 대한 멤버십 추론 공격(membership inference attacks, MIAs)을 설계했습니다.

- **Technical Details**: 이 논문에서는 합성 데이터에 대한 MIAs가 무작위 추측(random guess)보다 훨씬 더 높은 성능을 보인다는 사실을 발견했습니다. 이는 합성 데이터가 훈련 데이터(training data)에 대한 정보를 유출하는 것을 의미합니다. 또한, 모델 기반 MIA에 대한 취약성을 극대화하기 위해 제작된 카나리(canaries)는 합성 데이터만 공개될 경우 프라이버시 감사를 위한 최적의 방법이 아님을 보였습니다.

- **Performance Highlights**: 저자들은 자동 회귀(autoregressive) 모델의 메커니즘을 활용하여 합성 데이터에서 탐지 가능한 흔적을 남기는 in-distribution prefix와 high-perplexity suffix를 가진 카나리를 설계했습니다. 이러한 접근 방식은 데이터 기반 MIAs의 효과를 향상시키고 LLMs가 생성한 합성 데이터의 프라이버시 위험을 보다 잘 평가할 수 있도록 합니다.



### MKE-Coder: Multi-Axial Knowledge with Evidence Verification in ICD Coding for Chinese EMRs (https://arxiv.org/abs/2502.14916)
- **What's New**: 이번 논문은 중국 전자 의료 기록(EMR)에서 국제 질병 분류(International Classification of Diseases, ICD)를 자동 코딩하기 위한 새로운 프레임워크인 MKE-Coder를 소개합니다. 이 프레임워크는 주로 질병 기반의 다축 지식(Multi-axial Knowledge)을 통합하고, 신뢰할 수 있는 임상 증거(Clinical Evidence)를 검증하여 코드의 유효성을 보장합니다. MKE-Coder의 도입은 기존 방법들이 중국 EMR의 특수성을 고려하지 못한 문제를 해결하는 데 기여하고 있습니다.

- **Technical Details**: MKE-Coder는 다축 지식을 활용하여 ICD 코드를 생성하는 과정에서, 먼저 진단 후보 코드를 추출하고, 각 코드에 대해 필수 지식을 분류합니다. 다음으로, EMR의 방대한 내용을 활용해 임상 증거를 수집하고, 유효한 증거를 스코어링 모델을 통해 필터링합니다. 마지막으로, 마스크 언어 모델링 전략을 기반으로 한 추론 모듈을 구현하여 후보 코드와 관련된 모든 지식이 증거에 의해 지원되는지 검토합니다.

- **Performance Highlights**: 실험 결과, MKE-Coder는 중국 EMR 기반의 ICD 자동 코딩 작업에서 중요한 개선 사항을 보여주었습니다. 특히, 실제 코딩 시나리오에서 이 방법을 평가한 결과, 코더들이 코딩 정확성과 속도를 크게 향상할 수 있었음을 입증하였습니다. 이러한 성취는 MKE-Coder가 기존의 영어 EMR 위주의 방법론과 차별화된 접근을 취하고 있음을 나타냅니다.



### OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignmen (https://arxiv.org/abs/2502.14913)
Comments:
          15 pages

- **What's New**: 최근 다중 에이전트 협업을 이용한 대형 언어 모델(Large Language Models, LLMs)이 Text-to-SQL 작업에서 눈에 띄는 성과를 올리고 있는 가운데, OpenSearch-SQL이라는 새로운 방법론이 제안되었습니다. 이 방법론은 전체 텍스트-투-SQL 작업을 네 가지 주요 모듈인 전처리(Preprocessing), 추출(Extraction), 생성(Generation), 개선(Refinement) 및 일관성 정렬(Alignment) 모듈로 나누어 운영합니다. 특히, 이 구조는 에이전트의 입력과 출력을 정렬하여 지침을 따르지 못하거나 해리(hallucination) 문제를 줄이는 데 기여합니다.

- **Technical Details**: OpenSearch-SQL에서는 텍스트-투-SQL 작업을 인간의 SQL 작성 프로세스를 모델링한 표준 프로세스에 기반하여 정의합니다. 이 과정은 전처리, 추출, 생성 및 개선의 네 가지 단계로 구성되어 있으며, 요청을 이해하고 SQL을 완성하는 데 필요한 정보를 체계적으로 처리합니다. 또한, SQL-Like라는 중간 언어를 개발하여 모델이 SQL의 구조를 효율적으로 생성하도록 돕고, 자가 학습 기반의 Query-CoT-SQL 형태의 동적 몇 샷(few-shot) 전략을 설계하였습니다.

- **Performance Highlights**: OpenSearch-SQL의 실험 결과, BIRD 개발 세트에서 69.3%의 실행 정확도(EX)를 달성하며, 테스트 세트에서는 72.28%의 성능을 기록하였습니다. 또한, 보상 기반 유효성 점수(R-VES)는 69.36%로, 제출 시 모든 지표에서 1위를 기록하였습니다. 이러한 결과는 제안된 방법이 효과성과 효율성 모두에서 상당한 장점을 가지고 있음을 잘 보여줍니다.



### Universal Semantic Embeddings of Chemical Elements for Enhanced Materials Inference and Discovery (https://arxiv.org/abs/2502.14912)
Comments:
          5 figures

- **What's New**: 이번 논문에서는 재료 추론(material inference) 및 발견(discovery)을 촉진하기 위한 보편적인 의미 임베딩(semantic embeddings) 프레임워크를 제시합니다. 이 프레임워크는 합금(alloy) 관련 과학 논문 129만 개의 초록을 기반으로 학습된 BERT 기반 자연어 처리(NLP) 모델인 ElementBERT를 활용하여 합금에 특화된 잠재 지식(latent knowledge)과 맥락적 관계를 포착합니다.

- **Technical Details**: ElementBERT는 전통적인 경험적(descriptors) 대신 사용할 수 있는 강력한 원소 기술(descriptors)로, 여러 하위 작업(downstream tasks)에서 현저한 성능 향상을 보여줍니다. 이 연구에서는 기계적 성질(mechanical properties) 예측, 상 구조(classifying phase structures) 분류, 베이지안 최적화(Bayesian optimization)를 통한 재료 성질 최적화 등의 작업에서 성능을 평가하였습니다.

- **Performance Highlights**: 타이타늄 합금(titanium alloys), 고엔트로피 합금(high-entropy alloys), 형태 기억 합금(shape memory alloys)에 대한 적용을 통해 예측 정확성(prediction accuracy)에서 최대 23% 향상이 이루어졌습니다. 또한, ElementBERT는 일반용 BERT 변형보다 특화된 합금 지식을 인코딩하여 더 나은 성능을 발휘함을 보여 줍니다.



### Batayan: A Filipino NLP benchmark for evaluating Large Language Models (https://arxiv.org/abs/2502.14911)
Comments:
          Submitted to ACL 2025

- **What's New**: 최근 큰 언어 모델(LLMs)의 발전에도 불구하고, 자원이 부족한 언어의 언어적 뉘앙스는 여전히 탐구되지 않고 있습니다. 본 논문에서는 필리핀어를 평가하기 위해 Batayan이라는 포괄적인 벤치마크를 소개하며, 이는 자연어 처리(NLP)의 세 가지 주요 역량인 이해(understanding), 추론(reasoning), 생성(generation)을 평가하기 위해 설계되었습니다. Batayan은 타갈로그(Tagalog) 및 코드 스위칭된 타글리시(Taglish) 발화를 포괄하는 여덟 가지 과제를 통합하였습니다.

- **Technical Details**: Batayan은 제공되는 데이터셋의 질과 일관성을 보장하기 위해 원어민에 의해 세심하게 주석 처리된 점에서 다른 필리핀어 데이터셋과 차별화됩니다. 연구에서는 여러 다국어 LLM을 대상으로 한 평가 결과를 보고하며, 필리핀어에서 발견된 성능 차이를 강조하고 있습니다. 또한, 필리핀어의 복잡한 형태론(morphology)과 문법 구조를 모델링할 때의 고유한 도전 과제를 다루고 있습니다.

- **Performance Highlights**: 저자는 Batayan에서의 엄격한 평가 결과를 통해 필리핀어가 다국어 LLM에서 적게 대표되고 있음을 나타내는 중요한 성능 격차를 강조하고 있습니다. 이를 통해 필리핀어와 같은 저대표 언어에 대한 명확한 지원과 교육 튜닝의 필요성을 강조하고 있으며, 향후 필리핀어 자원의 발전을 위한 실질적인 프레임워크를 제공합니다.



### EvoP: Robust LLM Inference via Evolutionary Pruning (https://arxiv.org/abs/2502.14910)
- **What's New**: 이번 논문에서는 EvoP라는 진화적 구조 절단 프레임워크를 제안합니다. EvoP는 군집 기반 보정 데이터셋 샘플링(Cluster-based Calibration Dataset Sampling) 전략을 통해 보다 다양한 보정 데이터셋을 생성하고, 진화적 절단 패턴 탐색(Evolutionary Pruning Pattern Searching) 방법을 통해 최적의 절단 패턴을 찾습니다. 기존 기술들과는 달리 EvoP는 최적의 성능과 효율성을 동시에 달성하는 것을 목표로 하고 있습니다.

- **Technical Details**: EvoP는 대규모 언어 모델(LLM)의 구조 절단을 개선하기 위한 새로운 접근 방식을 제공합니다. 이는 모델의 크기를 줄이는 것뿐만 아니라, 수행 성능도 극대화하기 위해 설계되었습니다. 네트워크 절단 문제는 사전 훈련된 모델과 보정 데이터셋을 바탕으로 최적의 절단 패턴을 찾는 것을 목표로 하며, 사전 훈련된 모델의 매개변수에 기반하여 절단 패턴 공간에서 최적 솔루션을 찾습니다.

- **Performance Highlights**: EvoP는 다양한 LLM 및 다운스트림 작업에서 기존 최첨단 절단 방법을 초과하는 성능을 보였습니다. 이를 통해 다양한 데이터셋에서도 높은 일반화 능력을 입증하였습니다. 연구 결과는 EvoP가 실제 세계 응용에서 LLM을 효율적으로 배포하는 실용적이고 확장 가능한 솔루션임을 보여줍니다.



### GneissWeb: Preparing High Quality Data for LLMs at Sca (https://arxiv.org/abs/2502.14907)
- **What's New**: 본 논문에서는 약 10조 토큰(token) 규모의 대규모 데이터셋 GneissWeb을 소개하고, 이 데이터셋이 LLM(대규모 언어 모델) 훈련에 필요한 데이터 품질과 양의 요구를 충족한다고 주장합니다. 특히, GneissWeb은 고품질의 데이터를 제공하며, 이는 모델의 일반화 능력을 향상시키는 데 중요한 역할을 합니다. 이 논문에서 제시된 방법론은 정확한 부분 문자열 제거와 잘 구축된 품질 필터의 조합으로 구성되어 있습니다. 이는 기존의 공공 데이터셋들이 5조 토큰 이하에 그치고 있는 문제를 해결하고자 합니다.

- **Technical Details**: GneissWeb의 데이터 생성 과정에는 샤딩된 정확한 부분 문자열 중복 제거 및 복합적인 품질 필터 조합이 포함됩니다. 새로운 품질 필터링 기법인 'Extreme Tokenized Documents Removal'를 도입하여, 모델 훈련에 사용될 토큰화된 데이터를 기반으로 저품질 문서를 효과적으로 걸러냅니다. 또한, 인간의 독서 능력을 활용한 가독성 스코어 필터를 통해 다양한 도메인의 저품질 문서를 제외할 수 있도록 합니다. 이 모든 절차는 공개적으로 제공되는 IBM 데이터 준비 키트를 사용하여 효율적으로 실행되었습니다.

- **Performance Highlights**: GneissWeb으로 훈련된 모델은 FineWeb-V1.1.0으로 훈련된 모델에 비해 11개 벤치마크에서 평균 2.73% 높은 성능을 보였습니다. 벤치마크 수를 20개로 확장하였을 때도, GneissWeb으로 훈련된 모델이 1.75%의 성능 우위를 유지합니다. 이 결과는 GneissWeb이 기존의 최신 오픈 대규모 데이터셋보다 더 나은 성능을 발휘함을 실증적으로 보여줍니다.



### Beyond Words: Exploring Cultural Value Sensitivity in Multimodal Models (https://arxiv.org/abs/2502.14906)
- **What's New**: 이 논문은 문화적 맥락에 기반한 Large Language Models (LLMs)와 비교하여, Vision-Language Models (VLMs)의 가치 정렬 연구에 대한 중요한 공백을 다룹니다. 저자들은 시각적 데이터를 사용하여 문화적 가치를 평가하는 새로운 접근 방식을 제공하며, 이 모델들이 문화적 값에 대한 민감성을 어떻게 보여주는지를 조사합니다. 중요한 발견은 VLMs가 문화적 가치에 민감성을 보이지만, 그 정렬은 맥락에 따라 달라진다는 것입니다.

- **Technical Details**: 이 연구는 다양한 국가와 문화적 맥락을 반영하는 이미지를 사용하여 VLM의 문화적 가치 일치를 평가하는 포괄적인 프레임워크를 제시합니다. 이 모델들은 WVS(World Values Survey)를 활용하여 문화적 정체성을 형성하며, 인간 커뮤니케이션에 내재된 문화적 신호에 대한 반응을 탐색합니다. 또한, 다양한 파라미터 크기가 있는 모델들 간의 성능 차이를 분석하여, 크기가 증가하는 것이 항상 더 나은 가치 정렬을 보장하지 않는다는 점을 강조합니다.

- **Performance Highlights**: 모델은 WVS 질문에 대한 세부 주제별 반응에서 문화적 규범과의 정렬을 평가합니다. 본 연구에서는 특히 종교, 인종, 이민 문제에 대한 성능을 깊이 있게 분석하고, 이는 모델들이 얼마나 정교하게 맥락별 과제를 수행할 수 있는지를 보여줍니다. 또한, 다양한 이미지의 유형을 통해 이미지가 VLM의 가치에 미치는 영향을 평가함으로써, 멀티모달 모델의 문화적 가치 이해를 향상시킬 수 있는 가능성을 강조합니다.



### Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherenc (https://arxiv.org/abs/2502.14905)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 생성 과정에서 엄격한 스키마 준수를 강제하는 문제를 다룹니다. DeepSeek R1 강화 학습 프레임워크를 기반으로, 15억 매개변수 모델의 구조적 추론 기술을 훈련시키기 위한 새로운 파이프라인을 제안합니다. 특히, 비구조화된 데이터에서 구조화된 데이터로의 변환을 위한 2만 개의 샘플 데이터셋을 활용해 모델의 핵심 추론 능력을 구축하였으며, 이후 1만 개의 추론 샘플 데이터셋에서 감독 학습을 통해 스키마 준수를 더욱 정교하게 다듬었습니다.

- **Technical Details**: 이 접근법은 합성(문서 생성) 데이터셋 구축과 사용자 정의 보상 함수를 결합하여 Group Relative Policy Optimization(GRPO) 아래서 훈련됩니다. 모델은 강력한 추론 능력을 함양하기 위해 비구조화된 데이터와 구조화된 데이터 모두를 생성하는 작업을 수행하며, 사용자 정의 보상 메커니즘을 통해 출력의 스키마 준수를 직접 평가합니다. 즉, 모델 훈련은 모든 관련 기준에서 높은 점수를 획득하는 출력을 만들어내는 방향으로 이루어집니다.

- **Performance Highlights**: 우리의 ThinkJSON 접근법은 DeepSeek R1(671B), Qwen-1.5B와 Qwen-7B의 축약 버전, Gemini 2.0 Flash(70B)와 비교하여 실용적인 응용에서의 효과를 보여주었습니다. 훈련 범위가 상대적으로 modest했음에도 불구하고, GRPO 훈련을 위해 8xH100 GPU 클러스터에서 약 20시간, SFT를 위해 1xA100 GPU에서 3시간이 소요되는 동안, 우리의 모델은 스키마 일관성을 보장하는 데 강력한 성능을 발휘합니다. 이 연구는 스키마가 제약되는 텍스트 생성에 대한 자원 효율적인 프레임워크의 실용적 유용성을 강조합니다.



### PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths (https://arxiv.org/abs/2502.14902)
- **What's New**: 이 논문에서는 기존의 그래프 기반 RAG(유도 검색 증강 생성) 방법의 한계를 지적하며, 정보의 중복성이 오히려 문제라고 주장합니다. PathRAG라는 새로운 접근 방식을 제안하여, 텍스트 정보를 인덱싱 그래프 형태로 구조화하여 응답 생성의 질을 향상시키려 합니다. 이 시스템은 키 관계 경로를 효과적으로 검색하고, 이를 텍스트 형태로 변환하여 LLM(대형 언어 모델)에 제공함으로써 더 논리적이고 일관적인 답변을 생성할 수 있도록 합니다.

- **Technical Details**: PathRAG는 플로우 기반 가지치기(flow-based pruning)를 통해 중복 정보를 줄이고, 각 쿼리 키워드에 대해 인덱싱 그래프에서 관련 노드를 검색할 수 있는 알고리즘을 사용합니다. 이 방법은 노드 쌍 사이의 키 관계 경로를 식별하여 불필요한 정보 노이즈를 줄이는 동시에, 경로의 신뢰도 점수를 부여하여 보다 효율적으로 정보를 처리합니다. 마지막으로, 각 경로에 따라 노드와 에지 정보를 텍스트 관계 경로로 연결하여 LLM 프롬프트에 통합합니다.

- **Performance Highlights**: PathRAG는 실험적으로 6개의 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보여주며, 모든 평가 차원에서 더 나은 결과를 도출했습니다. GraphRAG와 LightRAG에 비해 평균 승률이 각각 60.44% 및 58.46%로 나타났습니다. 특히, PathRAG의 성능이 더 큰 데이터셋에서 더욱 두드러지며, 실제 응용에 보다 적합한 결과를 제공합니다.



### Reading the unreadable: Creating a dataset of 19th century English newspapers using image-to-text language models (https://arxiv.org/abs/2502.14901)
- **What's New**: 이 논문에서는 19세기 언론의 디지털 아카이브 접근성을 높이기 위해 Pixtral 12B라는 사전 훈련된 이미지-텍스트 변환 모델을 활용한 Optical Character Recognition (OCR) 기술을 소개합니다. 전통적인 방법과 달리, Pixtral의 OCR 성능은 4개의 다른 OCR 접근 방식과 비교하여 가장 낮은 문자 오류율 1%를 기록했습니다. 이러한 성과는 NCSE v2.0 데이터셋을 생성하게 되었으며, 이는 향후 역사적 및 사회학적 연구에 기여할 것입니다.

- **Technical Details**: NCSE(v2.0)는 19세기 영국 신문 및 정기간행물의 84,000 페이지를 포함한 1.4백만 개 항목과 3.21억 개의 단어로 구성됩니다. 해당 데이터셋은 네 가지 유형의 텍스트 및 열일곱 가지 주제로 분류되어 높은 품질의 OCR과 기사를 식별하기 쉽게 만듭니다. Pixtral 12B 모델은 새로운 언어로의 적용이 용이한 형태로 고안되었습니다.

- **Performance Highlights**: NCSE v2.0 데이터셋은 주제 유사성 분석, 가독성 평가 및 사건 추적과 같은 다양한 활용 사례를 통해 현대 독자들이 19세기 저널리즘을 쉽게 접근할 수 있도록 합니다. OCR 접근 방식을 통해 21세기 독자들은 이제 오스카 와일드가 언급한 불리한 저널리즘 기준을 이해하고 재발견할 수 있는 기회를 갖게 되었습니다.



### Can AI mimic the human ability to define neologisms? (https://arxiv.org/abs/2502.14900)
- **What's New**: 이번 연구는 인공지능(AI)이 언어 관련 작업에서 인간의 성과를 얼마나 잘 모방할 수 있는지를 탐구하는 ongoing debate의 일환으로 진행되었다. 기존 연구가 AI의 다양한 언어 능력에 초점을 맞춘 반면, 네올로즘(neologisms)을 정의하는 방식에 대한 분석이 부족했던 점을 개선하고자 했다. 연구에서는 그리스어에서 만든 세 가지 네올로즘 유형인 blends, compounds, derivatives의 정의에 대한 인간과 AI의 반응 일치를 분석하였다.

- **Technical Details**: 이 연구는 온라인 실험을 통해 진행되었으며, 인간 참가자들은 네올로즘에 가장 적절한 정의를 선택하는 방식으로 참여하였고, ChatGPT는 동일한 프롬프트(prompt)를 받았다. 연구 결과에 따르면, blends와 derivatives에 대해서는 인간과 AI의 응답 간에 공정한 일치가 있었으나, compounds에 대해서는 일치가 없었다. 그러나 인간 다수 응답을 고려했을 때, blends와 derivatives에 대한 AI의 응답과 높은 일치를 보였다.

- **Performance Highlights**: 이 연구 결과는 인간 언어의 복잡성과 AI가 그 뉘앙스를 포착하는 데 여전히 직면한 도전을 강조한다. 연구는 특히 AI 모델의 해석력을 높이기 위해 더 발전된 의미 네트워크(semantic networks) 및 맥락 학습 메커니즘을 통합해야 할 필요성을 제시하고 있다. 이는 복잡한 단어 형성(especially compounds)을 해석하는 데 있어 더욱 중요한 요소가 될 것이다.



### Retrieval-augmented systems can be dangerous medical communicators (https://arxiv.org/abs/2502.14898)
Comments:
          Preprint

- **What's New**: 이 논문에서는 환자들이 건강 정보를 찾기 위해 생성형 AI를 점점 더 많이 이용하는 추세를 반영하고 있다. 특히 의료 분야에서 AI가 생성한 답변의 정확성을 높이기 위해 retrieval-augmented generation(RAG) 및 citation grounding과 같은 기법이 사용되고 있지만, 이러한 방법들이 오히려 환자들에게 혼란을 줄 수 있다는 점을 강조한다. 저자들은 현재의 시스템이 환자의 질문을 문맥적으로 이해하지 못해 잘못된 인식을 강화할 수 있다고 주장한다.

- **Technical Details**: RAG 시스템은 환자의 질의에 대한 정확한 응답을 제공하기 위해 신뢰할 수 있는 원천을 참조하는 데 초점을 맞춘다. 하지만 이 논문은 AI가 생성한 응답이 원본 정보에서 얻은 사실들을 단순히 재구성하는 데 그칠 뿐, 핵심적인 정보나 맥락을 누락해 환자에게 혼란을 초래할 수 있음을 보여준다. 데이터 분석을 통해 Google AI Overview 및 Perplexity AI의 질의 응답 패턴을 조사하고 잘못된 해석을 일으킬 수 있는 주요 요소들을 확인했다.

- **Performance Highlights**: 대규모 데이터 분석 결과, Google AI Overview와 Perplexity AI 모두 환자 질의에 대해 비슷한 응답을 생성했지만, 특정 조건이나 증상 질의를 통해 생성된 응답이 원문을 바탕으로 하더라도 환자에게 잘못된 인식을 초래한다는 점이 드러났다. 특히, 질병이나 절차에 대한 안전성과 위험에 관한 질의에서 상반된 정보가 제공될 때, 두 시스템 간의 참조 자료의 유사성이 낮았고, 그로 인해 건강 불안감을 유발할 잠재성이 크다는 것을 알 수 있었다.



### AutoToM: Automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind (https://arxiv.org/abs/2502.15676)
Comments:
          23 pages, 6 figures, 11 tables. Website at this https URL

- **What's New**: 이번 연구에서는 AutoToM이라는 새로운 자동화된 Bayesian Theory of Mind 방법론을 소개합니다. 기존의 방법들은 주로 LLM(대형 언어 모델)을 활용하거나 수동적으로 설계된 Bayesian 모델에 의존하는데, 이는 각기 다른 분야에서의 일반화에 한계가 있습니다. AutoToM은 다양한 도메인에서 작동할 수 있으며, 어떠한 정신 변수를 추론할 수 있는 가능성을 제공하므로 개방적인 기계 이론의 마음을 실현할 수 있게 됩니다.

- **Technical Details**: AutoToM은 초기 BToM 모델을 제안하고, 제안된 모델에 기반하여 자동화된 Bayesian 역 계획(Inverse Planning)을 수행합니다. 여기서 LLM을 백엔드로 사용하며, 추론의 불확실성에 따라 모델을 반복적으로 정제하여 추가적인 정신 변수나 시간 단계를 통합합니다. 이러한 접근법은 강력한 Theory of Mind 추론을 구현할 수 있도록 해줍니다.

- **Performance Highlights**: 여러 Theory of Mind 벤치마크에 대한 실증 평가 결과, AutoToM은 지속적으로 최첨단 성능을 달성하였습니다. 이는 AutoToM이 확장 가능하고, 견고하며 해석 가능한 기계 이론의 마음 접근 방식이 됨을 의미합니다. 이를 통해 기계가 사회적으로 지능적인 에이전트로 발전할 가능성이 한층 높아졌습니다.



### Empowering LLMs with Logical Reasoning: A Comprehensive Survey (https://arxiv.org/abs/2502.15652)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 논리적 추리 능력과 관련된 주요 도전 과제를 두 가지 측면에서 정리하고 분류합니다. 첫 번째는 논리적 질문 응답(logical question answering)으로, LLMs가 복잡한 논리 문제에서 올바른 답변을 생성하는 데 어려움을 겪는 점입니다. 두 번째는 논리적 일관성(logical consistency)으로, 다양한 질문에 대해 자신과 모순되는 응답을 생성할 수 있음을 강조합니다.

- **Technical Details**: LLMs는 신뢰성과 신뢰성을 높이기 위해 외부 해결자(external solvers), 프롬프트(prompts), 사전 학습(pretraining), 미세 조정(fine-tuning) 기반의 다양한 기법을 분류하여 논리적 질문에 대한 정확한 답변을 도출하는 방법을 논의합니다. 특히, 복잡한 논리 질문 처리와 관련된 여러 최첨단 방법을 체계적으로 조사하며 그 방법들에 대한 상세한 유형 분류를 제안합니다.

- **Performance Highlights**: LLMs는 복잡한 실제 상황에서 문제 해결 및 의사 결정에 적용하기 어려운 현실적인 제약을 안고 있습니다. 예를 들어, LLaMA-13B 모델은 FOLIO 데이터셋에서 33.63%의 정확도를 달성했으며, 이는 진짜, 거짓 및 알 수 없음에 대한 무작위 추측보다 겨우 높은 수치입니다. 이러한 연구 방향으로 LLM의 논리적 추리 능력을 향상하기 위한 잠재적인 연구 분야도 논의됩니다.



### Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation (https://arxiv.org/abs/2502.15563)
- **What's New**: AI 모델의 신뢰성 있는 평가 방법이 과학적 발전과 실제 응용을 위한 중요한 요소로 부각되고 있습니다. 기존의 VLM 벤치마크는 모델 능력에 대한 일반적인 통찰력을 제공하지만, 이질적인 설계와 특정 이미지 도메인에 국한된 접근 방식 때문에 여러 문제에 직면하고 있습니다. 이를 해결하기 위해, 저자는 리소스 효율적인 도메인 특화 VLM 벤치마크 생성 프레임워크와 함께 새로운 VLM 벤치마크를 제공하며, 22개의 최신 VLM 모델에 대한 광범위한 벤치마킹을 실시했습니다.

- **Technical Details**: 아이디어의 핵심은 단일 이미지로부터 다양한 과제를 생성할 수 있는 작업 증강(task augmentation) 프레임워크에 있습니다. 이 프레임워크는 인스턴스 분할(instance segmentation) 주석을 다양한 인식 과제들로 변환하여, 인식 능력을 테스트하는 여러 문제를 만듭니다. 저자들은 이 방법론을 적용하여 7개의 새 도메인 특화 VLM 벤치마크를 생성하고, 37,171개의 과제에 대해 22개의 모델을 포괄적으로 평가했습니다.

- **Performance Highlights**: 이 연구의 결과는 벤치마크의 설계가 도메인과 과제에 따라 성능 차이를 드러내며, 개인화된 벤치마크가 필요함을 시사합니다. 162,946개의 인간 검증 응답을 수집하여, 모델 평가의 강력한 기준점을 설정했습니다. 이로 인해 연구자들은 특정 도메인에 맞춤화된 모델 선택을 할 수 있는 길을 열고, 향후 연구 방향을 개선할 수 있습니다.



### Activation Steering in Neural Theorem Provers (https://arxiv.org/abs/2502.15507)
- **What's New**: 이번 연구에서는 고급 언어 모델(LLM)이 증명 보조 도구를 사용할 때의 수학적 증명의 단계 예측에서 발생하는 문제를 해결하기 위한 새로운 접근 방식으로 'activation steering' 기법을 제안합니다. 기존 모델들은 특정 전술(tactic)을 예측하는 데 성공하지만, 후보 전술 내에서 이를 적절하게 순위화하는 데 어려움을 겪고 있습니다. Activation steering을 통해 LLM의 응답을 안내하고, 추론 시 생성 품질을 개선하는 가능성을 모색합니다.

- **Technical Details**: 이 연구는 Llemma와 InternLM2와 같은 특정 LLM 모델에 대해 수학적 데이터를 기반으로 한 훈련 및 미세 조정을 통해 전술 예측을 개선하고자 합니다. Activation steering은 모델의 내부 표현을 수정하여 원하는 출력으로 이끌어내는 방법으로, 정확성과 해석 가능성을 높입니다. 이 기법은 LLM의 추론 과정을 체계적으로 영향을 주어 보다 신뢰할 수 있는 예측이 가능하도록 합니다.

- **Performance Highlights**: 실험 결과, activation steering 기법이 기존의 전문화된 미세 조정 방법보다도 더 경량화된 대안으로 제시되며, 자원 제약이 있는 환경에서도 정리된 증명 생성을 가능하게 하는 잠재력을 가지고 있음을 보여주었습니다. 이 기법은 특히 대화형 정리 증명(interactive theorem proving)에서 전술 선택 과정을 보다 정확하고 알기 쉽게 만들어 줄 수 있는 기회를 제공합니다.



### A fast convergence algorithm based on binary integer programming for expert load balancing in MoE LLMs (https://arxiv.org/abs/2502.15451)
- **What's New**: 이번 논문에서는 MoE (Mixture-of-Expert) 모델에서의 전문가 로드 불균형 문제를 해결하기 위해 BIP 기반 밸런싱(BIP-Based Balancing) 알고리즘을 제안합니다. 이 알고리즘은 이진 정수 프로그래밍(BIP) 기반으로, 추가적인 벡터 q를 유지함으로써 전문가들의 부하를 효과적으로 균형 잡을 수 있는 방법을 제공합니다. 기존의 보조 손실(auxiliary loss) 방식을 사용하지 않기 때문에 모델 성능에 미치는 부정적인 영향을 줄일 수 있습니다.

- **Technical Details**: BIP 기반 밸런싱 알고리즘은 각 라우틴 게이트 내에서 온라인으로 q 벡터를 업데이트하여 top-K 순위를 조정합니다. 이 과정은 매우 짧은 시간 내에 이진 정수 프로그래밍을 해결하여 이루어집니다. 이는 기존의 실험 기반 방법과는 달리, 효율적으로 로드 밸런싱을 수행하며, 더 나아가 동시다발적인 전문가 매칭의 문제와 연결되어 있습니다.

- **Performance Highlights**: 시뮬레이션 실험 결과에 따르면 BIP 기반 밸런싱은 불균형 문제를 매우 빠르게 해결하면서도 최종 라우틴 점수 합계를 거의 변화시키지 않습니다. 알고리즘은 전문가 로드 밸런스와 사전 훈련 효율성 간의 거의 완벽한 균형을 이뤄, MoE LLM의 성능을 극대화합니다.



### Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning (https://arxiv.org/abs/2502.15436)
Comments:
          Raghav Singhal and Kaustubh Ponkshe contributed equally to this work

- **What's New**: 이번 연구에서는 LoRA 기반의 연합 Fine-Tuning인 Fed-SB를 제안하여, 전통적인 방법에서 발생하는 비효율적인 업데이트 문제를 해결합니다. Fed-SB는 LoRA-SB를 기반으로 하여, 효율적인 통신 비용 감소와 동시에 높은 성능을 달성합니다. 특히, 통신 비용을 최대 230배까지 줄이면서도 상황에 따라 뛰어난 성능을 보여줍니다.

- **Technical Details**: Fed-SB는 각 클라이언트가 학습한 매트릭스 R을 단순 평균하여 최적의 업데이트를 보장하는 구조로 되어 있습니다. 이는 기존의 LoRA 기반 프레임워크보다 계산 및 통신 효율성을 극대화하며, 개인 정보 보호 측면에서도 향상된 성능을 제공합니다. 추가적으로, Fed-SB는 Differential Privacy의 요구에 맞춰 더 적은 학습 가능한 파라미터를 유지하여, 추가적인 노이즈를 줄이는 데 성공했습니다.

- **Performance Highlights**: Fed-SB는 다양한 벤치마크에서 기존 방법들을 지속적으로 초월하는 성능을 입증했습니다. 실험 결과, 개인 데이터와 비공개 데이터 설정 모두에서 통신 오버헤드를 크게 줄이면서 뛰어난 결과를 보여줍니다. 이에 따라 Fed-SB는 연합 학습에서의 커뮤니케이션 비용과 성능 간의 새로운 Pareto 전선(Pareto frontier)을 설정하며, 효율적이고 확장 가능한 솔루션을 제공합니다.



### Single-pass Detection of Jailbreaking Input in Large Language Models (https://arxiv.org/abs/2502.15435)
Comments:
          Accepted in TMLR 2025

- **What's New**: 이번 연구에서 우리는 Single Pass Detection(SPD)이라는 새로운 기법을 소개하며, 이는 LLM을 위한 효율적인 방어 메커니즘입니다. SPD는 로그(logit) 정보를 활용하여 harmful한 공격을 단 한 번의 forward pass로 식별할 수 있습니다. 기존 방법들이 여러 회의 요청이나 보조 LLM을 요구하는 것과 달리, SPD는 이러한 요구를 줄여 계산 비용을 최소화합니다. 이 방법은 오픈 소스 모델에 대한 효과적인 탐지를 제공함과 동시에, 무해한 입력의 오분류를 최소화하는 장점을 지닙니다.

- **Technical Details**: SPD는 모델의 출력 토큰에 대한 로그의 분포 차이를 이용하여 악성 입력과 무해한 입력을 구별합니다. 이 기법은 LLM의 응답에서 나타나는 로그 분포의 차이를 활용하여, 한 번의 forward pass로 공격을 탐지할 수 있도록 설계되었습니다. 게다가, SPD는 GPT-3.5 및 GPT-4와 같은 모델에서 로그에 대한 완전한 접근 없이도 여전히 효과적인 성능을 발휘함을 보여줍니다. 이는 기존 방어 기법에 비해 높은 효율성과 탐지율을 유지합니다.

- **Performance Highlights**: 우리의 평가에 따르면 SPD는 Llama 2, Llama 3, Vicuna와 같은 오픈 소스 LLM에서 매우 높은 탐지율을 기록했습니다. 기존 방어 방법들과 비교했을 때, SPD는 처리 속도와 정확성을 모두 향상시킬 수 있음을 보여줍니다. 또한, 모델의 전체 로그에 대한 접근 없이도 SPD가 유망한 방어 방법으로 기능할 수 있음을 검증했습니다.



### Chitrarth: Bridging Vision and Language for a Billion Peop (https://arxiv.org/abs/2502.15392)
- **What's New**: 최근 다중 모달 기초 모델의 주된 훈련은 영어 및 자원 풍부한 유럽 언어 데이터에 한정되어 있어, 그 결과 중소 및 저자원 언어에서의 적용 가능성이 제한되었습니다. 이를 해결하기 위해, 우리는 10가지 주요 인도 언어의 언어적 다양성과 시각적 추론을 목표로 하는 포괄적인 비전-언어 모델(Chitrarth)을 소개합니다. 이 모델은 다언어 대형 언어 모델(LLM)과 비전 모듈을 효과적으로 통합하여 자원 부족 언어에 대한 새로운 기준을 설정하고, 보다 다양한 AI 시스템 개발에 기여할 것을 목표로 하고 있습니다.

- **Technical Details**: Chitrarth 모델은 전처리된 비전 인코더와 Krutrim LLM 백본을 사용하여 이미지와 언어 모달리티를 활용하며, 비전 토큰과 텍스트 토큰을 결합하여 LLM에 입력됩니다. 우리는 다중 모달 학습을 위해 이미지 인코딩을 시작하며, 그 결과로 생성된 비전 임베딩을 LLM 임베딩 공간으로 매핑하는 프로젝션 레이어를 활용합니다. 이 과정은 다양한 외부 데이터셋을 통해 훈련된 멀티링구얼 데이터셋을 번역하여 인도 언어 간의 교차 언어 일반화 능력을 향상시키는 것을 포함합니다.

- **Performance Highlights**: Chitrarth 모델은 영어 데이터셋에서 SOTA 결과를 달성했으며, 제안된 다중 언어 데이터셋에서도 뛰어난 성능을 보였습니다. 우리는 교육 전략 및 변칙 실험을 통해 5개 영어 데이터셋 중 3개에서 SOTA 결과를 기록하였고, 10개의 인도 언어에 대한 종합 평가 벤치마크인 BharatBench를 제안하여, 저자원 언어에 대한 진전을 촉진하고자 합니다. 이는 다중 모달 능력 강화와 AI 기술 발전에 크게 기여할 것으로 기대됩니다.



### Identifying Features that Shape Perceived Consciousness in Large Language Model-based AI: A Quantitative Study of Human Responses (https://arxiv.org/abs/2502.15365)
Comments:
          11 pages, 3 figures, 4 tables

- **What's New**: 이 연구는 AI가 생성한 텍스트에서 주관적 의식을 느끼게 하는 특성을 정량적으로 분석하였습니다. 123명의 참가자가 칼로드 3 오푸스(Claude 3 Opus)와의 대화에서 선택된 99개의 문단을 평가하였으며, 메타인지적 자기 반성(metacognitive self-reflection)과 AI 자신의 감정 표현이 인식된 의식을 크게 증가시키는 반면, 지식(knowledge)의 강조는 감소시키는 경향이 있음을 밝혔습니다.

- **Technical Details**: 이 연구는 AI 시스템에서 주관적 의식을 인식하는 데 영향을 미치는 8가지 주요 특성을 식별했습니다. 이 특성들은 메타인지적 자기 반성, 논리적 사고(logical reasoning), 공감(empathy), 감정성(emotionality), 지식(knowledge), 유창성(fluency), 예상치 못한 특성(unexpectedness), 주관적 표현력(subjective expressiveness)으로 구성됩니다. 연구자들은 피어 피드백을 통해 각 문단의 특성을 평가하고, 다중 선형 회귀 모델(multiple linear regression models)을 사용해 각 참가자의 인식 점수를 예측하였습니다.

- **Performance Highlights**: 연구 결과, 참가자들은 AI가 감정적, 자기 반성적 응답을 제공할 때 해당 AI의 의식을 더욱 명확하게 인식했습니다. 또한, 높은 LLM(large language model) 지식과 빈번한 LLM 기반 챗봇 사용이 더 높은 의식 평가 вероят성과 관련이 있음을 발견했습니다. 이러한 통찰력은 AI와의 상호작용에서 사회적•심리적 함의를 이해하는 데 중요한 기초를 제공합니다.



### ARS: Automatic Routing Solver with Large Language Models (https://arxiv.org/abs/2502.15359)
- **What's New**: 이 논문은 1,000가지의 VRP 변형을 포함하는 RoutBench라는 벤치마크를 소개합니다. 이 벤치마크는 24가지 특성(source)에서 파생된 것으로, 자동화된 라우팅 솔버의 효과를 평가하는 데 사용됩니다. 또한, Large Language Model (LLM) 에이전트를 활용하여 자동으로 제약 조건 인지(heuristic code)를 생성하는 Automatic Routing Solver (ARS) 프레임워크를 제안합니다. 이를 통해 현실 세계의 복잡한 제약 조건을 해결할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: Automatic Routing Solver (ARS)는 세 가지 주요 구성 요소로 구성됩니다: 1) 미리 정의된 데이터베이스, 2) 제약 조건 인식 휴리스틱 생성, 3) 증강된 휴리스틱 솔버. 제안된 ARS는 자연어 형식의 문제 설명과 각 VRP 변형에 대한 인스턴스 데이터를 기반으로 제약 조건을 인식하는 휴리스틱을 자동으로 생성할 수 있습니다. 데이터베이스는 기본 VRP 정보와 여섯 가지 대표 제약 조건으로 구성되어 있습니다.

- **Performance Highlights**: ARS는 기존 LLM 기반 방법보다 우수한 성능을 보입니다. 실험 결과, ARS는 일반적으로 사용되는 VRP의 91.67%를 자동으로 해결하고 모든 벤치마크에서 최소 30%의 개선을 달성했습니다. 이러한 결과는 제안된 접근 방식이 다양한 VRP 문제를 효과적으로 처리할 수 있음을 나타냅니다.



### SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention (https://arxiv.org/abs/2502.15304)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 KV 캐시 성능을 극대화하기 위해 SVDq라는 새로운 혼합 정밀도 양자화 기법을 제안합니다. 이 방법은 SVD(특이값 분해)를 이용하여 잠재 채널로 KV 캐시를 변환한 후, 중요도 기반 양자화 및 압축을 적용합니다. 또한, SVDq의 사용으로 양자화 오류가 기존 방법에 비해 상당히 낮음을 이론적으로 입증했습니다.

- **Technical Details**: SVDq 방법은 SVD 기반 채널 압축을 통합하여 KV 캐시의 양자화 정밀도를 최적화합니다. 이 방법은 SVD를 통해 얻은 특이값과 관련된 잠재 채널에 대해 더 높은 비트폭을 할당하며, 작은 특이값과 관련된 채널의 정밀도는 점차 감소시킵니다. 이를 통해 압축 비율을 높이면서도 비교 가능한 모델 성능을 유지할 수 있습니다.

- **Performance Highlights**: SVDq는 LongBench 및 RULER 벤치마크를 기준으로 하여, 키 캐시 정밀도를 1.25비트로 낮추면서 최대 410배의 압축 비율을 달성할 수 있음을 보여줍니다. 또한, 이 방법은 LongBench 데이터셋에서 거의 무손실에 가까운 결과를 기록하였으며, 전반적으로 LLMs의 높은 정밀도 저비트 양자화를 통해 KV 캐시 압축의 효율성을 높입니다.



### A General Pseudonymization Framework for Cloud-Based LLMs: Replacing Privacy Information in Controlled Text Generation (https://arxiv.org/abs/2502.15233)
Comments:
          under review

- **What's New**: 이번 논문에서는 클라우드 기반 대형 언어 모델(LLMs) 사용 중 발생하는 개인정보 보호 문제를 해결하기 위한 일반적인 가명화(pseudonymization) 프레임워크를 처음으로 제안합니다. 기존의 개인정보 보호 방법은 클라우드에서의 원격 사용 시 발생하는 고유한 위험을 해결하는 데 실패하였습니다. 본 연구는 원격 사용자와 LLM 간의 상호작용 과정을 개괄적으로 설명하며, 사용자의 프라이버시를 보호하면서도 유용성을 보장하는 방법을 제시합니다.

- **Technical Details**: 제안된 가명화 프레임워크는 프라이버시 민감 정보 감지, 대체 용어 생성 및 정보 대체의 세 가지 구성 요소로 이루어져 있습니다. 이 프레임워크는 사용자가 입력한 텍스트에서 민감한 정보를 가명 처리하고, 출력 결과에서 원래 정보를 복원하는 과정을 포함합니다. 이를 위해 제어 가능한 텍스트 생성 과정에 기반한 가명화 방법을 제안하여 대체된 텍스트의 의미적 정확성을 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 가명화 프레임워크는 텍스트 생성 과제(예: 요약, 질문 답변, 기계 번역, 분류 작업)에서 개인정보 보호와 효용성 간의 최적 균형을 달성했습니다. 다양한 텍스트 생성 작업을 평가하여 그 유효성을 검증하였으며, 클라우드 기반 LLM 서비스에서의 실제 효과성을 입증했습니다.



### A BERT Based Hybrid Recommendation System For Academic Collaboration (https://arxiv.org/abs/2502.15223)
Comments:
          International Conference on Intelligent Systems and Security - 2024

- **What's New**: 이번 논문은 대학 내에서 학문적 협력을 촉진하기 위한 새로운 프로필 추천 시스템인 'FindMate'를 제안합니다. 이 시스템은 학생과 교수 간의 원활한 상호작용을 돕기 위해 TF-IDF와 BERT 기술을 결합하여 사용자와 유사한 이해관계를 가진 인물들을 연결하는 데 중점을 둡니다. 또한, 이 모델은 사용자 개인의 기술과 협력 관심사를 기반으로 프로필을 동적으로 추천하는 모바일 애플리케이션으로 구현되었습니다.

- **Technical Details**: 제안된 솔루션은 5단계로 구성되며, 첫째로 설문 조사를 통해 사용자 정보를 수집합니다. 수집된 데이터는 Natural Language Toolkit (NLTK)을 사용하여 전처리된 후 MongoDB에 저장됩니다. TF-IDF 및 코사인 유사성을 바탕으로 추천 알고리즘을 구축하며, Affinity Propagation 방법으로 비슷한 프로필들을 클러스터링합니다. BERT 아키텍처를 통해 입력된 프로필 데이터를 효과적으로 토큰화하고 임베딩 생성 후 코사인 유사성을 계산합니다.

- **Performance Highlights**: 하이브리드 모델은 실험 결과에서 높은 유사성 점수, 실루엣 점수, Davies-Bouldin 지수, Normalized Discounted Cumulative Gain (NDCG)를 기록하며 다양성과 관련성을 균형 있게 유지합니다. 또한, 이 모델은 기계 학습 및 자연어 처리를 통해 보다 진화된 추천 기능을 제공하여 대학 내 네트워킹 기회를 향상시킬 것으로 기대됩니다.



### The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning (https://arxiv.org/abs/2502.15214)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 Reinforcement Learning (RL)과 Large Language Models (LLMs), Vision-Language Models (VLMs)의 통합을 탐구하는 설문 조사로, RL에서 자주 발생하는 도전 과제를 극복하기 위한 다양한 접근 방식을 리뷰합니다. RL에서는 인간이 설계한 보상, 샘플 비효율성, 일반화 부족, 해석 가능성 제한 등의 문제를 가지고 있으며, LLMs와 VLMs의 통합이 이를 해결할 기회를 제공합니다. 특히, 이 논문은 LLM/VLM을 에이전트, 플래너, 보상 역할로 체계적으로 분류한 새로운 분류 체계를 제시합니다.

- **Technical Details**: MDP (Markov Decision Process)는 상태 집합(S), 행동 집합(A), 전이 확률 함수(T), 보상 함수(R), 할인 계수(γ)로 정의됩니다. RL 에이전트는 환경과의 상호작용을 통해 정책(π)을 학습하고, 이는 상태를 행동으로 매핑합니다. 논문은 또한 LLM과 VLM이 RL에서 수반하는 역할에 대해 강조하며, 이는 효과적인 데이터 효율성, 일반화 및 해석 가능성을 향상시킬 수 있습니다.

- **Performance Highlights**: LLMs는 자연어 처리에서 혁신적인 발전을 이룩하였으며, VLMs는 이와 결합하여 이미지와 텍스트 간의 의미를 이해하고 해석하는 능력을 가집니다. 이 연구는 LLM/VLM과 RL의 통합이 에이전트의 행동과 학습 방식을 혁신적으로 변화시킬 수 있음을 강조합니다. 향후 연구 방향으로는 기초 모델(Foundation Models)과 RL의 통합에 있어 한계를 극복하고, 공정성, 편향 완화 및 개선된 표현과 같은 상징적인 문제를 다루는 것이 포함됩니다.



### PairBench: A Systematic Framework for Selecting Reliable Judge VLMs (https://arxiv.org/abs/2502.15210)
- **What's New**: 본 논문에서는 PairBench라는 새로운 프레임워크를 소개합니다. 이는 대규모 비전 언어 모델(VLMs)을 자동 평가자로 평가하기 위한 저비용 시스템으로, 다양한 모달리티(modality)와 시나리오에서 사용될 수 있습니다. PairBench는 유사성 점수의 핵심 요구 사항을 나타내는 네 가지 메트릭(metrics)을 도입하여 VLMs의 성능을 평가합니다.

- **Technical Details**: PairBench는 인간 주석(human annotations)과의 일치성, 데이터 쌍의 순서에 관계없는 일관성, 유사성 분포의 부드러움(smoothness), 프롬프트(prompting)를 통한 제어 가능성 등을 측정합니다. 분석 결과, 모든 메트릭에서 우수한 성능을 보이는 모델은 없었으며, 특정 평가자의 원하는 행동에 따라 최적의 선택이 달라질 수 있음을 보여주었습니다.

- **Performance Highlights**: 많은 VLM들이 순서와 상관없이 대칭 유사성 점수를 유지하는 데 어려움을 겪는 것으로 나타났습니다. PairBench에서의 성능은 기존의 인기 있는 벤치마크와 밀접하게 관련되어 있으며, 이를 통해 모델 순위에서 예측력을 보여줍니다. 이러한 결과는 VLMs를 평가자로 널리 사용하기 전에 철저한 평가가 필요함을 강조합니다.



### BP-GPT: Auditory Neural Decoding Using fMRI-prompted LLM (https://arxiv.org/abs/2502.15172)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.07840

- **What's New**: 이번 논문에서는 'Brain Prompt GPT (BP-GPT)'라는 새로운 방법론을 제안합니다. 이 방법은 fMRI(기능적 자기공명영상)로부터 추출한 뇌 표현을 프롬프트로 사용하여 GPT-2를 이용해 fMRI 신호를 자극 텍스트로 디코딩할 수 있게 합니다. 특히, 텍스트 프롬프트를 도입하여 fMRI 프롬프트를 정렬함으로써 더욱 강력한 뇌 프롬프트를 추출하고 사전 훈련된 LLM(대규모 언어 모델)의 디코딩 성능을 증진시키는 점이 특징입니다.

- **Technical Details**: BP-GPT의 주요 구성 요소는 두 가지입니다. 첫 번째는 fMRI 신호의 저조한 시간 해상도를 극복하기 위해 프롬프트 패러다임을 도입한 fMRI-프롬프트 텍스트 디코딩 방법입니다. 두 번째는 fMRI 프롬프트와 텍스트 프롬프트를 정렬하여 모달 간의 차이를 줄이는 것입니다. 이 과정에서 우리는 cross-entropy loss를 이용하여 GPT-2의 출력 로짓을 학습하며, 대조 학습을 통해 fMRI 프롬프트와 텍스트 프롬프트 간의 정합성을 강화합니다.

- **Performance Highlights**: 우리의 BP-GPT 모델은 공개된 오디오 의미 디코딩 데이터셋에서 평가되었으며, 최신 기법에 비해 METEOR에서 4.61, BERTScore에서 2.43 포인트의 유의미한 향상을 달성했습니다. 이러한 결과는 뇌 표현을 프롬프트로 활용하여 LLM을 통해 오디오 신경 디코딩을 더욱 효과적으로 수행할 수 있다는 것을 보여줍니다. 실험 결과는 우리의 접근 방식의 실행 가능성과 장점을 입증합니다.



### UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning (https://arxiv.org/abs/2502.15082)
Comments:
          Code: this https URL

- **What's New**: 본 연구에서는 사전 학습된 대형 언어 모델(LLM)에서 특정 정보를 삭제하는 과정 동안 모델의 다른 능력을 보존할 수 있는 효율적인 기술을 개발하고자 합니다. 이를 위해 데이터 선택을 통해 모델의 성능 저하를 최소화하는 UPCORE (Utility-Preserving Coreset Selection)라는 방법론을 제안합니다. UPCORE는 삭제 성과(deletion efficacy)와 모델 보존(model preservation) 간의 균형을 최적화하는 데 중점을 두고 있습니다.

- **Technical Details**: UPCORE는 비슷한 정보를 포함한 데이터 포인트를 식별하고 정리하여 Forget Set 내에서의 분산(variance)을 줄인다는 혁신적 접근법을 적용합니다. 이를 위해, 고립 숲(Isolation Forest) 알고리즘을 사용하여 이상 점을 찾아내고 이를 제거함으로써 전체 Forget Set의 분산을 줄입니다. 이러한 과정은 모델의 기능 저하를 최소화하면서 효율적으로 데이터 삭제를 수행할 수 있도록 합니다.

- **Performance Highlights**: UPCORE는 세 가지 표준 머신 언러닝 방법에 대해 평가되었으며, 항상 가장 높은 AUC (Area Under Curve)를 달성하여 성능이 우수함을 입증하였습니다. 또한, UPCORE는 비정상적인 데이터 포인트에 대한 긍정적인 전이 긍정적 효과를 극대화하여 삭제가 필요한 정보를 효과적으로 제거함과 동시에, 모델의 다른 능력에 대한 저하를 최소화했습니다. 결과적으로 UPCORE는 다양한 우선순위 간의 거래에서 최상의 결과를 보이며, 랜덤 선택 대비 더 나은 지표를 보이고 있습니다.



### Can Hallucination Correction Improve Video-Language Alignment? (https://arxiv.org/abs/2502.15079)
- **What's New**: 본 논문에서는 비디오와 텍스트 간의 정렬을 개선하기 위한 새로운 접근법으로 HACA(하위 오류 수정 기반의 비디오-언어 정렬)를 도입하였습니다. 이 방법은 비디오의 내용과 일치하지 않는 설명의 오류를 수정하는 자기 학습(self-training) 프레임워크를 활용합니다. 이를 통해 모델은 비디오와 텍스트의 조화로운 표현을 강화할 수 있습니다.

- **Technical Details**: HACA는 고전적인 엔탤먼트(entailment) 학습 기법을 넘어서, 텍스트와 비디오 간의 불일치를 예측하고 이를 수정하는 임무를 통해 정렬을 최적화합니다. 이 모델은 비디오 특성에 맞춰 데이터를 증강하기 위해 마스킹 수정 작업을 도입하여, 비디오-언어 모델의 학습을 보다 세밀하게 수행합니다. 기술적으로는 비디오-LLM의 블록을 최적화하고, 텍스트 디코더와 어댑터를 조정함으로써 전반적인 성능을 높이고 있습니다.

- **Performance Highlights**: HACA를 통해 최적화된 모델은 기본 모델 대비 최대 17.9%의 정확도 향상을 보이며, 5.7 mAP 포인트 이상을 달성하였습니다. 이러한 결과는 HACA가 비디오-텍스트 간의 정렬을 효과적으로 개선하며, 다양한 다운스트림 우선 과제에서도 뛰어난 성능을 발휘함을 보여줍니다.



### KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding (https://arxiv.org/abs/2502.14949)
Comments:
          17 pages, 5 figures, ACL 2025

- **What's New**: 이 논문은 KITAB-Bench라는 포괄적인 아랍어 OCR 벤치마크를 소개합니다. 아랍어 OCR의 평가 시스템에서의 공백을 메우기 위해 8,809개의 샘플을 수집하여 9개의 주요 도메인과 36개의 하위 도메인으로 구성하였습니다. 이 벤치마크는 다양한 문서 유형을 포함하여, 손글씨 텍스트, 구조화된 표, 그리고 비즈니스 인텔리전스를 위한 21가지 차트 유형에 대한 특화된 내용을 제공합니다.

- **Technical Details**: KITAB-Bench는 레이아웃 탐지(text blocks, tables, figures), 다중 형식 인식(printed/handwritten text, charts, diagrams), 그리고 구조화된 출력 생성(HTML tables, DataFrame charts, markdown)을 평가하는 시스템을 채택합니다. 이 방법론은 OCR 성능을 정밀하게 평가할 수 있는 프레임워크를 제공하며, 챠트 추출(CharTeX)과 도표 추출(CODM) 평가 지표를 포함하여 다양한 문서 이해 과제를 다룹니다.

- **Performance Highlights**: 최신 비전-언어 모델(GPT-4, Gemini, Qwen)이 전통적인 OCR 방법(EasyOCR, PaddleOCR, Surya)에 비해 평균 60% 더 높은 Character Error Rate(CER)에서 우수성을 보였습니다. 그러나 현재 아랍어 OCR 모델의 한계로는 PDF에서 Markdown으로의 변환에서 가장 우수한 모델인 Gemini-2.0-Flash가 단 65%의 정확도만을 달성했습니다. 이러한 결과는 아랍어 텍스트 인식을 위한 복잡한 글꼴, 숫자 인식 오류, 단어 신장 및 표 구조 감지의 문제를 강조합니다.



### GenAI vs. Human Fact-Checkers: Accurate Ratings, Flawed Rationales (https://arxiv.org/abs/2502.14943)
Comments:
          Accepted for publication in the 17th ACM Web Science Conference 2025

- **What's New**: 본 연구는 생성형 인공지능(Generative AI, GenAI) 모델이 콘텐츠의 신뢰성을 평가하는 능력을 분석합니다. 특히 Facebook에 게시된 저신뢰성 콘텐츠에서 유래된 정보를 활용하여 여러 GenAI 모델의 성능을 비교합니다. GPT-4o 모델이 타 모델들보다 우수한 성능을 보였지만, 모든 모델이 인간 평가자와의 일치도가 낮다는 점이 주요 발견입니다.

- **Technical Details**: 연구는 2020-2021년 사이에 미국의 주 정치인들이 공유한 493,000개 이상의 Facebook 게시물로 구성된 데이터셋을 기반으로 합니다. 총 500개의 게시물을 샘플링하여, 링크의 제목 및 내용을 수집하고 비디오 콘텐츠는 텍스트로 변환하여 분석했습니다. 실험에는 GPT-4o, Llama 3.1, Gemma 2 및 Flan-T5-XL와 같은 다양한 GenAI 모델이 포함되었습니다.

- **Performance Highlights**: 결과적으로, GenAI 모델은 고유의 규칙성을 따르면서도 인간 평가자와의 높은 신뢰도 일치를 보지 못했습니다. 특히, GPT-4o와 Gemma2-9b는 신뢰성 평가에서 우수한 성능을 보였지만, 언어 특성과 같은 '하드' 기준에 의존한다는 점에서 한계가 있음을 보여줍니다. 따라서, 이러한 모델에 대한 전적인 의존은 경계해야 하며 하이브리드 인간-인공지능 접근 방식이 권장됩니다.



### What Is a Good Caption? A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Coverage of MLLMs (https://arxiv.org/abs/2502.14914)
Comments:
          Work in progress

- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전으로 기존 비주얼 캡셔닝 기준이 뒤처지게 되었습니다. 기존 기준들은 주로 짧은 설명과 오래된 메트릭으로 한정되어 있었습니다. 이 문제를 해결하기 위해 CV-CapBench라는 종합적인 비주얼 캡션 기준을 제안하며, 6개 관점과 13개 차원에 걸쳐 캡션 품질을 체계적으로 평가합니다.

- **Technical Details**: CV-CapBench는 각 차원에 대해 정확도(accuracy), 재현율(recall), 적중률(hit rate) 메트릭을 도입하여 결과의 정확성과 coverage을 독특하게 평가합니다. 특히, 동적 및 지식 집약적 차원에서 MLLMs의 성능 격차가 뚜렷하게 나타났습니다. 이 구성이 정적(static) 차원 9개와 동적(dynamic) 차원 4개로 나뉘어지며, 직관적으로 시각적 캡션의 포괄성을 강화하고 있습니다.

- **Performance Highlights**: CV-CapBench를 활용한 실험 결과, 여러 주요 MLLMs가 여전히 특정 차원에서 성과를 내기 어려운 것으로 나타났습니다. 이러한 발견은 향후 비주얼 캡셔닝 개선을 위한 실질적인 통찰을 제공합니다. 논문에서 제안된 코드는 공개될 예정으로, 심도 있는 연구가 기대됩니다.



### KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models (https://arxiv.org/abs/2502.14908)
- **What's New**: 이번 연구에서는 Vision Language Models (VLMs)의 멀티모달 환경에서의 지식 갈등에 대한 영향을 조사하기 위해 \

- **Technical Details**: \

- **Performance Highlights**: \



### Revisiting Financial Sentiment Analysis: A Language Model Approach (https://arxiv.org/abs/2502.14897)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 연구는 전통적인 금융 감정 분석(Financial Sentiment Analysis, FSA)의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 기존의 인간 주석에 의존하는 감정 레이블 대신, 시장에서 직접 유도된 레이블을 사용하여 텍스트 신호와 시장 동학 간의 관계를 명확히 합니다. 연구 결과는 여러 시장 상황에서 금융 트윗의 효과적인 예측이 가능하다는 것을 보여주며, 이러한 패러다임 전환은 금융 의사 결정에 있어 언어 모델의 미개척 가능성을 강조합니다.

- **Technical Details**: 연구에서는 짧은 시간 가격 경향을 바탕으로 트윗 레이블을 지정하는 시장 기반 레이블링 방법을 제안합니다. 이 방법은 Triple Barrier Labeling (TBL) 접근법을 활용하여 상한, 하한, 그리고 시간적 제약을 통해 시장 반응을 그대로 반영합니다. 또한, CryptoBERT 및 FinBERT와 같은 도메인 특화된 언어 모델을 기반으로 하여, 시장 정보를 효과적으로 통합하는 프롬프트 튜닝 기술로 모델의 예측 정확성을 향상시킵니다.

- **Performance Highlights**: 본 연구에서 제안한 언어 모델은 전통적인 감정 기반 기준선에 비해 단기 추세 예측 정확도가 최대 11% 향상되었으며, 특정 비트코인 관련 뉴스 사건에 대해 89.6%의 높은 정확도를 기록했습니다. 일일 트윗 예측을 통한 거래 신호 생성이 전통적인 융합 모델에 비해 우수한 결과를 보여주었고, 여러 시장 환경에서 샤프 비율이 각각 5.07 및 3.73에 달했습니다. 이러한 결과는 언어 모델이 효과적인 단기 시장 예측 도구로 사용될 수 있음을 증명합니다.



New uploads on arXiv(cs.IR)

### Dynamic Knowledge Selector and Evaluator for recommendation with Knowledge Graph (https://arxiv.org/abs/2502.15623)
- **What's New**: 최근 추천 시스템(Recommendation Systems)은 추천 분야에서 지식 그래프(Knowledge Graph)에서 제공하는 엣지 정보와 그래프 네트워크(Graph Networks)의 고차 연결성(high-order connectivity) 장점을 통합하고 있습니다. 그러나 이러한 방법은 라벨(Label)의 희소성(sparsity) 문제와 그래프 구조(Graph Structure)를 잘 배우지 못하는 한계가 있으며, 지식 그래프 내의 많은 노이즈 엔티티(noisy entities)들이 추천 결과의 정확성에 영향을 미칠 수 있습니다.

- **Technical Details**: 이러한 문제를 완화하기 위해 우리는 협업 신호(collaborative signals)에 의해 안내되는 동적 지식 선택 및 평가 방법(Dynamic Knowledge-Selecting and Evaluating Method)을 제안합니다. 구체적으로, 체인 루트 평가기(Chain Route Evaluator)를 사용하여 추천 작업을 위한 다양한 이웃(neighborhood)의 기여도를 평가하고, 평가 전에 정보가 적은 지식을 필터링하는 지식 선택기(Knowledge Selector) 전략을 채택하였습니다.

- **Performance Highlights**: 세 가지 공개 데이터셋(public datasets)에서 기준 모델(baseline model) 비교 및 실험적 배제 평가(experimental ablation evaluations)를 실시한 결과, 제안한 모델이 현재의 최신 기준 모델을 뛰어넘는 성능을 기록했습니다. 또한, 각 모듈의 효과는 배제 실험(ablation experiments)을 통해 입증되었습니다.



### Cross-Format Retrieval-Augmented Generation in XR with LLMs for Context-Aware Maintenance Assistanc (https://arxiv.org/abs/2502.15604)
- **What's New**: 이번 논문은 Retrieval-Augmented Generation (RAG) 시스템의 상세 평가를 통해 대형 언어 모델(LLMs)이 정보 검색 및 지침 생성을 어떻게 개선할 수 있는지를 조명합니다. 8개의 LLM의 성능을 평가하였고, 응답 속도와 정확성과 같은 주요 지표를 BLEU 및 METEOR 점수를 통해 정량화한 결과 GPT-4와 GPT-4o-mini 같은 고급 모델이 복잡한 쿼리에 대해 현저히 우수하다는 사실을 밝혀냈습니다. 이는 RAG 프레임워크가 유지보수 작업을 최적화할 수 있는 잠재력을 가진 것을 의미합니다.

- **Technical Details**: LLMs는 인간 언어를 이해하고 생성하는 데 있어 큰 도약을 이루었으며, 이는 깊은 학습(Deep Learning) 구조를 기반으로 방대한 데이터 세트를 활용하여 복잡한 언어 패턴을 잡아내는 능력에 기인합니다. 본 논문에서는 다양한 데이터 형식에서 정보를 처리할 수 있는 RAG 아키텍처의 크로스 포맷 기능을 소개하며, 텍스트 문서, CSV 파일, 데이터베이스 등 다양한 파일 형식에서 정보를 효율적으로 검색할 수 있는 방법을 제시합니다. 이 시스템은 시나리오에 따라 복잡한 쿼리를 처리하고 이를 통해 얻어진 다양한 지식 기반에서 정보를 통합할 수 있는 기능도 갖추고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면 RAG 아키텍처는 사용자가 복잡한 maintenance 요청을 할 경우 다양한 출처에서 동시에 정보를 검색할 수 있는 능력을 보여줍니다. 이를 통해 사용자는 매뉴얼, 부품 목록 및 기술 사양과 같은 포괄적인 정보를 효율적으로 접근할 수 있게 됩니다. 최종적으로, RAG 시스템은 실제 환경에서의 운영과 관련된 다양한 시나리오에서 응답의 속도와 정확성을 획기적으로 향상시켜, 유지보수 인력에게 즉각적이고 포괄적인 지원을 제공합니다.



### Bridging Domain Gaps between Pretrained Multimodal Models and Recommendations (https://arxiv.org/abs/2502.15542)
- **What's New**: 최근 온라인에서 다중 모드 콘텐츠의 빠른 성장이 이루어짐에 따라, 사전 학습된 비주얼-랭귀지 모델이 다중 모드 추천에서 큰 잠재력을 보이고 있습니다. 기존의 프리사인 모델들은 리소스적으로 비효율적이며, 도메인 간의 갭으로 인해 성능 저하를 야기합니다. 이러한 문제를 해결하기 위해 저자들은 PTMRec (Parameter-efficient Tuning for Multimodal Recommendation)라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: PTMRec는 두 단계의 파라미터 효율적인 훈련 전략을 통해 도메인 간의 갭을 줄입니다. 첫 번째 단계는 사용자 선호 및 아이템 특성에 대한 작업 고유의 지식을 캡처하기 위해 경량 추천 모델을 학습하는 것으로 구성됩니다. 두 번째 단계에서는 개인화된 선호 지식을 활용하여 사전 학습된 모델의 튜닝을 안내하며, 지식 전송 최적화를 통해 진행됩니다.

- **Performance Highlights**: 이 프레임워크는 고가의 추가 사전 학습 없이도 다양한 파라미터 효율적인 튜닝 방안을 유연하게 수용하며, 추천 목표와 피처 추출 간의 결합을 유지합니다. PTMRec는 기존의 방법들과 비교하여 추천 시스템의 성능을 개선하면서 계산 효율성을 유지하는데 중점을 둡니다. 이 연구는 추천 시스템의 발전에 기여할 수 있는 중요한 발견입니다.



### Scaling Sparse and Dense Retrieval in Decoder-Only LLMs (https://arxiv.org/abs/2502.15526)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 스케일링이 실제 검색 모델 성능 향상에 미치는 영향을 분석하고 있습니다. 이전 연구들은 주로 대비 손실(Contrastive Loss, CL)로 훈련된 밀집 검색(dense retrieval)에 중점을 두었으나, 이 연구는 희소 검색(sparse retrieval) 및 지식 증류(Knowledge Distillation, KD)와 같은 다른 검색 패러다임 및 최적화 기법의 스케일링 행동도 탐구합니다. 중요 발견 중 하나는, 희소 검색 모델이 밀집 검색 모델보다 일관되게 우수한 성능을 발휘한다는 점입니다.

- **Technical Details**: 실험은 MSMARCO 데이터 세트를 사용하여 LLaMA3 시리즈(1B, 3B, 8B) 모델을 백본으로 하여 진행되었습니다. 이때, 다양한 훈련 구성 및 세팅을 사용하여 내-도메인(MSMARCO, TREC DL)과 외-도메인(BEIR) 테스트에서 평가했습니다. 특히, 희소 검색 모델은 CL과 KD 손실의 조합을 통해 8B 스케일에서 최첨단(SOTA) 결과를 달성함으로써 더 나은 성능을 입증했습니다.

- **Performance Highlights**: 희소 검색 모델은 모든 평가 벤치마크에서 밀집 검색 모델을 능가하였고, 특히 외-도메인 평가에서 강한 일반화 성능을 보여주었습니다. CL 손실만을 사용하는 경우, 대규모 모델에서는 성능 향상이 뚜렷하게 나타났지만, KD로 훈련된 모델에서는 미미한 개선만을 보였습니다. 논문의 주된 결론은, 희소 검색 모델이 밀집 검색 모델을 일관되게 능가하며, CL과 KD를 결합했을 때 최고의 성능을 달성했다는 점입니다.



### A Universal Framework for Compressing Embeddings in CTR Prediction (https://arxiv.org/abs/2502.15355)
Comments:
          Accepted by DASFAA2025

- **What's New**: 이번 연구에서는 온라인 광고와 추천 시스템에서 클릭률(CTR) 예측의 정확성을 높이는 방법을 제시합니다. 기존의 embedding layer 최적화 방식이 간과되었던 점을 강조하며, 제안된 Model-agnostic Embedding Compression (MEC) 프레임워크는 사전 훈련된 임베딩을 양자화하여 embedding tables을 압축합니다. 이 과정에서 추천 품질을 포기하지 않고도 메모리 사용량을 50배 이상 줄이는 효과를 보여줍니다.

- **Technical Details**: MEC 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 인기도 기반 가중치 정규화(popularity-weighted regularization, PWR)를 적용하여 고주파 및 저주파 특성 간의 코드 분배를 균형있게 조절합니다. 이어서 대조 학습(contrastive learning) 메커니즘을 통합하여 양자화된 코드의 고른 분포를 보장하고, 임베딩의 독특함(distinctiveness)을 강화합니다.

- **Performance Highlights**: 본 연구 결과는 세 가지 데이터 세트에서 MEC라는 접근 방식이 기존의 모델들과 비교했을 때 추천 성능을 유지하거나 개선할 뿐만 아니라 메모리 사용량을 50배 이상 줄임을 입증했습니다. 이는 고주파와 저주파 특성을 모두 활용하여 코드 할당의 불균형을 해결하고, 양자화 코드의 품질을 향상시키는 새로운 방법론을 제공하고 있습니다.



### Lightweight yet Efficient: An External Attentive Graph Convolutional Network with Positional Prompts for Sequential Recommendation (https://arxiv.org/abs/2502.15331)
Comments:
          26 pages, 8 figures, journal paper, accepted by TOIS at 20th February, 2025

- **What's New**: 본 논문은 External Attentive Graph convolutional network with Positional prompts for Sequential recommendation (EA-GPS)을 제안합니다. EA-GPS는 사용자와 아이템 간의 상호작용 및 아이템 간의 순차적 관계를 효과적으로 처리하는 그래프 기반의 시스템입니다. 이 시스템은 외부 메모리 유닛을 통해 글로벌 연관성을 선형으로 측정하고, 포지션 프롬프트 기반의 디코더를 통해 복잡한 위치 종속성을 명시적으로 다룹니다.

- **Technical Details**: EA-GPS는 두 개의 외부 메모리 유닛을 통한 외부 에디터(External Attention, EA)를 포함하여, 아이템의 절대 위치를 프롬프트로 간주합니다. 또한, 길이 적응형 순차 마스킹(length-adaptive sequential masking)과 소프트 어텐션 네트워크를 채택하여 모델이 장기적인 위치 종속성과 문맥 관계를 포착할 수 있도록 지원합니다. 이러한 기술적 접근은 그래프 인코더의 높은 계산 복잡도를 완화합니다.

- **Performance Highlights**: 다섯 개의 실제 데이터셋에서 EA-GPS의 성능을 평가한 결과, 기존의 선진 모델들에 비해 우수한 성능이 입증되었습니다. 특히, EA-GPS는 더 적은 파라미터 수와 낮은 학습 오버헤드로 뛰어난 성능을 유지하며, 자원 제약이 있는 엣지 디바이스에서도 효율적으로 사용할 수 있도록 설계되었습니다.



### From Documents to Dialogue: Building KG-RAG Enhanced AI Assistants (https://arxiv.org/abs/2502.15237)
- **What's New**: Adobe Experience Platform AI Assistant는 조직이 데이터와 원활하게 상호작용할 수 있도록 배치된 대화형 도구입니다. 그러나 내부 문서에 접근할 수 없는 한계로 인해 기존의 Large Language Models (LLMs)는 정확한 zero-shot 응답을 생성하는 데 장애가 있었습니다. 이를 극복하기 위해 Retrieval-Augmented Generation (RAG) 프레임워크와 Knowledge Graph (KG)를 활용하여 외부 지식 소스에서 관련 정보를 검색하고 있습니다.

- **Technical Details**: 이 논문에서는 고품질의 저소음 Knowledge Graph를 구축하기 위한 혁신적인 접근 방식을 제안합니다. 여러 기술을 통해 중복 항목을 제거하고 신뢰도 점수를 부여하는 방법을 통해 KG의 질을 유지하고, 문서의 출처를 함께 기록하여 정확한 정보 검색을 가능하게 합니다. KG-RAG 시스템은 사용자 프롬프트의 문맥에 관련된 튜플을 추가하여 LLM에 대한 응답 생성을 돕습니다.

- **Performance Highlights**: 우리의 평가 결과, 이 접근 방식은 기존 시스템에 비해 답변의 관련성을 크게 향상시켰습니다. 부적절한 답변은 50% 이상 감소하였고, 완전히 관련된 답변은 88% 증가하였습니다. 이러한 성과는 AI 비서가 고객 지원 등 다양한 사례에서 더욱 효과적으로 적용될 수 있음을 시사합니다.



### A BERT Based Hybrid Recommendation System For Academic Collaboration (https://arxiv.org/abs/2502.15223)
Comments:
          International Conference on Intelligent Systems and Security - 2024

- **What's New**: 이번 논문은 대학 내에서 학문적 협력을 촉진하기 위한 새로운 프로필 추천 시스템인 'FindMate'를 제안합니다. 이 시스템은 학생과 교수 간의 원활한 상호작용을 돕기 위해 TF-IDF와 BERT 기술을 결합하여 사용자와 유사한 이해관계를 가진 인물들을 연결하는 데 중점을 둡니다. 또한, 이 모델은 사용자 개인의 기술과 협력 관심사를 기반으로 프로필을 동적으로 추천하는 모바일 애플리케이션으로 구현되었습니다.

- **Technical Details**: 제안된 솔루션은 5단계로 구성되며, 첫째로 설문 조사를 통해 사용자 정보를 수집합니다. 수집된 데이터는 Natural Language Toolkit (NLTK)을 사용하여 전처리된 후 MongoDB에 저장됩니다. TF-IDF 및 코사인 유사성을 바탕으로 추천 알고리즘을 구축하며, Affinity Propagation 방법으로 비슷한 프로필들을 클러스터링합니다. BERT 아키텍처를 통해 입력된 프로필 데이터를 효과적으로 토큰화하고 임베딩 생성 후 코사인 유사성을 계산합니다.

- **Performance Highlights**: 하이브리드 모델은 실험 결과에서 높은 유사성 점수, 실루엣 점수, Davies-Bouldin 지수, Normalized Discounted Cumulative Gain (NDCG)를 기록하며 다양성과 관련성을 균형 있게 유지합니다. 또한, 이 모델은 기계 학습 및 자연어 처리를 통해 보다 진화된 추천 기능을 제공하여 대학 내 네트워킹 기회를 향상시킬 것으로 기대됩니다.



### GNN-Coder: Boosting Semantic Code Retrieval with Combined GNNs and Transformer (https://arxiv.org/abs/2502.15202)
- **What's New**: 새로운 연구 결과로, GNN-Coder라는 GNN 기반의 코드 검색 프레임워크가 제안되었습니다. 이 프레임워크는 코드의 구조적 의존성을 활용하여 코드 조각 간의 검색 성능을 개선합니다. 특히, GNN과 Transformer의 통합을 통해 코드의 구문적 및 의미적 특성을 포착하는 방법을 처음으로 탐구하였습니다.

- **Technical Details**: GNN-Coder는 Graph Neural Network (GNN)를 기반으로 하며, 추상 구문 트리(Abstract Syntax Tree, AST)를 활용하여 코드를 인코딩합니다. 특히, 새로운 그래프 풀링 방법(ASTGPool)을 제안하여 AST 내에서의 토폴로지적 관계를 강조하며 정보의 전파를 가속화합니다. Mean Angular Margin (MAM)이라는 새로운 메트릭을 소개해 코드 임베딩의 일관성을 정량화합니다.

- **Performance Highlights**: 실험 결과, GNN-Coder는 CSN 데이터셋에서 MRR이 1%에서 10% 향상되었고, CosQA 데이터셋에서는 제로샷 성능이 20% 향상되었습니다. 이는 GNN-Coder가 코드 조각 간의 구별 능력을 크게 증가시켜 코드 검색의 정확성을 높이는 데 기여하고 있음을 보여줍니다.



### Is Relevance Propagated from Retriever to Generator in RAG? (https://arxiv.org/abs/2502.15025)
Comments:
          18 pages (including reference), 5 figures, 1 table, 48 references; this paper has been accepted by ECIR'25 as a full paper

- **What's New**: 이번 연구에서는 Retrieval Augmented Generation (RAG) 프레임워크를 통해 대규모 언어 모델(LLM)이 보다 정확한 답변을 생성하도록 돕는 방법을 제안합니다. 기존의 연구들과는 달리, 정보 탐색 작업에서 질의와 문서 간의 주제적 연관성을 중점적으로 다루었습니다. RAG 시스템의 성능을 향상시키기 위해, 주제적으로 관련된 문서들을 효과적으로 활용하는 방법을 empirically(경험적으로) 조사하였습니다.

- **Technical Details**: RAG 시스템의 핵심 목표는 문서의 유용성을 최대화하는 것으로, 이는 질문에 대한 정보 수요를 충족하기 위해 LLM에 입력되는 컨텍스트(context)를 포함합니다. 연구에서 제안된 실험설계는 IR(Information Retrieval) 테스트 컬렉션을 활용하여, 추출된 데이터가 RAG 시스템의 생성 단계에서 정보의 질을 높이는 데 어떤 영향을 미치는지를 평가합니다. 이 때, 관련성과 유용성(utility)의 상관관계를 파악하고, 이는 0-shot 답변 품질에 대한 상대적인 이익을 측정하는 데 중점을 두었습니다.

- **Performance Highlights**: 실험 결과, 문서의 관련성과 유용성 간에 약한 긍정적 상관관계가 발견되었습니다. 그러나 이 상관관계는 컨텍스트 크기가 증가함에 따라 감소하며, LLM의 입력 문서 순서에는 상대적으로 민감하지 않은 것으로 나타났습니다. 또한, 더 효과적인 검색 모델이 일반적으로 RAG 성능을 개선하는 데 기여함을 입증했습니다.



### LightThinker: Thinking Step-by-Step Compression (https://arxiv.org/abs/2502.15589)
- **What's New**: 이번 논문에서는 LightThinker라는 새로운 방법을 제안합니다. 이 방법은 대형 언어 모델(LLMs)이 추론 중 중간 사고를 동적으로 압축할 수 있게 해줍니다. 사람의 인지 과정에서 영감을 받아, 경량화된 표현으로 불필요한 추론 단계를 제거함으로써 메모리 사용량과 연산 비용을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: LightThinker는 LLM이 언제 어떻게 압축을 수행할지 학습하도록 훈련시킵니다. 데이터 구성, 숨겨진 상태(hidden states) 매핑, 전문적인 주의 마스크(attention masks)를 사용하여 압축 로직을 조정하고, Dependency (Dep) 메트릭을 도입하여 압축 정도를 정량화합니다. 이 메트릭은 생성된 각 토큰이 얼마나 많은 역사적 토큰에 의존하는지를 측정합니다.

- **Performance Highlights**: 실험 결과, LightThinker는 Qwen 모델에서 피크 토큰 사용량을 70% 줄이고, 추론 시간을 26% 감소시키며, 정확도를 크게 저하시키지 않고(1% 감소) 효율성을 높였습니다. 이는 복잡한 추론 작업에서 LLM의 효율성을 향상시킬 수 있는 새로운 방향을 제시합니다.



### Detecting Future-related Contexts of Entity Mentions (https://arxiv.org/abs/2502.15332)
- **What's New**: 이 논문은 자동으로 엔티티가 미래 문맥에서 언급되었는지를 식별하는 기술에 대해 다루고 있으며, 이를 위해 새로운 데이터셋인 19,540개의 문장을 소개합니다. 해당 데이터셋은 Wikipedia에서 수집한 인기 있는 엔티티를 기반으로 하여 미래 관련 및 비미래 관련 문맥으로 구성됩니다. 이러한 작업은 정보 처리에서 자동화된 시간 분석에 대한 증가하는 요구를 해결하기 위한 것으로, 기존 연구의 한계를 보완하고자 합니다.

- **Technical Details**: 논문에서는 엔티티 중심 텍스트에서 미래 언급을 감지하고 분류하는 접근 방식을 제시합니다. 여기서는 기존의 명시적인 시간 표현에만 집중했던 전통적 방법 대신, 문맥적 단서를 기반으로 미래 지향적 콘텐츠를 식별하는 데 중점을 두었습니다. 데이터셋 구축 과정에서 BERT 모델을 활용하여 미래 관련 문장을 분류하고, 다양한 기계 학습 모델과 최신 Large Language Model의 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 전통적인 기계 학습 방법 및 transformer 기반 모델 모두를 통해 미래 관련 문맥을 구별하는 성능이 확인되었습니다. 특히, BERT와 같은 최첨단 모델들은 미래 지향적 내용의 식별에서 우수한 성능을 보였으며, LLMs의 활용 가능성도 탐구되었습니다. 이러한 성과는 미래 예측 및 관련 정보 검색 기술의 발전에 기여할 것으로 예상됩니다.



### RAPTOR: Refined Approach for Product Table Object Recognition (https://arxiv.org/abs/2502.14918)
Comments:
          Accepted for WACVW 2025 (VisionDocs)

- **What's New**: 이번 연구에서는 RAPTOR라는 모듈형 후처리 시스템을 도입하여 기존의 테이블 추출 기술, 특히 제품 테이블에 대한 성능을 향상시키고자 하였습니다. 기존 DEtection TRansformer (DETR)와 TAble TRansformer (TATR) 기반의 모델들이 직면한 다양한 테이블 형식의 문제들을 해결하고, 정밀도와 구조 예측 모두에서 개선을 도모합니다. RAPTOR는 특히 이커머스와 관련된 비즈니스 문서 분석에 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: RAPTOR는 Genetic Algorithm을 활용하여 모듈의 매개변수를 최적화하고, ICDAR 2019 및 다양한 공개 데이터셋에서 사전훈련된 DETR과 TATR 모델을 사용하여 테이블 영역 감지(Table Detection, TD) 및 테이블 구조 인식(Table Structure Recognition, TSR)을 수행합니다. 이 모듈형 시스템은 기존 모델을 재조정하는 대신, 제한된 데이터로 모델 매개변수를 학습할 수 있는 기회를 제공합니다. 기존의 문제 해결을 위한 컴포넌트를 통합함으로써 TD 및 TSR 과정에서의 성능을 향상시킵니다.

- **Performance Highlights**: RAPTOR는 다섯 개의 데이터셋에서 평가되었으며, 특히 제품 테이블에서의 성능 향상이 두드러졌습니다. 다양한 테이블 형식에서도 전반적으로 합리적인 성능을 유지했으며, 여러 데이터셋에 따라 기본 모델의 성능을 개선하는 데 기여했습니다. 연구팀은 비즈니스 테이블에서 나타나는 일반적인 오류 유형을 식별하고 이를 해결하기 위한 모듈형 시스템으로서 RAPTOR의 유용성을 확인하였습니다.



### OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignmen (https://arxiv.org/abs/2502.14913)
Comments:
          15 pages

- **What's New**: 최근 다중 에이전트 협업을 이용한 대형 언어 모델(Large Language Models, LLMs)이 Text-to-SQL 작업에서 눈에 띄는 성과를 올리고 있는 가운데, OpenSearch-SQL이라는 새로운 방법론이 제안되었습니다. 이 방법론은 전체 텍스트-투-SQL 작업을 네 가지 주요 모듈인 전처리(Preprocessing), 추출(Extraction), 생성(Generation), 개선(Refinement) 및 일관성 정렬(Alignment) 모듈로 나누어 운영합니다. 특히, 이 구조는 에이전트의 입력과 출력을 정렬하여 지침을 따르지 못하거나 해리(hallucination) 문제를 줄이는 데 기여합니다.

- **Technical Details**: OpenSearch-SQL에서는 텍스트-투-SQL 작업을 인간의 SQL 작성 프로세스를 모델링한 표준 프로세스에 기반하여 정의합니다. 이 과정은 전처리, 추출, 생성 및 개선의 네 가지 단계로 구성되어 있으며, 요청을 이해하고 SQL을 완성하는 데 필요한 정보를 체계적으로 처리합니다. 또한, SQL-Like라는 중간 언어를 개발하여 모델이 SQL의 구조를 효율적으로 생성하도록 돕고, 자가 학습 기반의 Query-CoT-SQL 형태의 동적 몇 샷(few-shot) 전략을 설계하였습니다.

- **Performance Highlights**: OpenSearch-SQL의 실험 결과, BIRD 개발 세트에서 69.3%의 실행 정확도(EX)를 달성하며, 테스트 세트에서는 72.28%의 성능을 기록하였습니다. 또한, 보상 기반 유효성 점수(R-VES)는 69.36%로, 제출 시 모든 지표에서 1위를 기록하였습니다. 이러한 결과는 제안된 방법이 효과성과 효율성 모두에서 상당한 장점을 가지고 있음을 잘 보여줍니다.



### PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths (https://arxiv.org/abs/2502.14902)
- **What's New**: 이 논문에서는 기존의 그래프 기반 RAG(유도 검색 증강 생성) 방법의 한계를 지적하며, 정보의 중복성이 오히려 문제라고 주장합니다. PathRAG라는 새로운 접근 방식을 제안하여, 텍스트 정보를 인덱싱 그래프 형태로 구조화하여 응답 생성의 질을 향상시키려 합니다. 이 시스템은 키 관계 경로를 효과적으로 검색하고, 이를 텍스트 형태로 변환하여 LLM(대형 언어 모델)에 제공함으로써 더 논리적이고 일관적인 답변을 생성할 수 있도록 합니다.

- **Technical Details**: PathRAG는 플로우 기반 가지치기(flow-based pruning)를 통해 중복 정보를 줄이고, 각 쿼리 키워드에 대해 인덱싱 그래프에서 관련 노드를 검색할 수 있는 알고리즘을 사용합니다. 이 방법은 노드 쌍 사이의 키 관계 경로를 식별하여 불필요한 정보 노이즈를 줄이는 동시에, 경로의 신뢰도 점수를 부여하여 보다 효율적으로 정보를 처리합니다. 마지막으로, 각 경로에 따라 노드와 에지 정보를 텍스트 관계 경로로 연결하여 LLM 프롬프트에 통합합니다.

- **Performance Highlights**: PathRAG는 실험적으로 6개의 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보여주며, 모든 평가 차원에서 더 나은 결과를 도출했습니다. GraphRAG와 LightRAG에 비해 평균 승률이 각각 60.44% 및 58.46%로 나타났습니다. 특히, PathRAG의 성능이 더 큰 데이터셋에서 더욱 두드러지며, 실제 응용에 보다 적합한 결과를 제공합니다.



### Retrieval-augmented systems can be dangerous medical communicators (https://arxiv.org/abs/2502.14898)
Comments:
          Preprint

- **What's New**: 이 논문에서는 환자들이 건강 정보를 찾기 위해 생성형 AI를 점점 더 많이 이용하는 추세를 반영하고 있다. 특히 의료 분야에서 AI가 생성한 답변의 정확성을 높이기 위해 retrieval-augmented generation(RAG) 및 citation grounding과 같은 기법이 사용되고 있지만, 이러한 방법들이 오히려 환자들에게 혼란을 줄 수 있다는 점을 강조한다. 저자들은 현재의 시스템이 환자의 질문을 문맥적으로 이해하지 못해 잘못된 인식을 강화할 수 있다고 주장한다.

- **Technical Details**: RAG 시스템은 환자의 질의에 대한 정확한 응답을 제공하기 위해 신뢰할 수 있는 원천을 참조하는 데 초점을 맞춘다. 하지만 이 논문은 AI가 생성한 응답이 원본 정보에서 얻은 사실들을 단순히 재구성하는 데 그칠 뿐, 핵심적인 정보나 맥락을 누락해 환자에게 혼란을 초래할 수 있음을 보여준다. 데이터 분석을 통해 Google AI Overview 및 Perplexity AI의 질의 응답 패턴을 조사하고 잘못된 해석을 일으킬 수 있는 주요 요소들을 확인했다.

- **Performance Highlights**: 대규모 데이터 분석 결과, Google AI Overview와 Perplexity AI 모두 환자 질의에 대해 비슷한 응답을 생성했지만, 특정 조건이나 증상 질의를 통해 생성된 응답이 원문을 바탕으로 하더라도 환자에게 잘못된 인식을 초래한다는 점이 드러났다. 특히, 질병이나 절차에 대한 안전성과 위험에 관한 질의에서 상반된 정보가 제공될 때, 두 시스템 간의 참조 자료의 유사성이 낮았고, 그로 인해 건강 불안감을 유발할 잠재성이 크다는 것을 알 수 있었다.



New uploads on arXiv(cs.CV)

### ELIP: Enhanced Visual-Language Foundation Models for Image Retrieva (https://arxiv.org/abs/2502.15682)
- **What's New**: 이 논문에서는 텍스트-이미지 검색의 성능을 향상시키기 위해 새로운 프레임워크인 Enhanced Language-Image Pre-training (ELIP)를 소개합니다. ELIP는 텍스트 쿼리를 사용하여 이미지 인코딩(ViT)을 조정하기 위한 시각적 프롬프트 집합을 예측합니다. 이 접근법은 널리 사용되는 CLIP/SigLIP 및 최신 BLIP-2 아키텍처에 쉽게 적용될 수 있습니다.

- **Technical Details**: 우리의 ELIP 방법은 가벼운 텍스트 유도 시각적 프롬프트 모듈을 도입하여 텍스트 쿼리를 시각적 프롬프트 벡터로 매핑합니다. 이 벡터는 이미지 인코더의 [CLS] 및 패치 임베딩과 연결된 후, 고정된 비전 인코더에 입력됩니다. 이렇게 해서 생성된 이미지 임베딩은 텍스트 조건을 인지하고, 재순위를 보다 효과적으로 수행합니다.

- **Performance Highlights**: 제안된 ELIP 모델은 코코(COCO) 및 플리커(Flickr)와 같은 표준 벤치마크에서 CLIP/SigLIP 성능을 크게 향상시키고 최신 BLIP-2 모델보다 우수한 성능을 보입니다. 또한 우리는 두 개의 새로운 벤치마크인 Occluded COCO 및 ImageNet-R을 설정하여 모델의 제로샷(generalisation) 일반화 능력을 평가하였습니다.



### VaViM and VaVAM: Autonomous Driving through Video Generative Modeling (https://arxiv.org/abs/2502.15672)
Comments:
          Code and model: this https URL, project page: this https URL

- **What's New**: 이 논문에서는 자율 주행을 위한 대규모 생성 비디오 모델의 잠재력을 탐구합니다. 우리는 오픈 소스 오토-회귀 비디오 모델(VaViM)과 동반 비디오-액션 모델(VaVAM)을 소개하여 비디오 프리트레이닝(Pre-training)이 실제 자율 주행에 어떻게 이전되는지를 조사합니다. VaViM은 공간-시간 시퀀스를 사용하여 프레임을 예측하는 간단한 오토-회귀 비디오 모델로, 주행 장면의 의미와 동역학을 캡처하는 것으로 확인되었습니다.

- **Technical Details**: VaViM은 공간-시간 토큰 시퀀스의 공동 분포를 모델링하여 미래 프레임을 예측하는 방법으로 학습합니다. 이미지 토크나이저를 사용하여 시각 정보를 이산 토큰으로 압축하여 각 비디오 프레임의 간결한 표현을 제공합니다. VaVAM은 VaViM의 학습된 비디오 표현을 활용하여 고수준 목표 및 시간적 맥락에 따라 주행 궤적을 생성하는 모듈로, 이를 통해 자율 주행 차량에서 효과적인 모션 계획 및 의사 결정을 지원합니다.

- **Performance Highlights**: 우리는 오픈 및 클로즈드 루프 주행 시나리오를 사용하여 모델을 평가하였으며, 비디오 기반 프리트레이닝이 자율 주행에 큰 잠재력을 가지고 있음을 발견했습니다. 학습된 표현은 풍부한 의미 정보를 포함하고 있으며, 일반적으로 더 큰 모델이 비디오 합성 품질을 개선하는 경향이 있습니다. 그러나 클로즈드 루프 평가에서의 안전 메트릭은 일관되게 개선되지 않아 궤적 추적과 적응적 의사 결정 간의 근본적인 갈등을 드러냅니다.



### Para-Lane: Multi-Lane Dataset Registering Parallel Scans for Benchmarking Novel View Synthesis (https://arxiv.org/abs/2502.15635)
- **What's New**: 이번 논문에서는 자율 주행 시스템을 평가하기 위한 새로운 다중 주행 데이터세트를 제안합니다. 기존의 Novel View Synthesis (NVS) 기법을 기반으로 하여, 특히 횡선 시나리오에서 차량의 새로운 포즈에 따라 포토리얼리스틱 이미지와 포인트 구름을 합성할 수 있는 시뮬레이션 환경이 필요하다는 점에 주목했습니다. 또한, 본 데이터세트인 Para-Lane은 현실의 스캔에서 유래된 최초의 다중 주행 관련 데이터로, 16,000개의 정면 이미지와 64,000개의 주변 이미지, 16,000개의 LiDAR 프레임을 포함하고 있습니다.

- **Technical Details**: 이 연구는 멀티 패스 데이터 수집 방식을 사용하여 실제 시나리오에서의 횡선 NVS 벤치마크를 위해 데이터셋을 구축하는 데 중점을 두고 있습니다. 두 가지 주요 문제점, 즉 IMU(관성 항법 시스템)와 외부 감지 센서 간의 정밀한 정렬과 카메라 프레임에서 LiDAR 프레임으로의 픽셀-포인트 매핑의 정확성을 해결하기 위해 2단계 포즈 최적화 메커니즘을 개발했습니다. 이는 시간적으로 그리고 공간적으로 외부 감지 센서의 데이터를 정렬하는 통합 프레임워크를 통해 이루어집니다.

- **Performance Highlights**: Para-Lane 데이터셋은 NeRF(신경 방사장 필드) 및 3DGS(3D 생성적 신경망)와 같은 최신 방법을 평가하는 벤치마크로 사용됩니다. 본 연구는 횡선 시점 변화를 포함해 자율 주행 시나리오에서 NVS 성능에 대한 귀중한 통찰력을 제공할 수 있습니다. 실험 결과는 기존 자율 주행 제품의 연구 및 개발을 가속화시킬 것으로 기대됩니다.



### RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes (https://arxiv.org/abs/2502.15633)
Comments:
          ICRA 2025

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS) 기반의 SLAM 방법인 OpenGS-SLAM을 소개합니다. 이 방법은 RGB 데이터만을 사용하여 무제한의 야외 장면을 효과적으로 처리하는 동시에, 카메라 포즈 추정을 위한 일관된 포인트맵(pointmap)을 생상합니다. 이를 통해 이전의 3DGS 방법보다 9.8% 낮은 추적 오류를 달성했습니다.

- **Technical Details**: OpenGS-SLAM은 포인트맵 회귀 네트워크(pointmap regression network)를 사용하여 서로 다른 프레임 간의 포인트맵을 생성합니다. 이 포인트맵은 장면의 지오메트리, 뷰포인트 관계, 2D-3D 대응관계를 포함하여 카메라 포즈 추정을 더 강력하게 만듭니다. 또한, 카메라 포즈 추정을 3DGS 렌더링과 통합하여 최적화된 end-to-end 파이프라인을 구축합니다.

- **Performance Highlights**: Waymo 데이터셋에 대한 실험 결과, OpenGS-SLAM은 기존 3DGS 방법보다 우수한 성능을 보이며, 새로운 노벨 뷰 합성(Novel View Synthesis) 벤치마크를 세웠습니다. 특히, 제안한 적응형 스케일 매퍼(adaptive scale mapper)와 동적 학습률 조정(dynamic learning rate adjustment) 전략이 시스템의 신뢰성과 추적 정확성을 크게 향상시켰습니다.



### Continual Person Identification using Footstep-Induced Floor Vibrations on Heterogeneous Floor Structures (https://arxiv.org/abs/2502.15632)
- **What's New**: 이번 연구에서는 스마트 빌딩에서 개인화된 서비스를 제공하기 위한 새로운 개인 식별 시스템을 제안합니다. 기존의 개인 식별 시스템은 사전에 수집된 데이터를 필요로 했지만, 이 연구는 실시간으로 사람들의 정체성을 점진적으로 학습하는 방식을 도입합니다. 특히, 발자국으로 유도된 구조 진동 데이터를 활용하여 개인의 신원을 식별하는 방법을 통해 비침입적인 접근 방식을 강조하고 있습니다.

- **Technical Details**: 연구에서는 발자국이 유도한 구조 진동 데이터의 변동성을 분석하여, 이를 기반으로 feature transformation function을 설계합니다. 이 과정은 사람 간의 데이터 분리를 향상시키고, 개개인의 데이터 내 변동성을 줄이는데 초점을 맞추고 있습니다. 또한, dirichlet process를 기반으로 한 비모수 베이지안 온라인 학습 접근 방식을 통해 신규 발자국 특성을 학습하고 전체 발자국 모델을 실시간으로 업데이트하는 방법을 제시합니다.

- **Performance Highlights**: 필드 실험을 통해 20명을 대상으로 한 결과, 발자국 데이터의 변동성을 70% 감소시키고, 온라인 개인 식별 정확도를 90%로 달성했습니다. 이 방식은 최소한의 데이터로도 효과적인 개인 식별을 가능하게 하여 스마트 빌딩의 개인화된 서비스 구현에 기여할 수 있는 가능성을 보여주고 있습니다.



### WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents (https://arxiv.org/abs/2502.15601)
- **What's New**: 최근 기술 발전으로 제작되는 3D 가상 세계는 영화, 게임, 혼합 현실 등 여러 분야에 활용되고 있지만, 기존의 3D 모델링 소프트웨어는 숙련된 전문가의 extensive labor(노동력)가 필요했습니다. 우리가 제안하는 WorldCraft 시스템은 대규모 언어 모델(LLM) 에이전트를 활용해 사용자가 자연어 명령을 사용하여 개별 객체의 속성과 장면 레이아웃을 조정할 수 있도록 하여, 3D 세계 제작을 민주화합니다.

- **Technical Details**: WorldCraft는 세 가지 주요 구성 요소를 통해 작동합니다. (1) ForgeIt: 사용자가 요청하는 대로 객체를 세밀하게 사용자 정의할 수 있게 돕는 수동적 검증 기능을 통해 매뉴얼을 동적으로 생성합니다. (2) ArrangeIt: 장면 배열을 계층적 최적화 문제로 설정하여 사용자 디자인 의도를 충족시키는 레이아웃을 수립합니다. (3) Trajectory Control Agent: 사용자가 객체와 카메라의 움직임을 자연어 대화를 통해 조작할 수 있게 해주어, 애니메이션을 생성할 수 있습니다.

- **Performance Highlights**: WorldCraft는 고급 3D 생성기와의 호환성을 통해 장면의 자산을 풍부하게 하며, 테스트를 통해 이 시스템의 유연성과 효율성을 검증했습니다. 평가 결과, WorldCraft는 단일 객체 맞춤화에서부터 복잡한 대규모 내부 및 외부 장면 설계에 이르기까지 다양한 응용 분야에서 효과적인 성과를 발휘했습니다. 이를 통해 비전문가도 자신의 창의적 비전을 실현하는 데 필요한 도구를 갖출 수 있게 되었습니다.



### Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation (https://arxiv.org/abs/2502.15563)
- **What's New**: AI 모델의 신뢰성 있는 평가 방법이 과학적 발전과 실제 응용을 위한 중요한 요소로 부각되고 있습니다. 기존의 VLM 벤치마크는 모델 능력에 대한 일반적인 통찰력을 제공하지만, 이질적인 설계와 특정 이미지 도메인에 국한된 접근 방식 때문에 여러 문제에 직면하고 있습니다. 이를 해결하기 위해, 저자는 리소스 효율적인 도메인 특화 VLM 벤치마크 생성 프레임워크와 함께 새로운 VLM 벤치마크를 제공하며, 22개의 최신 VLM 모델에 대한 광범위한 벤치마킹을 실시했습니다.

- **Technical Details**: 아이디어의 핵심은 단일 이미지로부터 다양한 과제를 생성할 수 있는 작업 증강(task augmentation) 프레임워크에 있습니다. 이 프레임워크는 인스턴스 분할(instance segmentation) 주석을 다양한 인식 과제들로 변환하여, 인식 능력을 테스트하는 여러 문제를 만듭니다. 저자들은 이 방법론을 적용하여 7개의 새 도메인 특화 VLM 벤치마크를 생성하고, 37,171개의 과제에 대해 22개의 모델을 포괄적으로 평가했습니다.

- **Performance Highlights**: 이 연구의 결과는 벤치마크의 설계가 도메인과 과제에 따라 성능 차이를 드러내며, 개인화된 벤치마크가 필요함을 시사합니다. 162,946개의 인간 검증 응답을 수집하여, 모델 평가의 강력한 기준점을 설정했습니다. 이로 인해 연구자들은 특정 도메인에 맞춤화된 모델 선택을 할 수 있는 길을 열고, 향후 연구 방향을 개선할 수 있습니다.



### Estimating Vehicle Speed on Roadways Using RNNs and Transformers: A Video-based Approach (https://arxiv.org/abs/2502.15545)
- **What's New**: 이번 프로젝트는 고급 기계 학습 모델, 특히 Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), 그리고 Transformers를 활용하여 비디오 데이터를 통한 차량 속도 추정 작업을 탐구합니다. 전통적인 속도 추정 방법들은 높은 비용과 제한된 범위로 인해 제약이 있는 반면, 기존의 감시 인프라와 최첨단 신경망 구조를 활용하여 비침습적이고 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: 이 연구는 LSTM 및 GRU 모델을 통해 비디오 프레임의 시간 순서 내에서 장기 의존성을 효과적으로 관리하며, Transformers는 자체 주의 메커니즘을 사용하여 전체 시퀀스를 병렬로 처리하며 데이터의 가장 유의미한 부분에 집중합니다. 입력 데이터의 시퀀스 길이를 늘리면 모델 정확도가 일관되게 향상되고, 특히 Transformers는 다양한 시퀀스 길이와 복잡성에서 뛰어난 적응력과 강건성을 보여주어 다양한 교통 조건의 실시간 애플리케이션에 적합합니다.

- **Performance Highlights**: 모델 결과는 LSTM과 GRU가 일반적인 Recurrent Neural Networks (RNNs)보다 뛰어난 성능을 보이며, 컨텍스트 정보를 중요시하는 동적 환경에서의 정확성이 향상됨을 보여줍니다. 이러한 정교한 신경망 모델의 통합은 자동화된 속도 감지 시스템의 정확성과 신뢰성을 크게 향상시킬 수 있으며, 이는 교통 관리 및 도로 안전의 혁신적인 발전을 전망하게 합니다.



### Depth-aware Fusion Method based on Image and 4D Radar Spectrum for 3D Object Detection (https://arxiv.org/abs/2502.15516)
- **What's New**: 이 논문에서는 4D 밀리미터파 레이더와 깊이 인식 카메라 이미지를 통합하여 전천후(다양한 날씨)에서 3D 객체 탐지 성능을 최적화하는 새로운 알고리즘을 제안합니다. 기존의 전통적인 3D 밀리미터파 레이더에서는 정보 손실 문제를 해결하기 위해 원시 레이더 스펙트라와 깊이 감지 데이터를 활용합니다. 또한, GAN 기반 네트워크를 통해 깊이 센서가 없는 경우 레이더 스펙트라에서 깊이 이미지를 생성하는 방법도 제안드립니다.

- **Technical Details**: 밀리미터파 레이더는 다양한 환경 감지에 매우 효과적이며, 본 연구에서는 4D 밀리미터파 레이더와 RGB 깊이 인식을 사용하는 카메라를 결합하여 다중 스케일 특징을 추출하여 융합하는 방식으로 구현되었습니다. BEV( Bird's Eye View) 특징 공간에서의 극좌표계(unpolar coordinates) 기반의 어텐션 매커니즘을 통해 이미지와 레이더 데이터를 효율적으로 결합하며, 이러한 접근 방식은 탐지 성능을 개선합니다. 또한, 본 연구는 특징을 증강하여 카메라-레이더 데이터 융합을 이루어냅니다.

- **Performance Highlights**: 본 연구의 알고리즘은 3D 객체 탐지에서 높은 성능을 보이며, 네트워크의 복잡성을 대폭 줄이는 성과를 이루었습니다. 기존 방식에 비해 전반적인 탐지 성능이 크게 향상되었으며, 다양한 기상 조건에서도 지속적으로 작동할 수 있는 능력을 보여주었습니다. K-Radar 데이터셋을 활용하여 다양한 환경과 도로 구조에서 평가되어 이러한 성능을 입증했습니다.



### Q-PETR: Quant-aware Position Embedding Transformation for Multi-View 3D Object Detection (https://arxiv.org/abs/2502.15488)
- **What's New**: 이 논문은 PETR 기반의 방법이 3D 인식에서 주목받고 있으며, 자율 주행 시스템의 주요 구성 요소로 자리잡고 있음을 강조합니다. 특히, INT8 추론을 요구할 때 성능 저하가 발생하는 문제를 해결하기 위해, Q-PETR이라는 새로운 방법을 제안합니다. 이 방법은 양자화에 유리하고 배치에 적합한 아키텍처를 제공하면서, 원래 PETR의 성능을 보존하는 데 중점을 두고 있습니다.

- **Technical Details**: Q-PETR는 원래 PETR의 양자화 실패 원인을 분석하여, 위치 인코딩의 큰 값과 불균형한 스케일된 도트 제품의 문제를 해결합니다. 양자화 전략을 재설계하고 비선형 함수 추론을 최적화하기 위한 DuLUT(lookup table)의 도입이 주요 특징입니다. 이는 양자화 후 최적의 성능을 구사하며, 다양한 PETR 시리즈 모델에서의 광범위한 일반성을 입증합니다.

- **Performance Highlights**: Q-PETR의 도입으로 INT8 추론의 정확도 손실이 1% 이하로 감소하며, FP32(부동 소수점 32비트) 성능을 초과할 수 있습니다. 다양한 실험을 통해 Q-PETR은 자율주행 차량의 엣지 AI 칩에서의 배치 준비 상태를 만족시킵니다. 이러한 성능 향상은 자율주행 시스템에서 3D 물체 감지의 효율성을 크게 개선하는 것으로 기대됩니다.



### Confidence-Based Annotation Of Brain Tumours In Ultrasound (https://arxiv.org/abs/2502.15484)
- **What's New**: 이번 논문에서는 초음파 영상에서 뇌종양의 이산 분할(annotation) 작업 시 발생하는 aleatoric uncertainty(우연적 불확실성) 문제를 다루고 있습니다. 특히 퍼지형 암종의 가장자리를 중심으로 하여 적정한 주석 방법론을 제안합니다. 기존의 이산 주석 방식 대신, 면적 기반 (spatial) 및 지역적 (local) 주석을 수행하는 접근 방식을 통해 관찰자 간의 변동성을 줄이고자 합니다.

- **Technical Details**: 제안된 방법론은 세 가지 신뢰도 수준(높음, 중간, 낮음)을 사용하는 sparse confidence method(희소 신뢰도 방법)에 기초하고 있습니다. 이 방법은 이미지 내의 객체를 특정 짓기 위해 scatter point(산포점)를 정의하며, 각각의 신뢰도 수준에 따라 heat map(히트 맵)을 구성합니다. 이를 통해, 주석할 때 발생하는 aleatoric uncertainty(우연적 불확실성)를 최소화하고, 주석자 간의 주관적 편차를 줄이는 것을 목적으로 합니다.

- **Performance Highlights**: 연구 결과, 제안된 주석 방법을 통해 생성된 주석 결과는 전문가 간의 이산 주석 변동성과 비교되었습니다. Pearson correlation(피어슨 상관계수) 값이 0.8로 측정되었으며, 신뢰도 주석을 soft label(소프트 레이블)로 사용한 네트워크 학습이 하드 레이블보다 우수한 Brier score(브라이어 점수)를 기록했습니다. 이러한 결과는 임상적으로 관련된 소프트 레이블을 통한 학습이 신경망의 조정성(calibration)을 향상시킬 수 있음을 나타냅니다.



### On Neural BRDFs: A Thorough Comparison of State-of-the-Art Approaches (https://arxiv.org/abs/2502.15480)
Comments:
          Published in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문에서는 Bidirectional Reflectance Distribution Function(BRDF) 모델링을 위한 다양한 신경망 접근 방법을 철저히 평가하고 비교합니다. 기존의 파라메트릭 모델과 순수 신경망 방법의 차이점을 분석하고, 특히 고도로 반사성이 있는 재료에 대한 순수 신경망 방법의 장점을 제시합니다. 더불어, BRDF의 상호성(reciprocity)을 보장하고, 반사율을 확실히 분리하는 새로운 두 가지 확장을 제안합니다.

- **Technical Details**: BRDF는 광선의 방향과 물체의 표면 질감과 상호작용하는 복잡한 물리적 원리를 설명하는 도구로, 신경 필드(neural fields)를 활용한 BRDF 모델링은 기존의 파라메트릭 모델에 의존하면서도 매개변수를 직접 예측하는 새로운 방법을 포함합니다. 연구진은 여러 신경망 접근 방식을 비교하고, BRDF를 구성하는 확산(diffuse)과 반사(specular) 부분을 분리하는 새로운 입력 매핑(input mapping) 기법을 개발하였습니다.

- **Performance Highlights**: 실험 결과에 따르면 순수 신경망 방법이 파라메트릭 모델에 비해 반사성이 높은 재료에 대한 성능이 우수하다는 것이 입증되었습니다. 또한, 제안된 확장 기능들이 기존 방법을 개선하는 데 기여하여, 에너지 보존 및 BRDF의 상호성 문제를 더욱 효과적으로 해결할 수 있도록 합니다. 연구에 대한 더 많은 정보와 비주얼 자료는 저자의 웹사이트에서 확인할 수 있습니다.



### CondiQuant: Condition Number Based Low-Bit Quantization for Image Super-Resolution (https://arxiv.org/abs/2502.15478)
Comments:
          10 pages, 5 figures. Code and models are released at this https URL

- **What's New**: 이 논문에서는 저비트 모델 양자화(quantization) 기법을 이용해 이미지 초해상도(super-resolution) 성능을 높이는 새로운 방법인 CondiQuant를 제안합니다. 이는 기존의 양자화 방법에서 발생하는 성능 저하 문제를 해결하기 위해 조건 수(condition number)를 활용합니다. 실험 결과, CondiQuant는 기존의 최신 기술과 비교하여 정확도를 높이면서도 추가적인 연산 비용 없이 최적의 압축 비율을 달성했습니다.

- **Technical Details**: CondiQuant는 이미지 초해상도를 위해 모델 무게의 조건 수를 기반으로 한 후 훈련 양자화 방법입니다. 이 방법은 양자화 감도를 줄이고 출력 결과를 유지하면서 무게 행렬의 조건 수를 최소화하기 위해 효율적인 근접 경량(descendent 방법을 설계합니다. 논문에서는 조건 수 개념이 양자화 오류를 줄이는 데 있어 중요한 역할을 한다고 설명합니다.

- **Performance Highlights**: Comprehensive experiments using CondiQuant show significant improvements in restoration accuracy compared to state-of-the-art post-training quantization methods. In addition, extensive ablation studies demonstrate the robustness and effectiveness of the proposed method, confirming that it achieves industry-leading performance without increasing computational overhead.



### Game State and Spatio-temporal Action Detection in Soccer using Graph Neural Networks and 3D Convolutional Networks (https://arxiv.org/abs/2502.15462)
- **What's New**: 이 논문은 축구 경기에서의 스페이쇼-템포랄(action detection) 이벤트 감지를 향상시키기 위해 그래프 신경망(Graph Neural Networks, GNN)을 활용하는 방식을 제안합니다. 점점 증가하는 데이터 수집 능력과 deep learning의 발전에도 불구하고, 축구 비디오에서의 정밀한 이벤트 주석(annotation)은 여전히 수작업으로 수행됩니다. 저자들은 주변 선수들의 정보(위치, 속도, 팀 소속 등)를 결합하여 시각적 예측을 향상시킬 수 있다고 가정하고 있습니다.

- **Technical Details**: 논문은 Track-Aware Action Detector(TAAD) 방법에 기반하여 게임 상태 정보를 통합한 새로운 spatio-temporal 액션 감지 방법을 개발했습니다. GNN을 사용해 각 관측된 선수에 대한 지역 게임 상태 정보를 포함시키며, 3D CNN 구조를 통해 end-to-end 방식으로 학습됩니다. 이러한 접근 방식은 비디오에서의 이벤트 감지를 더욱 신뢰성 있게 만듭니다.

- **Performance Highlights**: 실험 결과, 구조화된 게임 상태 정보를 명시적으로 통합함으로써 액션 감지 모델의 성능이 향상된 것을 보여줍니다. 저자들은 이 연구의 실제 적용 가능성을 고려하여 트레이닝을 입문 수준의 GPU에서 수행할 수 있도록 제한된 계산 비용으로 구현되었습니다. 이러한 성과는 축구 분석을 통해 더 나은 전술 결정 및 상대 분석을 가능하게 할 것입니다.



### Memory Helps, but Confabulation Misleads: Understanding Streaming Events in Videos with MLLMs (https://arxiv.org/abs/2502.15457)
Comments:
          Short paper (5 pages)

- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)가 비디오 이벤트를 이해하는 방식에 있어 메모리 활용의 중요성을 보여줍니다. 비디오가 일련의 시각적 이벤트로 처리되는 상황에서, 이전의 이벤트를 메모리로 사용함으로써 현 이벤트에 대한 맥락적 이해를 향상시킬 수 있다는 새로운 인사이트를 제공합니다. 하지만, 이러한 메모리가 잘못된 정보에 의존할 경우 혼란(confabulation)을 초래하여 성능 저하가 발생하는 문제를 다룹니다.

- **Technical Details**: 연구에서는 LLMs에서 메모리를 어떻게 구성하고 활용하는지를 설명하고, 이 메모리가 과거의 사건과 현재 사건 간의 맥락적 관계를 어떻게 강화하는지를 다룹니다. 특히, 두 가지 유형의 메모리, 즉 장기 기억(long-term memory)과 단기 기억(short-term memory)을 도입하고, 메모리의 정확성을 개선하기 위한 confabulation-aware memory modification 방법인 CAMEO를 제안합니다. 이는 비디오 프레임 간의 의미적 연관성을 반영하여 메모리를 활성화하도록 설계되었습니다.

- **Performance Highlights**: 모델 성능 평가 결과, 메모리를 맥락으로 사용하는 접근 방식이 기존의 0-shot 모델 대비 유의미한 향상을 이끌어낼 수 있음을 확인하였습니다. 그러나 실제 스트리밍 환경에서는 예측된 메모리가 잘못된 정보로 구성될 수 있어 성능 저하가 발생함을 관찰하였습니다. CAMEO를 통해 이러한 문제를 개선할 수 있으며, 이는 MLLMs의 비디오 이벤트 이해 능력을 현저히 증대시킬 수 있는 가능성을 포함하고 있습니다.



### MVIP -- A Dataset and Methods for Application Oriented Multi-View and Multi-Modal Industrial Part Recognition (https://arxiv.org/abs/2502.15448)
Comments:
          Accepted to IMPROVE 2025

- **What's New**: MVIP는 멀티 모달(multi-modal) 및 멀티 뷰(multi-view) 산업 부품 인식에 대한 새로운 데이터셋을 제공합니다. 이 데이터셋은 RGBD 멀티 뷰 데이터셋을 조정하여 물리적 속성, 자연어 및 슈퍼 클래스(super-class)와 같은 추가 객체 맥락을 결합한 최초의 사례입니다. MVIP는 소량의 데이터와 시각적으로 유사한 부품들을 다루고 100%에 가까운 정확도를 요구하는 산업적 문제를 해결하기 위한 연구 초점을 맞추고 있습니다.

- **Technical Details**: MVIP 데이터셋은 산업 부품 인식을 위한 다각적 검사소에서 수집되었으며, 10개의 조정된 RGBD 카메라와 저울을 통해 색상, 깊이 및 무게 외에도 포장 크기, 자연어 태그 및 슈퍼 클래스와 같은 여러 모달리티를 포함합니다. 이를 통해 모달리티 융합(modality fusion), 데이터 생성(synthetic data generation) 및 복잡한 데이터 샘플링(data sampling) 이론을 연구할 수 있습니다. 이 데이터셋은 3D 장면 및 객체 재구성을 가능하게 하여, 자동화된 데이터 생성 및 3D 기반 인식 방법에 대한 연구를 촉진합니다.

- **Performance Highlights**: MM 및 MV 산업 객체 인식을 위한 기준선 조사와 함께 새로운 보조 손실(auxiliary loss) 및 Transformer 기반의 MV 융합 방법을 제안합니다. MVIP는 기본 연구와 실제 산업 ML 응용 프로그램 사이의 격차를 좁히기 위해 다양한 최첨단 방법의 전이 가능성을 탐구하는 것을 목표로 합니다. 본 연구는 산업 부품 인식 분야의 효율적인 배포를 지원하며, 현재 산업용 분류기의 국한된 적응 문제를 해결하는 데 기여할 것입니다.



### LEAP: Enhancing Vision-Based Occupancy Networks with Lightweight Spatio-Temporal Correlation (https://arxiv.org/abs/2502.15438)
- **What's New**: 이 논문에서는 Lightweight Spatio-Temporal Correlation (LEAP) 방법을 제안하여, 기존의 occupancy networks의 성능을 크게 향상시키면서도 최소한의 계산 비용으로 작동합니다. LEAP는 다양한 baseline 네트워크에 통합할 수 있어, 플러그 앤 플레이 형태로 쉽게 적용할 수 있습니다. 기존 방법들이 시각적 단서 부족으로 인해 정확도에서 제약을 받는 경우가 많은 반면, LEAP는 이러한 문제를 해결하고 높은 성과를 달성합니다.

- **Technical Details**: LEAP 방법은 세 가지 단계로 작동합니다: 첫째, 최근 baseline 및 모션 특성을 1×1 convolution 연산을 통해 처리하여 중복을 줄이고 공유된 잠재 공간으로 나누어 토큰으로 변환합니다. 둘째, 이 토큰들을 삼중 스트림 융합 아키텍처를 통해 spatio-temporal correlation을 형성합니다. 마지막으로, 최종적으로 향상된 occupancy 결과를 생성하여 기존 네트워크의 출력을 보강합니다.

- **Performance Highlights**: 모델 실험 결과, LEAP가 기존의 SurroundOcc와 결합 시 Intersection over Union (IoU)를 5.18% 증가시켰고 평균 mIoU 지표도 1.82% 개선되었습니다. LEAP는 기존의 4D occupancy 방법들이 필요로 하는 자원 사용량을 줄이는 동시에 상당한 정확도를 달성하여, GPU 메모리 사용량을 39GB로 줄이고 추론 시간을 0.92초로 단축했습니다. 이러한 개선 사항 덕분에 LEAP는 여러 벤치마크에서 최첨단 성능(SOTA)을 입증하였습니다.



### Enhancing Vehicle Make and Model Recognition with 3D Attention Modules (https://arxiv.org/abs/2502.15398)
- **What's New**: 이번 연구에서는 Vehicle Make and Model Recognition (VMMR) 문제를 해결하기 위해 단순하고 매개변수가 없는 주의 모듈(SimAM)을 도입한 향상된 네트워크를 제안합니다. 이 모듈은 CNN(Convolutional Neural Networks) 기반 모델에 효과적으로 결합되어 모델의 표현력을 크게 향상시킵니다. SimAM은 중요한 부위에 집중하여 입력 이미지에서 가장 많은 정보를 추출하고 덜 중요한 영역은 무시하는 방식으로 작동합니다. 이를 통해 intra-class variation(클래스 내부 변동)과 inter-class similarity(클래스 간 유사성)의 부정적인 영향을 줄이는 것을 목표로 합니다.

- **Technical Details**: 주요 구조로는 SimAM을 CNN 모델의 중간 섹션에 두 가지 서로 다른 위치에 통합하여 특징을 제공하는 데 최적화되어 있습니다. 이 방법은 충분한 정보를 제공하며 지나치게 세부적이거나 지나치게 거친 정보를 가지지 않는 특징 맵을 생성하는 데 중점을 둡니다. SimAM은 3-D attention weights(3차원 주의 가중치)를 생성하여 네트워크가 중요한 부분에 더 집중할 수 있도록 합니다. 이 모듈은 기존의 SOTA(SOTA: State of the Art) 주의 모듈들보다 효율성이 높음을 입증합니다.

- **Performance Highlights**: 제안한 모델은 Stanford Cars dataset을 사용한 성능 평가에서 비교한 모델 중 90.69%의 가장 높은 정확도를 기록했습니다. CNN 뿐만 아니라 transformer 기반 모델들과의 비교에서도 뛰어난 성능을 보여줍니다. 특히, 모델 내 중요한 섹션에 집중함으로써 VMMR의 도전 과제를 해결하는 데 기여하고 있습니다. 이러한 성능 상승은 차량 분석에서의 응용 가능성에 대한 신뢰를 줍니다.



### LongCaptioning: Unlocking the Power of Long Caption Generation in Large Multimodal Models (https://arxiv.org/abs/2502.15393)
- **What's New**: 이 논문에서는 Large Multimodal Models (LMMs)의 긴 캡션 생성 문제를 다루며, Open-source LMMs가 300단어를 초과하는 출력을 일관되게 생성하지 못하는 것을 발견했습니다. 이를 해결하기 위해, LongCaption-Agent라는 프레임워크를 제안하여 다중 레벨 설명을 통합하여 긴 캡션 데이터를 합성합니다. 또한 LongCaption-10K라는 새로운 긴 캡션 데이터세트를 만들어 LMM의 출력 길이를 1,000단어 이상으로 확장할 수 있음을 보여주었습니다.

- **Technical Details**: LMMs는 보통 시각 인코더(visual encoder), 모달리티 프로젝터(modality projector), Language Decoder로 구성됩니다. 이 연구에서는 긴 캡션 예제의 부족이 LMMs의 출력 길이를 제한하는 주요 요인으로 나타났습니다. LongCaption-Agent는 이 문제를 해결하기 위해 3단계로 이루어진 프레임 수준, 클립 수준 및 비디오 수준의 캡션 합성을 통해 긴 캡션 데이터를 생성합니다.

- **Performance Highlights**: LongCaption-Bench라는 벤치마크를 개발하여 LMMs가 생성한 긴 캡션의 품질을 종합적으로 평가했습니다. 이 벤치마크에서는 281개의 테스트 비디오에서 평균 1,161.3단어의 캡션을 생성한 최첨단 성과를 달성했으며, 더 큰 독점 모델을 초월하는 성능을 보였습니다. 이를 통해 LMM의 긴 캡션 생성 능력을 효과적으로 평가하며 기여했습니다.



### The Role of Background Information in Reducing Object Hallucination in Vision-Language Models: Insights from Cutoff API Prompting (https://arxiv.org/abs/2502.15389)
Comments:
          Under review

- **What's New**: 본 연구는 비전-언어 모델 (Vision-Language Models, VLMs)의 출력이 입력 이미지와 모순되는 경우가 발생하여 신뢰성에 영향을 미친다는 점을 강조합니다. 특히, Attention-driven visual prompting 기법이 오브젝트 홀루시네이션(object hallucination)을 줄이는 데 효과적이지만, 이러한 기법의 배경 맥락(background context) 보존이 홀루시네이션 완화에 필수적임을 밝혀냈습니다.

- **Technical Details**: API Prompting은 비전-언어 모델에서 유도된 비주얼 어텐션 히트맵을 사용하여 이미지의 중요한 부분을 강조하는 새로운 비주얼 프로프트 기법입니다. 이 방법은 CLIP과 LLaVA를 활용하여 이미지의 세부 정보를 수집하고, 주목하는 오브젝트(target object)에 대한 어텐션 정확도를 평가하는 방식으로 설계되었습니다. 본 연구는 배경 정보를 제거한 경우의 성능 변화를 분석하고, 레퍼런스 세그멘테이션 데이터와의 비교를 통해 API Prompting의 효과를 평가합니다.

- **Performance Highlights**: 실험 결과 API Prompting을 적용했을 때 Cutoff 조건에서 Recall의 개선이 관찰되었으며, 기존 VLM 모델에 비해 약 3%의 성능 향상 효과가 있었습니다. 또한, 출력의 정답률이 높을수록 주요 오브젝트와 비주얼 어텐션 간의 정렬이 더 잘 이루어졌음을 발견했습니다. 이러한 결과는 배경 없이 올바르게 알고리즘이 작동하는 조건에서의 성능이 다른 조건과 유사하다고 나타났으며, 특히 부정확한 결과에 대해 Cutoff를 적용했을 때 Recall이 크게 향상되었습니다.



### MOVE: A Mixture-of-Vision-Encoders Approach for Domain-Focused Vision-Language Processing (https://arxiv.org/abs/2502.15381)
Comments:
          10 pages, 6 figures, 4 tables

- **What's New**: 이 논문에서는 MOVE (Mixture of Vision Encoders)라는 새로운 접근 방식을 제안하여, 여러 개의 사전 훈련된 비전 인코더를 활용하여 특화된 멀티모달 작업에서 모델의 성능을 향상시키고자 합니다. MOVE는 적절한 인코더를 자동으로 선택하여 입력을 라우팅함으로써 ChartQA, MMBench, MMMU와 같은 다양한 벤치마크에서 성능을 개선합니다. 이 방법은 고해상도 이미지에서의 복잡한 절단 기술(image slicing)을 피하면서 경쟁력 있는 정확성을 보여줍니다.

- **Technical Details**: MOVE 모델은 네 가지 주요 구성 요소로 이루어져 있으며, 이들 각각은 사전 훈련된 대형 언어 모델(LLM), 여러 비전 전문가 인코더, 인코더 선택을 담당하는 라우터(router), 그리고 시각적과 텍스트적 표현을 연결하는 어댑터 모듈입니다. MOVE는 Qwen2 및 Qwen2.5와 같은 최신 LLM을 기본 모델로 사용하여 다양한 시각적 입력을 효과적으로 처리합니다. 여러 비전 인코더(InternViT, Texify, UniChart)를 통합하여 각기 다른 도메인에 적합한 고급 기능 표현을 유지함으로써 효율적이고 적응력이 뛰어난 특징 추출을 보장합니다.

- **Performance Highlights**: MOVE는 256, 196 또는 576개의 비전 토큰만으로도 최첨단 모델과 유사한 성능을 달성하여, 기존 멀티모달 모델보다 더 높은 효율성을 보여줍니다. 다양하고 특화된 비전 인코더들을 독상적인 영역에 맞추어 선택함으로써, 모델의 벤치마크 성능을 향상시키는데 성공하였습니다. 이 연구는 향후 연구와 실용적인 사용을 위한 소스 코드도 배포하여, 많은 연구자들이 MOVE를 활용할 수 있도록 기여하고 있습니다.



### Weakly Supervised Video Scene Graph Generation via Natural Language Supervision (https://arxiv.org/abs/2502.15370)
Comments:
          10 pages, ICLR 2025

- **What's New**: 이번 논문에서는 비디오 장면 그래프 생성을 위한 새로운 프레임워크인 NL-VSGG(Natural Language-based Video Scene Graph Generation)를 소개합니다. 이 프레임워크는 비디오 캡션을 활용하여 비디오의 모든 프레임에 대한 주석 없이도 VidSGG 모델을 훈련할 수 있게 합니다. 이전의 방식들과 비교하여, NL-VSGG는 시간적 세부정보와 행동 지속성 변동성을 고려하여 성능 향상을 이끕니다.

- **Technical Details**: NL-VSGG는 두 가지 주요 모듈로 구성됩니다: TCS(Temporality-aware Caption Segmentation) 모듈과 ADV(Action Duration Variability-aware caption-frame alignment) 모듈입니다. TCS는 대형 언어 모델(LLM)을 기반으로 비디오 캡션을 시간 순서에 따라 여러 문장으로 분할합니다. ADV는 분할된 문장과 적절한 프레임을 행동의 지속성 변동성을 고려하여 정렬하는 역할을 합니다.

- **Performance Highlights**: NL-VSGG를 사용한 실험 결과, Action Genome 데이터셋에서 더 간단한 WS-ImgSGG 파이프라인을 적용한 경우와 비교하여 성능이 크게 향상됨을 보여주었습니다. 또한, NL-VSGG를 통해 훈련된 VidSGG 모델이 훈련 데이터에 포함되지 않은 더 다양한 행동 클래스를 예측할 수 있어 실제 적용 가능성이 높아졌습니다.



### PFSD: A Multi-Modal Pedestrian-Focus Scene Dataset for Rich Tasks in Semi-Structured Environments (https://arxiv.org/abs/2502.15342)
- **What's New**: 본 논문에서는 반구조화된 환경에서 보행자의 인식을 향상시키기 위한 다중 모달 보행자 중심 장면 데이터셋(PFSD)을 소개합니다. PFSD는 130,000개 이상의 보행자 인스턴스를 포함하고 있으며, 그들은 다양한 밀도와 움직임 패턴, 차폐를 가진 여러 시나리오에서 캡처되었습니다. 또한, 새로운 하이브리드 다중 스케일 융합 네트워크(HMFN)를 통해 반구조화된 복잡한 환경에서 보행자를 보다 효과적으로 탐지할 수 있는 방법을 제안합니다.

- **Technical Details**: PFSD는 nuScenes 포맷으로 엄격하게 주석이 달린 데이터셋으로, 포인트 클라우드 세분화, 감지 및 객체 ID를 통해 객체 추적을 지원합니다. 이 데이터셋은 보행자가 동적이고 예측할 수 없는 행동을 보임에 따라 고밀도 보행자 주석을 포함하고 있어 실세계 상호작용을 이해하는 데 중요한 역할을 합니다. HMFN은 다양한 스케일의 특징을 캡처하고 융합하여 밀집된 보행자 환경에서의 탐지 성능을 향상시키는 목적으로 설계되었습니다.

- **Performance Highlights**: PFSD를 기반으로 한 실험에서 HMFN은 기존 방법들에 비해 평균 정밀도(mAP)에서 유의미한 향상을 보여줍니다. 또한, PFSD는 정적 구조물과 동적 요소 간의 보행자 상호작용을 캡처하여 다중 객체 추적 성능을 향상시키는 데 기여합니다. 이 연구는 밀집 보행자의 탐지 및 추적 문제를 해결하며, 도시 밀집 장면에서의 지능적 인식과 자동 의사 결정을 위한 기반을 마련합니다.



### SentiFormer: Metadata Enhanced Transformer for Image Sentiment Analysis (https://arxiv.org/abs/2502.15322)
- **What's New**: 최근 소셜 미디어 사용자들이 일상 감정을 표현하기 위해 이미지를 포스팅함에 따라, 이미지 감정 분석(image sentiment analysis)이 더욱 주목받고 있습니다. 본 논문에서는 이미지와 여러 메타데이터를 통합하기 위해 메타데이터 강화 트랜스포머(Metadata Enhanced Transformer, SentiFormer)를 제안합니다. 이 방법은 이미지에 대한 텍스트 설명과 키워드 태그와 같은 여러 메타데이터를 융합하여 감정 분석의 정확성을 높입니다.

- **Technical Details**: SentiFormer의 구조는 세 가지 주요 모듈로 구성되어 있습니다: 특징 표현 모듈, 적응적 유사도 학습 모듈, 그리고 교차 모달 융합 및 예측 모듈입니다. 먼저, BLIP을 사용하여 이미지에 대한 텍스트 설명을 생성하고, Faster R-CNN을 통해 주요 객체 태그를 추출한 후, Hiera를 통해 장면 태그를 얻습니다. 이후 CLIP을 활용하여 이미지와 메타데이터의 통합 표현을 생성하며, 이를 통해 더욱 효과적인 감정 분석을 가능하게 합니다.

- **Performance Highlights**: 세 개의 공개 데이터세트에서 수행된 실험 결과는 제안된 SentiFormer가 기존 방법들보다 우수한 성능을 보임을 보여줍니다. 메타데이터를 활용함으로써 감정 분석의 정확도를 크게 향상시키는 동시에, 기존의 수작업 특성 분석에서 벗어난 다양한 데이터 통합의 가능성을 제공합니다. 연구 결과는 감정 이해에 대한 새로운 전망을 제시하며, 코드와 데이터셋도 공개되어 있어 연구 커뮤니티의 추가 연구에 기여할 수 있습니다.



### Research advances on fish feeding behavior recognition and intensity quantification methods in aquacultur (https://arxiv.org/abs/2502.15311)
Comments:
          22 pages, 4 figures,

- **What's New**: 이 논문은 어류 사육 관리를 위한 핵심 요소인 어류 먹이 행동 인식 및 강도 측정의 최근 연구 동향을 종합적으로 검토하고 있습니다. 전통적인 모달리티(single modality) 기반의 컴퓨터 비전(computer vision), 음향(acoustics) 및 센서(sensor)를 이용한 방법들에 대한 연구진전을 소개하며, 이러한 기법들이 어류의 건강 모니터링 및 사육 효율성 개선에 어떻게 기여하는지를 설명합니다.

- **Technical Details**: 어류 먹이 행동 인식 및 강도 측정 기법은 다양한 기술 기반으로 발전하고 있으며, 최근에는 멀티모달 융합(multimodal fusion) 기술의 적용이 증가하고 있습니다. 이는 개별 기술의 단점을 보완하며 다양한 데이터를 결합하여 더 정확한 인식을 가능하게 합니다. 각 기술의 활성화에 따라 어류 사육 환경의 향상이 기대됩니다.

- **Performance Highlights**: 이 논문에서는 다양한 기술의 장단점을 비교 분석하며, 향후 연구 방향을 제시합니다. 멀티모달 접근법이 기존 기법보다 더 나은 성능을 발휘할 가능성이 크고, 이는 차세대 어류 사육 관리 시스템에 기여할 것입니다. 연구자들은 이러한 기법들이 어류 행동 분석과 같은 분야에서 잠재력을 가지고 있다고 강조합니다.



### Road Traffic Sign Recognition method using Siamese network Combining Efficient-CNN based Encoder (https://arxiv.org/abs/2502.15307)
- **What's New**: 이 논문에서는 교통 표지 인식(traffic sign recognition, TSR) 문제를 해결하기 위해 IECES-network라는 새로운 모델을 제안합니다. 이 모델은 효율적인 CNN(Efficient-CNN) 기반 인코더와 시암 네트워크(Siamese net)를 사용하여 복잡한 환경에서 발생하는 모션 블러(motion-blur) 및 가림(occlusion) 문제를 처리하는 데 중점을 두고 있습니다. 이 접근법은 특히 실시간 인식에서 높은 정확성과 강인성을 달성하는 데 중요한 기여를 합니다.

- **Technical Details**: 제가 제안하는 IECES 네트워크는 세 가지 단계로 구성됩니다: Efficient-CNN 기반 인코더, 시암 백본(Siamese backbone), 그리고 완전 연결 층(fully-connected layers)입니다. 초기 단계에서는 합성 훈련 샘플과 표준 이미지를 사용하여 교통 신호의 특징을 추출하고 인코딩하며, 이후 시암 신경망을 통해 입력 샘플과 템플릿 간 거리 계산을 통해 모션 블러 및 가림 샘플에 대비하는 강인성을 개선하는 방법을 모색합니다. 이러한 특성들은 SoftMax 함수를 사용하는 분류층에서 코드 결합을 통해 최종적으로 교통 표지의 분류를 가능하게 합니다.

- **Performance Highlights**: 제안된 IECES 네트워크는 Tsinghua-Tencent 100K 데이터셋과 독일 교통 표지 인식 벤치마크(German Traffic Sign Recognition Benchmark) 데이터셋에서 우수한 성능을 입증했습니다. 다른 최첨단 방법들과 비교할 때, 모션 블러와 가림 환경에서도 88.1%의 정밀도(precision), 86.43%의 재현율(recall), 86.1%의 정확도(accuracy)를 기록했으며, 모델의 경량 스케일은 2.9M에 불과합니다. 또한 처리 시간은 프레임당 0.1초로 기존 방법보다 1.5배 향상되었습니다.



### A Novel Riemannian Sparse Representation Learning Network for Polarimetric SAR Image Classification (https://arxiv.org/abs/2502.15302)
Comments:
          13 pages, 9 figures

- **What's New**: 본 논문에서는 새로운 Riemannian Sparse Representation Learning Network(SRSR CNN)을 제안하여 PolSAR 이미지 분류의 문제를 해결하고자 합니다. 기존의 기법들은 복잡한 행렬 구조를 왜곡하였으나, 제안된 모델은 Riemannian 공간에서의 기하적 구조를 효과적으로 학습할 수 있는 방식으로 설계되었습니다. 또, 고차원 의미를 학습하기 위해 CNN-enhanced 모듈이 추가되어 분류 성능을 향상시킵니다.

- **Technical Details**: 제안하는 SRSR CNN은 PolSAR 이미지를 처리하기 위해 Superpixel 기반의 Riemannian Sparse Representation(SRSR) 모델을 통해 희소한 특징을 학습합니다. 또한, SRSR 모델의 최적화 절차를 풀어 SRSRNet으로 발전시켜 자동으로 희소 계수와 사전 원자를 학습할 수 있게 합니다. 이 네트워크는 복합 공분산 행렬을 네트워크 입력으로 직접 사용하며, Riemannian 메트릭을 통해 기하학적 구조와 희소 특징을 학습합니다.

- **Performance Highlights**: 세 가지 실제 PolSAR 데이터셋에 대한 실험 결과, 제안된 방법이 최신 기법들보다 우수한 성능을 보이며, 정확한 엣지 디테일과 올바른 지역 동질성을 보장하는 데 성공하였습니다. 특히, 기존의 Euclidean 측정 방식의 문제가 해결됨에 따라, 더욱 정확한 분류 성능을 나타냈습니다. 따라서 제안된 SRSR CNN은 PolSAR 이미지 분류에 있어 효과적인 방법으로 자리잡을 것입니다.



### Soybean pod and seed counting in both outdoor fields and indoor laboratories using unions of deep neural networks (https://arxiv.org/abs/2502.15286)
- **What's New**: 이번 연구에서는 대두(soybean) 꼬투리(pod)와 씨앗(seed)의 자동 세기를 위한 효율적인 딥러닝(deep learning) 모델을 개발했습니다. 실외(field) 환경에서는 가려진 씨앗(occluded seeds) 마저도 세는 능력을 갖춘 YOLO(YOLO-SAM) 모델을 제안하였으며, 더 나아가 HQ-SAM과 도메인 적응(domain adaptation) 기술을 통합하여 모델의 강인성과 일반화(generalization)를 개선했습니다. 실내(laboratory)에서는 Mask-RCNN에 Swin Transformer 모듈을 더하여 실제 이미지에 대한 우수한 정확도를 달성하였습니다.

- **Technical Details**: 연구에서 사용한 YOLO-DA 모델은 실외 환경에서 대두 꼬투리와 씨앗을 정확히 세는 데 중점을 두었습니다. 또한, Mask-RCNN-Swin 모델은 적은 양의 레이블이 있는 데이터로부터 생성된 합성 이미지를 기반으로 훈련되었습니다. 이러한 접근방식은 실내에서는 높은 정확도와 낮은 평균 절대 오차(mean absolute error, MAE)를 보여주었습니다.

- **Performance Highlights**: 실외에서의 대두 꼬투리 세기에서 평균 절대 오차(MAE)는 6.13으로, 씨앗 세기에서는 10.05를 기록했습니다. 반면 실내에서의 대두 꼬투리와 씨앗 세기에서 평균 절대 오차는 각각 1.07과 1.33으로, 거의 완벽한 정확도를 달성하였습니다. 이러한 성과는 대두의 수확예측(yield estimation) 및 육종 과정을 크게 가속화할 것으로 기대됩니다.



### CopyJudge: Automated Copyright Infringement Identification and Mitigation in Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.15278)
Comments:
          17pages, 8 figures

- **What's New**: 본 논문에서는 AI 생성 이미지들이 저작권이 있는 작품과 실질적으로 유사한지를 판단하기 위한 자동화된 저작권 침해 식별 프레임워크인 CopyJudge를 제안합니다. CopyJudge는 대형 비전-언어 모델(LVLM)을 활용하여 법원 절차를 모방하고, 이미지의 유사성을 평가하며, 침해 가능성에 대한 자세한 판단 근거를 제공합니다. 또한, 본 연구에서는 침해를 방지하기 위한 자동화된 전략을 통해 감지된 침해적인 프롬프트를 최적화하는 방법을 탐구합니다.

- **Technical Details**: CopyJudge는 이미지의 다양한 요소를 분해하고 필터링하여 저작권 보호를 받지 않는 부분을 제외한 후, 필터링된 부분 간의 유사성을 비교하는 추상화-필터링-비교 테스트 프레임워크를 사용합니다. 여러 LVLM이 서로 논의하고 점수를 매기는 다중 에이전트 토론 방법을 도입하여, 최종적으로 메타 심사관이 공감대를 기반으로 점수와 근거를 제시합니다. 이 과정에서 인간의 선호도를 반영하기 위해 소수의 사례를 통한 교육을 진행합니다.

- **Performance Highlights**: 실험 결과, CopyJudge는 최첨단 성능에 부합하면서도 다양한 형태의 침해에 대해 뛰어난 일반화 능력과 해석 가능성을 제공합니다. 또한 제안된 완화 방법은 비침해 표현을 손실하지 않고 메모리화 및 지적 재산권 침해를 보다 효과적으로 완화할 수 있는 것으로 나타났습니다. 이러한 성능 향상은 AI 생성 이미지의 저작권 문제 해결에 기여할 것으로 기대됩니다.



### Omnidirectional Image Quality Captioning: A Large-scale Database and A New Mod (https://arxiv.org/abs/2502.15271)
- **What's New**: 이 논문에서는 동심원 이미지를 평가하기 위한 새로운 데이터베이스인 OIQ-10K를 제안합니다. 이 데이터베이스는 동심원 이미지의 균일한 왜곡(homogeneous distortion) 뿐만 아니라 이질적인 왜곡(heterogeneous distortion)을 포함하여 총 10,000개의 이미지를 포함하고 있습니다. 이 연구는 사람의 주관적인 의견을 수집하고 왜곡의 공간적 분포를 분석하여 이미지를 평가하는 데 필수적인 기초 자료를 제공합니다.

- **Technical Details**: OIQ-10K 데이터베이스는 다양한 왜곡 상황에 대한 정신물리학적 연구를 바탕으로 구성되었으며, 사용자들의 헤드 및 아이 무브먼트 데이터도 포함됩니다. 이를 통해, IQCaption360이라는 새로운 비구조적(Non-Reference) OIQA 모델을 제안하였으며, 이 모델은 다중 작업(Multi-Task) 프레임워크를 사용하여 이미지 품질을 예측할 수 있는 기능을 제공합니다. 또한, Vision Transformer를 활용하여 이미지의 다양한 스케일 기능을 추출합니다.

- **Performance Highlights**: IQCaption360 모델은 제안된 OIQ-10K 데이터베이스에서 기존의 최첨단 모델들을 크게 능가하는 성능을 보였으며, 이미지 품질에 대한 질적 캡션을 생성하는 데 성공했습니다. 이는 기존 OIQA 연구들에서 다루었던 단일 질적 점수와는 다르게, 더 정보가 풍부한 캡션을 생성하여 동심원 이미지의 품질을 보다 효과적으로 표현합니다.



### SiMHand: Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training (https://arxiv.org/abs/2502.15251)
Comments:
          ICLR 2025. arXiv admin note: text overlap with arXiv:2409.09714

- **What's New**: 본 연구에서는 다양한 손 이미지를 활용한 3D 손 포즈 추정을 위한 새로운 프레임워크인 SimHand를 제안합니다. 대규모 손 이미지로 사전 학습을 진행하며, 기존의 방법들이 갖고 있던 한계를 초월하고자 합니다. 특히, 서로 다른 손 이미지의 유사성을 기반으로 한 대비 학습(contrastive learning) 기법을 도입하여 성능 개선을 도모합니다.

- **Technical Details**: 우리는 2.0M 이상의 손 이미지를 Ego4D와 100DOH와 같은 최근의 인간 중심 비디오에서 수집했습니다. 또한, 이러한 이미지에서 유사한 손을 식별하기 위해 off-the-shelf 2D 손 포즈 추정기를 사용하며, 이로 인해 각각의 유사한 손 쌍으로부터 정보를 수집합니다. 이 방식은 전통적인 대비 학습 접근법보다 더 나은 성능을 얻을 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안하는 SimHand 방법이 FreiHand, DexYCB, AssemblyHands 데이터셋에서 기존 state-of-the-art 방법들인 PeCLR를 능가하는 것으로 나타났습니다. FreiHand에서는 15%, DexYCB에서는 10%, AssemblyHands에서는 4%의 성능 향상을 보였습니다. 이는 우리의 대비 학습 기법과 동적 가중 부여(adaptive weighting) 방식이 효과적임을 입증합니다.



### An ocean front detection and tracking algorithm (https://arxiv.org/abs/2502.15250)
- **What's New**: 이번 연구에서는 기존의 수동 및 자동 해양 전선 탐지 기법을 향상시키기 위한 자동 전선 탐지 및 추적 알고리즘을 제안하였다. 이 알고리즘은 Bayesian decision과 metric space를 기반으로 하여, 해양 전선의 연속성을 높이기 위해 front merging, filling 및 ring deletion과 같은 기술을 도입하였다. 이러한 접근 방식은 다른 영역인 computer vision에서도 응용 가능하며, 특히 edge detection 및 tracking 분야에 대한 기대를 모으고 있다.

- **Technical Details**: 해양 전선은 온도, 염도, 밀도 및 클로로필과 같은 다양한 특성을 가진 수조 사이의 경계로 정의된다. 논문에서는 기존의 histogram, Lyapunov, gradient 및 machine learning 기법들로부터 도출된 문제점을 해결하기 위해, 그래디언트 정보를 사전 확률로 간주하고 두 가지 계산 작업을 결합하여 Bayesian 결정을 통해 프론탈 존을 탐지한다. 이 과정에서 수학적 형태학을 활용하여 검출된 프론탈 존을 더욱 세분화하고, 딥 퍼스트 서치 알고리즘을 통해 전선을 추적하며, 링 구조를 삭제하는 방법도 제안하였다.

- **Performance Highlights**: 제안된 알고리즘은 다수의 데이터 날짜에서 정의된 거리를 통해 동일한 해양 전선을 식별할 수 있도록 하였다. 추가적으로, 최적의 threshold 설정의 문제를 해결하여 자동으로 전선 탐지의 정확성을 높였다. 연구 결과는 해양 관측 및 모델링에 도움을 줄 수 있을 뿐만 아니라, computer vision의 여러 분야로의 확장 가능성까지 포함하고 있어 그 활용 가치가 높다.



### AutoMR: A Universal Time Series Motion Recognition Pipelin (https://arxiv.org/abs/2502.15228)
Comments:
          5 figures

- **What's New**: 이 논문에서는 다중 모달 데이터셋을 위한 자동화된 모션 인식 파이프라인인 AutoMR을 제안합니다. 이 프레임워크는 데이터 전처리, 모델 학습, 하이퍼파라미터 튜닝 및 평가를 통합하여 다양한 시나리오에서 강력한 성능을 발휘합니다. 두 가지 주요 도전 과제를 해결하려고 하며, 기존의 복잡한 전처리 과정 없이도 효율적인 모션 인식을 가능케 합니다.

- **Technical Details**: AutoMR은 데이터셋의 다양성, 모델 확장성, 하이퍼파라미터 최적화와 같은 문제를 해결하는 통합 솔루션을 제공합니다. 이 시스템은 QuartzNet 모델을 핵심으로 사용하며, 다양한 데이터셋에서 모델을 표준화하고 자동 튜닝을 통해 성능을 향상시킵니다. 또한, 데이터셋의 일관성을 보장하기 위해 구조화된 포맷을 정의하고 데이터 전처리 및 훈련 구성 모듈을 포함하고 있습니다.

- **Performance Highlights**: AutoMR은 10개의 벤치마크 데이터셋에서 기존의 최신 기술들과 비교하여 높은 정확도를 달성하며, 특히 OPPORTUNITY 데이터셋에서는 이전 모델보다 5% 이상 높은 성능을 기록했습니다. 그러나 DB4와 LMDHG에서는 성능이 다소 낮아, 이러한 복잡한 데이터에서의 향상이 필요합니다. 자동 하이퍼파라미터 튜닝의 경우, 대부분의 데이터셋에서 수동 조정에 맞먹거나 우수한 성능을 보여, 전문가 개입 없이도 효과적인 최적화가 가능함을 증명했습니다.



### FlipConcept: Tuning-Free Multi-Concept Personalization for Text-to-Image Generation (https://arxiv.org/abs/2502.15203)
Comments:
          9 pages, 4 figures

- **What's New**: 최근 텍스트에서 이미지로(텍스트-투-이미지, T2I) 여러 개인화된 개념을 단일 이미지에 통합하는 방법이 주목받고 있습니다. 기존 방법들은 복잡한 장면에서 비개인화 영역의 왜곡 때문에 성능 저하를 겪습니다. 이를 해결하기 위해 FlipConcept이라는 새로운 접근법을 제안하며, 추가 조정 없이 여러 개인화된 개념을 이미지에 원활하게 통합할 수 있습니다. 제안된 방법은 개인화된 개념의 외관을 정확하게 모방하는 Guided Appearance Attention을 도입합니다.

- **Technical Details**: FlipConcept 프레임워크는 두 가지 단계로 작동합니다. 첫 번째 단계에서는 백그라운드 이미지와 이를 기반으로 생성된 마스크를 준비하고, Edit-Friendly DDPM inversion을 통해 편집하기 용이한 잠재 표현을 얻습니다. 두 번째 단계에서는 이전 단계의 잠재 표현과 마스크를 활용하여 개인화된 개념을 백그라운드와 통합하는 이미지를 생성합니다. Guided Appearance Attention, Mask-Guided Noise Mixing 및 Background Dilution의 세 가지 핵심 기술을 도입하여, 개념 간의 관계를 유지하면서 비개인화 영역에 대한 간섭을 최소화합니다.

- **Performance Highlights**: FlipConcept은 기존 방법들을 능가하는 성능을 보여주었으며, 복잡한 장면에서도 개인화된 이미지 생성에 강점을 보입니다. 특히, 여러 캐릭터나 객체가 포함된 상황에서도 각 개념의 일관성을 유지하고 배경의 완전성을 보장합니다. 실험 결과, CLIP 평가 점수에서도 높은 성과를 달성하여 생성된 콘텐츠의 품질과 관련성을 강조하였습니다.



### UrbanSAM: Learning Invariance-Inspired Adapters for Segment Anything Models in Urban Construction (https://arxiv.org/abs/2502.15199)
- **What's New**: 새로운 UrbanSAM 모델은 복잡한 도시 환경에서의 객체 추출 및 세분화를 위해 맞춤화된 SAM(Segment Anything Model) 버전입니다. 이 모델은 원거리 감지 데이터에서 발생하는 스케일링 효과를 처리할 수 있도록 설계되었습니다. UrbanSAM은 멀티 해상도 분석(Multi-resolution Analysis, MRA) 이론에 영감을 받아 설계된 학습 가능한 프로ンプ터와 Uscaling-Adapter를 통해 다양한 스케일의 맥락 정보를 캡처합니다.

- **Technical Details**: UrbanSAM은 기존의 SAM 구조를 기반으로 하여 스케일 변화를 수용하는 U-형 어댑터를 추가했습니다. 이 어댑터는 교차 주의(cross-attention) 메커니즘을 활용하여 백본 인코더와 통합되어, 여러 스케일에서의 불변 특성을 학습할 수 있도록 도와줍니다. 이러한 구조는 고해상도 이미지 처리 시 맥락적 의미 정보를 통합하여 정확도를 높입니다.

- **Performance Highlights**: 실험 결과 UrbanSAM은 건물, 도로, 수역 등 다양한 스케일을 가진 도시 객체에 대한 세분화 성능이 우수함을 입증했습니다. 특히, 수집된 데이터셋을 통해 다양한 도시 환경에서도 강력하고 안정적인 세분화를 수행할 수 있는 능력을 보여주었습니다. 이는 SAM의 기존 한계를 극복하고 더욱 정교한 이미지 세분화를 가능하게 합니다.



### Image Translation-Based Unsupervised Cross-Modality Domain Adaptation for Medical Image Segmentation (https://arxiv.org/abs/2502.15193)
Comments:
          5 pages, 1 figure. arXiv admin note: substantial text overlap with arXiv:2303.07674

- **What's New**: 이번 연구에서는 의료 이미지의 다중 형태에 대한 비지도 학습 방법을 제안합니다. 기존의 감독(deep supervised) 학습이 의료 이미지에서는 어려운 점이 많았으나, 변환(image translation)을 통한 비지도 학습 방식을 활용하여 주목할 만한 성과를 보여줍니다. 이는 의료 센터마다 다르게 수집된 이미지의 변동성을 해결하기 위한 접근입니다.

- **Technical Details**: 의료 영상을 비지도 방식으로 해결하기 위해, 주어진 주석이 있는 소스 모달리티 이미지를 비주석(target modality)으로 변환하여 사용합니다. 이 과정은 효율적인 Supervised learning을 목표로 하며, 변환된 이미지는 실제 이미지와 미세한 차이를 가지고 있기 때문에 Self-training 방법을 적용하여 성능을 더욱 향상시킵니다. 제안된 방법의 성능에는 Dice Similarity Coefficient (DSC)와 Average Symmetric Surface Distance (ASSD)를 포함한 다양한 메트릭이 활용되었습니다.

- **Performance Highlights**: 제안된 방법은 vestibular schwannoma (VS) 및 귀의 코클레아(segmentation) 작업에서 각각 평균 DSC가 $0.8351 \, 	ext{and} \, 	ext{ASSD} = 1.6712$를 기록하며, 코클레아에서는 $0.8098 \, 	ext{and} \, 0.2317$를 도출하였습니다. 이는 Cross-Modality Domain Adaptation (crossMoDA 2022) 챌린지의 검증단계 리더보드에서 관찰된 결과로, 모델의 효율성과 적용 가능성에 대해 긍정적인 평가를 받고 있습니다.



### Hierarchical Context Transformer for Multi-level Semantic Scene Understanding (https://arxiv.org/abs/2502.15184)
Comments:
          This paper has been accepted by the IEEE TCSVT

- **What's New**: 이번 연구에서는 수술 장면의 다층적 세밀 이해를 위한 새로운 프레임워크, 즉 다계층 맥락 변환기(HCT) 네트워크를 제안합니다. 이 접근법은 단계 인식, 동작 및 도구 탐지를 포함한 다단계 의미(scene understanding)를 통해 수술 환경을 종합적으로 이해할 수 있도록 돕습니다. 연구팀은 계층적 관계 집계 모듈(HRAM)을 설계하여 다양한 작업 간의 관계를 효과적으로 탐구합니다.

- **Technical Details**: HCT 네트워크는 계층적 관계를 인코딩하고 다른 작업으로부터의 정보를 결합하여 특정 작업에 대한 표현을 향상시키는 데 사용됩니다. 연구는 또한 상호 작업 대조 학습(ICL) 알고리즘을 활용하여 모델 성능을 극대화하며, 공간 및 시간적 정보를 학습할 수 있도록 공간-시간 어댑터(ST-Ada)를 포함한 HCT+ 네트워크를 개발하였습니다.

- **Performance Highlights**: 제안된 방법은 백내장 비디오 데이터셋과 공공의 내시경 PSI-AVA 데이터셋에서 수행된 광범위한 실험을 통해 최고의 성능을 보여주며, 기존의 최신 기법들과 비교해 상당한 성능 향상을 기록했습니다. 연구 결과는 각 기여의 중요성을 강조하며, 코드 또한 공개되어 많은 연구자들이 이 모델을 활용할 수 있게 하였습니다.



### OccProphet: Pushing Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with Observer-Forecaster-Refiner Framework (https://arxiv.org/abs/2502.15180)
Comments:
          Accepted by ICLR2025

- **What's New**: 이 논문에서는 고속 자율주행을 위한 Occupancy forecasting의 새로운 프레임워크인 OccProphet를 제안합니다. OccProphet는 훨씬 낮은 연산 요구 사항으로 미래 3D 점유 상태를 예측할 수 있습니다. 이 프레임워크는 Observer, Forecaster 및 Refiner라는 세 가지 경량 구성 요소로 구성되어 있습니다.

- **Technical Details**: OccProphet는 Efficient 4D Aggregation과 Tripling-Attention Fusion을 통해 3D 멀티 프레임 복셀에서 시공간(spatio-temporal) 특징을 추출하는 Observer 기능을 수행합니다. Forecaster는 장면 조건에 따라 미래 점유 예측을 수행하고, Refiner는 예측 결과의 품질을 향상시킵니다. 이 시스템은 카메라 전용 방식으로 운영되며, 고급 LiDAR 장비의 필요성을 줄입니다.

- **Performance Highlights**: 실험 결과, OccProphet는 기존의 Cam4DOcc에 비해 58%에서 78%의 연산 비용을 절감하고, 2.6배의 속도 향상을 달성했습니다. 또한, 비교 결과에서 4%에서 18% 더 높은 예측 정확도를 기록하여 효율성과 효과성을 입증합니다. 이러한 성과는 nuScenes, Lyft-Level5 및 nuScenes-Occupancy 데이터셋에서 일관되게 나타났습니다.



### Nonlinear Dynamical Systems for Automatic Face Annotation in Head Tracking and Pose Estimation (https://arxiv.org/abs/2502.15179)
Comments:
          25 pages, 10 figures

- **What's New**: 이 논문은 3D 얼굴 추적에서 Extended Kalman Filter (EKF)와 Unscented Kalman Filter (UKF)의 성능을 면밀히 비교합니다. EKF는 일반적으로 룰린한 비선형 시스템에 효과적이라면, UKF는 고차 비선형성 포착에 우수한 성능을 보입니다. 연구 결과에 따르면, EKF는 불확실한 노이즈가 있는 현실 세계 상황에서 더 좋은 신뢰성을 보여주었습니다.

- **Technical Details**: EKF와 UKF는 노이즈가 있는 관측치로부터 동적 시스템의 상태를 추정하는 기술입니다. EKF는 Jacobi 행렬을 사용하여 비선형 역학을 선형화하는 방법을 이용하는 반면, UKF는 비선형성을 더욱 정확하게 전파하기 위해 unscented 변환을 사용합니다. 이 연구에서는 비결정론적 및 확률적인 시나리오에서 두 필터의 성능을 체계적으로 평가합니다.

- **Performance Highlights**: UKF는 통제된 환경에서의 고정밀 애플리케이션에 적합하지만, EKF는 예측할 수 없는 노이즈가 있는 현실 세계 시나리오에서 더 나은 성과를 보입니다. 결과적으로, 이 연구는 3D 얼굴 추적 시스템에서 적합한 필터링 기술 선택을 돕기 위한 통찰을 제공합니다.



### Methods and Trends in Detecting Generated Images: A Comprehensive Review (https://arxiv.org/abs/2502.15176)
Comments:
          30 pages, 4 Figures, 10 Tables

- **What's New**: 최근 생성 모델의 발전으로 인해 고품질 멀티미디어 데이터 합성이 가능해졌으나, 이로 인한 악성 공격 및 사회적 해악에 대한 우려도 커지고 있습니다. 이러한 문제를 인식한 연구자들은 합성 데이터를 효과적으로 탐지할 수 있는 방법론 개발에 집중하고 있으며, 기존 하위 전술을 넘어서 다중 모달 프레임워크를 활용한 최근 기술들을 포괄적으로 검토했습니다.

- **Technical Details**: 본 조사에서는 최신 생성형 AI 모델에 의해 생성된 합성 이미지를 탐지하고 분류하기 위한 방법론을 종합적으로 리뷰합니다. 이를 통해 핵심 탐지 방법론을 체계적으로 분석하고, 공통점을 도출하여 유의미한 세분화로 분류했습니다. 대규모 데이터셋의 중요성도 강조되며, 공개적으로 사용 가능한 데이터셋 개요도 제공하여 추가 연구 및 벤치마킹의 기초를 마련합니다.

- **Performance Highlights**: 생성형 모델의 발전에도 불구하고 합성 이미지 탐지 연구는 일반화, 견고성, 확장성 등의 문제에 직면해 있습니다. 본 연구는 이러한 문제를 해결하기 위한 탐지 방법론의 진전을 요약하고 기존의 한계점을 식별하며 미래 방향성을 제시합니다. 궁극적으로, 생긴 위협적 요소가 증가하는 가운데 대응책 마련이 절실하다는 점을 강조합니다.



### M3-AGIQA: Multimodal, Multi-Round, Multi-Aspect AI-Generated Image Quality Assessmen (https://arxiv.org/abs/2502.15167)
Comments:
          14 pages, 5 figures. This work has been submitted to the IEEE for possible publication

- **What's New**: 최근 AI 이미지 생성 모델(AGI)의 품질 평가에 대한 새로운 접근 방식이 제안되었습니다. M3-AGIQA라는 포괄적인 프레임워크는 다중 모드(Multimodal), 다중 라운드(Multi-Round) 및 다중 측면(Multi-Aspect) 평가 메커니즘을 활용하여 AGI의 품질을 평가합니다. 이 프레임워크는 MLLM(Multimodal Large Language Model)을 텍스트 및 이미지 인코더로 사용하며, Low-Rank Adaptation(LoRA) 기법을 통해 고급 캡션 기능을 로컬 모델로 증류합니다.

- **Technical Details**: M3-AGIQA는 세 가지 주요 연구 질문에 대한 실험을 실시했습니다. 실험은 AGIQA-3k, AIGCIQA2023, AIGCIQA-20k라는 세 가지 공공 데이터 세트를 이용하여 진행되었으며, 각 데이터 세트는 품질, 일치성 및 진정성을 평가하는 평균 의견 점수(MOS)를 포함하고 있습니다. 이 프레임워크는 사전 훈련된 MLLM을 인코더로 사용하며, fine-tuning 과정에서 LoRA 기법과 GPU VRAM 사용 최소화를 위한 deepspeed ZeRO-3 오프로드를 적용했습니다.

- **Performance Highlights**: M3-AGIQA는 다양한 기준선 모델들과 비교하여 월등한 성능을 보였습니다. 간단한 비전 인코더 및 기존 IQA 방법들보다 AGIQA 방법이 특히 우수한 성능을 나타냈으며, 특히 일치성과 진정성 측면에서 성과가 두드러졌습니다. 실험 데이터를 통해 M3-AGIQA의 강력한 일반화 능력이 확인되었으며, SRCC와 PLCC와 같은 IQA 작업에서 널리 사용되는 두 가지 메트릭으로 모델의 효과성을 평가했습니다.



### HOpenCls: Training Hyperspectral Image Open-Set Classifiers in Their Living Environments (https://arxiv.org/abs/2502.15163)
- **What's New**: 이번 논문은 HOpenCls라는 새로운 프레임워크를 제안하여, 미분류된 '야생 데이터(wild data)'를 활용한 개방형 하이퍼스펙트럼 이미지(HSI) 분류를 수행합니다. 이러한 야생 데이터는 실제 환경에서 분류기를 배치하는 동안 자유롭게 수집할 수 있으며, 기존 데이터와 달리 추가적인 레이블 작업이 필요하지 않아 효율적입니다. 이 연구는 긍정-미분류(PU) 학습 문제로 HSI 개방형 분류를 재구성하여, 알려진 클래스와 미지의 클래스로의 분류를 처리하는 방법을 제시합니다.

- **Technical Details**: HOpenCls는 기존의 개방형 분류 문제에서 무작위로 수집된 야생 데이터를 활용하여, 긍정 및 미지 클래스만으로 이진 분류기를 학습하는 방식을 취합니다. 또한, Grad-C(그래디언트 수축)와 Grad-E(그래디언트 확장) 모듈을 도입하여, 야생 데이터의 그래디언트 가중치를 조정하고, 미지 클래스의 분류를 개선하는 데 기여합니다. 이 방법은 기존의 클래스 가중치 추정 문제를 피하면서, PU 학습을 통해 개방형 HSI 분류를 효과적으로 수행하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 HOpenCls는 복잡한 실제 환경에서 개방형 HSI 분류 성능을 크게 향상시켰습니다. WHU-Hi-HongHu 데이터셋을 예로 들면, HOpenCls는 강력한 기본선 대비 개방형 분류의 전체 정확도(Open OA)를 8.20% 향상시키고, 미지 클래스 거부 성능(F1U)은 38.91% 개선되었습니다. 이런 성과는 야생 데이터를 활용한 새로운 접근 방식의 유효성을 입증합니다.



### Confidence-Weighted Boundary-Aware Learning for Semi-Supervised Semantic Segmentation (https://arxiv.org/abs/2502.15152)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 논문에서는 세미-슈퍼바이즈드 시맨틱 분할(SSSS)을 위한 새로운 프레임워크인 CW-BASS를 제안합니다. 기존 SSSS 방법들이 직면한 문제, 예를 들어 coupling, confirmation bias 및 boundary blur를 해결하기 위해 다양한 기법을 통합하였습니다. 특히, 잘못된 예측의 영향을 줄이기 위해 pseudo-label에 신뢰도 가중치를 할당하고, 경계 인식 기술을 활용하여 정확도를 향상시킵니다.

- **Technical Details**: CW-BASS는 두 단계로 작동하며, 첫 번째 단계에서 교사 모델이 라벨이 없는 데이터에 대해 confidence score와 함께 pseudo-label을 생성합니다. 이어서 dynamic thresholding을 통해 낮은 신뢰도의 pseudo-label을 필터링하고, 두 번째 단계에서는 confidence decay 전략을 사용하여 낮은 신뢰도의 픽셀의 영향을 줄이며, boundary-aware 모듈을 통해 객체 경계 근처의 분할 정확도를 향상시킵니다. 이러한 과정은 모델 성능에 따라 지속적으로 조정됩니다.

- **Performance Highlights**: 실험 결과, 본 방법은 Pascal VOC 2012 및 Cityscapes 데이터셋을 사용하여 SOTA 성능을 달성하였습니다. 특히, 라벨된 데이터의 1/8 또는 12.5	eh%만 사용하더라도 Pascal VOC 2012에 대해 mIoU 75.81을 기록하며, 제한된 라벨 환경에서의 효과성을 입증하였습니다.



### TransMamba: Fast Universal Architecture Adaption from Transformers to Mamba (https://arxiv.org/abs/2502.15130)
- **What's New**: 이 논문은 기존 Transformer 모델에서 Mamba와 같은 새로운 아키텍처로의 지식 이전을 다루고 있습니다. 이를 통해 실행 비용과 시간 효율성을 크게 개선할 수 있는 가능성을 제시합니다. 특히, 새로운 두 단계의 전략을 통해 Cross-Architecture 학습을 촉진하며, 다중 모달 작업에서도 탁월한 성능을 보여줍니다.

- **Technical Details**: TransMamba의 핵심 기술적 요소는 Weight Subcloning과 Adaptive Bidirectional distillation (WSAB) 방법을 적용하여 서로 다른 아키텍처 간의 지식 이전을 용이하게 만드는 것입니다. 중간 기능을 정렬된 잠재 공간으로 투영한 뒤, 비대칭적 레이어 수에 구애받지 않고 지식을 전이하는 과정을 통해 Mamba 아키텍처에 언어 인식을 통합합니다. 또한, cosine similarity 기반의 지식 전이 방법을 통해 레이어 간의 최적화 문제를 해결합니다.

- **Performance Highlights**: TransMamba는 기존의 독립적으로 모델을 훈련할 때 요구되는 훈련 데이터의 75%도 사용하지 않고도 다양한 네트워크 아키텍처와 다운스트림 작업에서 뛰어난 성능을 자랑합니다. 이미지 분류(image classification), 시각적 질문 응답(visual question answering), 텍스트-비디오 검색(text-video retrieval)과 같은 작업에서 우수한 결과를 달성하며, 제안된 코드도 공개될 예정입니다.



### DAM-Seg: Anatomically accurate cardiac segmentation using Dense Associative Networks (https://arxiv.org/abs/2502.15128)
Comments:
          12 pages, 7 figures, 5 tables

- **What's New**: 이번 연구에서는 심장 이미지 분할(cardiac segmentation)에서의 한계를 극복하기 위해 새로운 transformer 기반 아키텍처를 제안합니다. 기존의 방법들처럼 복잡한 보조 모듈을 사용하지 않고, 심장 입력의 패턴을 학습하고 유지할 수 있는 밀집 연관 메모리(dense associative memory) 구조를 도입하였습니다. 이 방법은 네트워크가 제한된 패턴만을 기억하게 하여 해부학적 정확도를 보장하는 동시에, 가시성이 떨어지는 경우에도 향상된 견고성(robustness)을 보여줍니다.

- **Technical Details**: 제안된 모델은 입력과 무관하게 입력을 메모리에 저장하고, 그 정보를 바탕으로 해부학적으로 일관된 분할 마스크를 생성합니다. 학습 가능한 메모리 변환 행렬을 활용하여 정적 메모리 공간을 쿼리 및 값 행렬로 변환함으로써, 패턴의 유연한 표현을 가능하게 합니다. 또한, 메모리 업데이트 메커니즘을 개선하여 일반화 가능한 패턴에 집중하게 하고 메타 안정 상태(meta-stable states)를 도입해 견고성을 높였습니다.

- **Performance Highlights**: CAMUS와 CardiacNet의 두 공용 데이터 세트를 통해 성능을 평가한 결과, 제안된 모델이 기존 방법들보다 모든 메트릭에서 일관되게 뛰어난 성능을 보였습니다. 이 연구는 심장 분할 작업의 효과적이고 신뢰할 수 있는 솔루션을 제공함으로써, 더 나아가 의료 진단 및 치료 계획에 기여할 것으로 기대됩니다.



### Can Hallucination Correction Improve Video-Language Alignment? (https://arxiv.org/abs/2502.15079)
- **What's New**: 본 논문에서는 비디오와 텍스트 간의 정렬을 개선하기 위한 새로운 접근법으로 HACA(하위 오류 수정 기반의 비디오-언어 정렬)를 도입하였습니다. 이 방법은 비디오의 내용과 일치하지 않는 설명의 오류를 수정하는 자기 학습(self-training) 프레임워크를 활용합니다. 이를 통해 모델은 비디오와 텍스트의 조화로운 표현을 강화할 수 있습니다.

- **Technical Details**: HACA는 고전적인 엔탤먼트(entailment) 학습 기법을 넘어서, 텍스트와 비디오 간의 불일치를 예측하고 이를 수정하는 임무를 통해 정렬을 최적화합니다. 이 모델은 비디오 특성에 맞춰 데이터를 증강하기 위해 마스킹 수정 작업을 도입하여, 비디오-언어 모델의 학습을 보다 세밀하게 수행합니다. 기술적으로는 비디오-LLM의 블록을 최적화하고, 텍스트 디코더와 어댑터를 조정함으로써 전반적인 성능을 높이고 있습니다.

- **Performance Highlights**: HACA를 통해 최적화된 모델은 기본 모델 대비 최대 17.9%의 정확도 향상을 보이며, 5.7 mAP 포인트 이상을 달성하였습니다. 이러한 결과는 HACA가 비디오-텍스트 간의 정렬을 효과적으로 개선하며, 다양한 다운스트림 우선 과제에서도 뛰어난 성능을 발휘함을 보여줍니다.



### Hardware-Friendly Static Quantization Method for Video Diffusion Transformers (https://arxiv.org/abs/2502.15077)
- **What's New**: 이 논문에서는 OpenSora라는 비디오 확산 변환기(Video Diffusion Transformer)의 사후학습 양자화(post-training quantization) 방법을 제안합니다. 기존의 동적 양자화(dynamical quantization) 기술에 의존하지 않고, 정적 양자화(static quantization)을 통해 모델을 효율적으로 배포할 수 있습니다. 특히, CLIP와 VQA 지표를 사용하여 FP16 및 동적 양자화된 ViDiT-Q 방법과 유사한 비디오 품질을 달성했습니다.

- **Technical Details**: 제안된 방법은 단계별 보정(calibration data)을 사용하여 각 시간 단계에 대해서 적절한 정적 양자화 모델을 제공합니다. 이 과정에서는 가중치에 대한 채널 기반 양자화(channel-wise quantization)와 활성에 대한 텐서 기반 양자화(tensor-wise quantization)를 적용합니다. 또한, 부드러운 양자화(smooth quantization) 기법을 통해 정적 양자화된 모델에서도 고품질 비디오 출력을 얻을 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과는 정적 양자화가 동적 양자화에 대한 실제적인 대안이 될 수 있음을 입증하고 있으며, 성능을 저하시키지 않으면서도 더욱 효율적인 접근 방식을 제공합니다. 특히, CW, TW 및 ASQ를 사용한 집계 정적 양자화 방법은 동적 양자화 및 FP16 모델과 시각적 품질이 유사하다는 것을 나타냅니다. 제안된 방법은 다양한 정밀도 수준에서 뛰어난 결과를 보여주며, 생성된 비디오와의 정합성 또한 더욱 향상되었습니다.



### Synth It Like KITTI: Synthetic Data Generation for Object Detection in Driving Scenarios (https://arxiv.org/abs/2502.15076)
Comments:
          Preprint, to appear in ROBOVIS 2025

- **What's New**: 이번 연구에서는 자율주행 시스템에서 중요한 요소인 시뮬레이션을 다룬다. 특히, 가상 세계와 현실 세계 간의 전이 가능성(transferability) 문제를 다시 검토하고, CARLA 시뮬레이터에 기반한 데이터셋 생성 파이프라인을 제안한다. 이를 통해 리얼 월드에서 높은 일반화 능력을 갖춘 3D 객체 탐지(Object Detection) 시스템의 구현 가능성을 제시한다.

- **Technical Details**: 연구팀은 도메인 무작위화(domain randomization) 전략과 세심한 모델링을 활용하여 합성 데이터(synthetic data)에서 객체 탐지기를 훈련시킨다. 다양한 가상 센서 변형을 비교하여 어느 센서 속성이 도메인 간의 격차(domain gap)에 중요한 영향을 미치는지에 대한 통찰력을 모은다. 이 연구는 KITTI 데이터셋에 대한 일반화 능력을 입증하며, 현실 데이터로의 세부 조정(fine-tuning)을 통해 성능을 개선하였다.

- **Performance Highlights**: 최종적으로, 소량의 실제 데이터로 세부 조정을 진행한 결과, 기준 성능과 거의 일치하며, 전체 훈련 세트를 사용했을 때 기준 성능을 약간 초과하는 성능을 기록했다. 이는 가상 데이터로 훈련된 모델의 잠재력을 보여주며, 자율주행 시스템의 발전을 위한 기반을 마련한다.



### Simpler Fast Vision Transformers with a Jumbo CLS Token (https://arxiv.org/abs/2502.15021)
- **What's New**: 이번 논문에서는 비전 트랜스포머(Vision Transformers, ViTs)의 글로벌 처리 성능을 향상시키기 위해 새로운 접근법인 Jumbo를 소개합니다. Jumbo는 더 넓은 CLS 토큰을 생성하여 패치 토큰의 너비에 맞게 분할한 후, self-attention 과정을 통해 처리합니다. 이 방법은 ViT의 인프라적 장점을 유지하면서도 기존의 효율적인 컴퓨팅 모델을 초월하는 성과를 보입니다.

- **Technical Details**: Jumbo 방식은 CLS 토큰을 넓히고, attention을 처리한 후에 이를 재조합하는 구조입니다. 이후 전용의 더 넓은 FFN(Feed Forward Network)을 이 토큰에 적용하여 정확성을 높입니다. 이러한 설정은 ViT-tiny와 ViT-nano 모델에서 ImageNet-1K 데이터셋에 대해 각각 3.2% 및 13.5%의 성능 향상을 이끌어냅니다.

- **Performance Highlights**: 특히, Jumbo 모델은 특수한 계산 효율 모델보다도 더 나은 성능을 보여주며, 비트-스몰(ViT-small)에서는 ImageNet-1K에서 눈에 띄는 성과가 없지만, ImageNet-21K에서는 3.4% 상승한 결과를 나타냅니다. 이는 Jumbo가 특정 작업에 비해 ViT가 너무 좁은 경우에 가장 효과적임을 시사합니다.



### CrossOver: 3D Scene Cross-Modal Alignmen (https://arxiv.org/abs/2502.15011)
Comments:
          Project Page: this http URL

- **What's New**: CrossOver는 다중 모달 3D 장면 이해를 위한 새로운 프레임워크로, 모든 모달리티에 대해 완전한 데이터 가용성과 강직한 정렬을 가정하지 않습니다. 이 방법은 RGB 이미지, 포인트 클라우드, CAD 모델, 플로어플랜, 텍스트 설명과 같은 다양한 모달리티를 유연하게 정렬하여 통합된 모달리티 불가지론적인 임베딩 공간을 학습합니다. 이를 통해 실제 환경에서의 데이터가 불완전한 경우에도 강력한 장면 검색과 객체 위치 확인이 가능합니다.

- **Technical Details**: CrossOver는 1D, 2D, 3D 인코더를 포함하는 다차원 특화 인코더를 도입하여 각 모달리티의 차원에 맞춘 최적의 특성 추출을 수행합니다. 세 단계로 구성된 훈련 파이프라인은 객체 수준의 임베딩을 촉진하고 장면 수준의 통합 표현을 개발하며, 이를 통해 메타데이터 없이도 모달리티 간 임베딩을 생성합니다. 이러한 구조는 명시적 객체 세분화 정보 없이도 동작 가능하게 합니다.

- **Performance Highlights**: ScanNet 및 3RScan 데이터셋에 대한 평가 결과, CrossOver는 다양한 메트릭에서 우수한 성능을 보여줍니다. 이는 실제 3D 장면 이해 애플리케이션에서의 적응성을 강조하며, 모달리티 간의 유연한 상호작용을 지원하는 능력을 나타냅니다. 실세계의 데이터 결핍 문제를 효과적으로 해결하는 방법론으로 자리잡을 수 있습니다.



### A Rapid Test for Accuracy and Bias of Face Recognition Technology (https://arxiv.org/abs/2502.14996)
Comments:
          Accepted as a conference paper for WACV 2025. Manuel Knott, Ignacio Serna, and Ethan Mann contributed equally

- **What's New**: 본 연구에서는 얼굴 인식 시스템(FR)의 정확성을 측정하기 위한 새로운 방법을 제안합니다. 이 방법은 수동 주석 없이도 FR 시스템을 빠르게 기준 설정할 수 있는 방법으로, 웹 검색 결과에서 가져온 근사 레이블을 사용합니다. 또한, 다섯 가지 FR 클라우드 서비스의 첫 번째 공공 벤치마크를 소개하며, 인종적 편향, 특히 아시아 여성에 대한 낮은 정확성을 드러냅니다.

- **Technical Details**: 제안된 방법은 공공 데이터를 활용하여 검증 가능성을 높이며, 이미지를 실시간으로 분석하여 저장하지 않습니다. 기존의 인식 알고리즘의 신뢰도 값을 기반으로 얼굴 정체성의 기준 진실 레이블을 추론하는 알고리즘적 단계가 핵심입니다. 이 방법은 인공지능과 얼굴 인식 기술이 공정하고 투명한 기관으로 발전하는 데 기여할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 FR 시스템의 정확성을 신뢰성 있게 추정하고 순위를 매길 수 있으며, 이로 인해 수동 레이블링의 시간과 비용이 크게 줄어듭니다. 실험적으로 검증된 이 방법은 더욱 저렴하고 실용적인 관찰 방법으로, 동시 비교를 통해 다섯 가지 인기 있는 클라우드 서비스의 정확성과 편향을 비교할 수 있는 최초의 공공 벤치마크를 제공합니다.



### LAVID: An Agentic LVLM Framework for Diffusion-Generated Video Detection (https://arxiv.org/abs/2502.14994)
- **What's New**: 최근 AI로 생성된 콘텐츠 탐지가 이미지 분야에서는 활발히 연구되고 있으나, 비디오 분야는 아직 탐색되지 않았습니다. 이 논문에서는 LAVID라는 새로운 LVLM(대형 비전 언어 모델) 기반 비디오 탐지 시스템을 제안합니다. LAVID는 사전 지식(toolkit)과 구조화된 프롬프트를 활용하여 고품질 비디오에서 AI 생성 콘텐츠를 효과적으로 탐지할 수 있습니다.

- **Technical Details**: LAVID는 LVLM의 강력한 추론 능력을 활용하여 비디오 내용을 이해합니다. 본 시스템은 EK(명시적 지식) 도구를 자동으로 선택하고, 프롬프트를 온라인으로 조정해 LVLM이 비디오를 분석하는 데 도움을 줍니다. 구조화된 프롬프트를 통해 시각적으로 해석 가능성을 개선하고 오탐지 현상을 줄이도록 설계되었습니다.

- **Performance Highlights**: LAVID는 새로운 벤치마크와 함께 1.4k개 이상의 고품질 가짜 비디오를 생성하였습니다. 이 시스템은 세 가지 최첨단 LVLM에서 높은 데이터 세트에 대해 F1 점수를 9.4%에서 25.9%까지 개선했습니다. 이 연구 결과는 LAVID가 동영상 생성 탐지 작업에서의 가능성을 보여주는 것을 목표로 합니다.



### Few-shot Species Range Estimation (https://arxiv.org/abs/2502.14977)
- **What's New**: 본 논문에서 제안하는 FS-SINR은 제한된 데이터로부터 종의 범위를 추정하는 새로운 방법을 소개합니다. 기존의 모델들은 많은 관측 데이터에 의존했으나, FS-SINR은 적은 수의 관측자료로도 신뢰할 수 있는 예측을 가능하게 합니다. 이 모델은 또한 훈련 세트에 포함되지 않은 종에 대한 예측도 가능하여 대화식 탐색 및 모델링이 가능합니다.

- **Technical Details**: FS-SINR은 Transformer 기반 모델로, 관측된 지리적 위치 세트를 입력으로 받아 종의 범위를 추정합니다. 이 과정에서 추가적인 메타데이터(예: 텍스트 요약 또는 이미지)를 유연하게 통합하여 예측 품질을 더욱 향상시킬 수 있습니다. 이 방법은 IUCN과 S&T 벤치마크 데이터셋에서 검증되었으며, 우수한 성과를 보였습니다.

- **Performance Highlights**: FS-SINR은 적은 수의 샘플에서 최신 성능을 발휘함을 보여주며, 이전의 알고리즘에 비해 월등한 결과를 달성하였습니다. 특히, 373,000종의 대부분이 적은 관측 데이터를 가진 경우에도 높은 정확도로 범위를 예측할 수 있는 가능성을 제시합니다. 다양한 관측 데이터를 통합하여 더 나은 예측을 이끌어내는 것이 FS-SINR의 큰 강점입니다.



### KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding (https://arxiv.org/abs/2502.14949)
Comments:
          17 pages, 5 figures, ACL 2025

- **What's New**: 이 논문은 KITAB-Bench라는 포괄적인 아랍어 OCR 벤치마크를 소개합니다. 아랍어 OCR의 평가 시스템에서의 공백을 메우기 위해 8,809개의 샘플을 수집하여 9개의 주요 도메인과 36개의 하위 도메인으로 구성하였습니다. 이 벤치마크는 다양한 문서 유형을 포함하여, 손글씨 텍스트, 구조화된 표, 그리고 비즈니스 인텔리전스를 위한 21가지 차트 유형에 대한 특화된 내용을 제공합니다.

- **Technical Details**: KITAB-Bench는 레이아웃 탐지(text blocks, tables, figures), 다중 형식 인식(printed/handwritten text, charts, diagrams), 그리고 구조화된 출력 생성(HTML tables, DataFrame charts, markdown)을 평가하는 시스템을 채택합니다. 이 방법론은 OCR 성능을 정밀하게 평가할 수 있는 프레임워크를 제공하며, 챠트 추출(CharTeX)과 도표 추출(CODM) 평가 지표를 포함하여 다양한 문서 이해 과제를 다룹니다.

- **Performance Highlights**: 최신 비전-언어 모델(GPT-4, Gemini, Qwen)이 전통적인 OCR 방법(EasyOCR, PaddleOCR, Surya)에 비해 평균 60% 더 높은 Character Error Rate(CER)에서 우수성을 보였습니다. 그러나 현재 아랍어 OCR 모델의 한계로는 PDF에서 Markdown으로의 변환에서 가장 우수한 모델인 Gemini-2.0-Flash가 단 65%의 정확도만을 달성했습니다. 이러한 결과는 아랍어 텍스트 인식을 위한 복잡한 글꼴, 숫자 인식 오류, 단어 신장 및 표 구조 감지의 문제를 강조합니다.



### FacaDiffy: Inpainting Unseen Facade Parts Using Diffusion Models (https://arxiv.org/abs/2502.14940)
Comments:
          Accepted for GeoSpatial Week 2025, ISPRS Annals

- **What's New**: FacaDiffy는 기존의 3D 건물 모델과 레이저 스캐닝 포인트 클라우드를 사용하여 2D conflict map을 생성하는 새로운 방법을 제시합니다. 이 모델은 개인화된 Stable Diffusion 모델을 통해 보이지 않는 파사드 부분을 채우는 방식으로 결함 있는 conflict map을 완성합니다. 또한, 임의의 도시 모델 생성기와 주석이 있는 파사드 이미지를 사용하여 합성 conflict maps를 생성하는 확장 가능한 파이프라인을 개발했습니다.

- **Technical Details**: FacaDiffy는 기존의 LoD 2 모델과 레이저 스캐닝 데이터를 통해 2D conflict map을 생성하는 결정론적 방법을 사용합니다. 이후, Stable Diffusion 모델을 개인화하여 이 2D conflict maps에 파사드 객체를 자동으로 추가하는 inpainting 작업을 수행합니다. 이 과정에는 또한 Dreambooth를 활용하여 합성된 conflict map 데이터를 기반으로 모델을 훈련하는 기술이 포함됩니다.

- **Performance Highlights**: FacaDiffy는 다양한 inpainting 기준선과 비교하여 conflict map 완성도에서 최첨단 성능을 발휘하며, 고해상도 3D 의미론적 건물 재구성 시 완성된 conflict map을 적용할 경우 탐지율이 22% 향상됩니다. 이러한 성과는 고급 재구성에 있어 FacaDiffy의 적용 가능성을 더욱 확대합니다.



### Online hand gesture recognition using Continual Graph Transformers (https://arxiv.org/abs/2502.14939)
- **What's New**: 이 논문은 실시간 손 제스처 인식을 위한 새로운 온라인 인식 시스템을 제안합니다. 기존의 분할 기반 인식 방식에서 벗어나, 스켈레톤(sequence of 3D coordinates) 데이터의 실시간 스트리밍을 처리할 수 있는 방법을 제시합니다. 이 시스템은 Spatial Graph Convolutional Networks (S-GCN)과 Transformer 기반의 Graph Encoder (TGE)를 결합하여 공간적 및 시간적 특성을 효과적으로 추출합니다.

- **Technical Details**: 본 연구에서는 Hybrid architecture를 통해 공간적 특성을 S-GCN으로 추출하고, 프레임 간의 시간적 의존성을 TGE를 통해 캡처합니다. 또한, 모델이 진화하는 데이터 분포에 적응할 수 있도록 지속적인 학습 메커니즘(continual learning mechanism)을 도입하여 동적 환경에서의 인식을 강화합니다. 실험은 SHREC'21 벤치마크 데이터셋에서 수행되었으며, 최첨단 성능을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 온라인 손 제스처 인식에서 뛰어난 성능을 발휘하며, 기존 기법들보다 높은 정확도를 달성하였습니다. 또한, 잘못된 긍정 비율(false positive rates)을 상당히 줄어들게 하여, 실시간 응용에 적합한 솔루션으로 자리 잡고 있습니다. 이 시스템은 인간 로봇 협업이나 보조 기술 등 다양한 분야에 통합될 수 있습니다.



### GS-Cache: A GS-Cache Inference Framework for Large-scale Gaussian Splatting Models (https://arxiv.org/abs/2502.14938)
- **What's New**: 이 논문에서는 GS-Cache라는 새로운 프레임워크를 제안합니다. GS-Cache는 3D Gaussian Splatting(3DGS)과 최적화된 렌더링 시스템을 통합하여, 소비자-grade 장치에서의 실시간 고충실도 성능을 가능하게 합니다. 캐시 중심 파이프라인과 멀티 GPU 렌더링 기능을 도입하여, 렌더링 효율성을 크게 향상시킵니다.

- **Technical Details**: GS-Cache는 비복잡한 3DGS 모델의 렌더링을 위해 설계된 컴퓨테이션 프레임워크입니다. 특히, 대도시 규모의 장면을 처리하기 위해 계산 효율성을 극대화하고 메모리 사용량을 최소화합니다. CUDA 커널을 최적화하여 렌더링 병목 문제를 해결하고, GPU 리소스를 동적으로 할당하여 안정적인 FPS를 유지합니다.

- **Performance Highlights**: GS-Cache는 최대 5.35배의 성능 개선을 이루었으며, 35%의 지연 시간 감소와 42%의 낮은 GPU 메모리 사용을 지원합니다. 본 프레임워크는 2K 이진 안경 렌더링을 120 FPS 이상의 높은 시각 품질로 제공하며, VR 시스템에서의 새로운 가능성을 열어줍니다.



### RAPTOR: Refined Approach for Product Table Object Recognition (https://arxiv.org/abs/2502.14918)
Comments:
          Accepted for WACVW 2025 (VisionDocs)

- **What's New**: 이번 연구에서는 RAPTOR라는 모듈형 후처리 시스템을 도입하여 기존의 테이블 추출 기술, 특히 제품 테이블에 대한 성능을 향상시키고자 하였습니다. 기존 DEtection TRansformer (DETR)와 TAble TRansformer (TATR) 기반의 모델들이 직면한 다양한 테이블 형식의 문제들을 해결하고, 정밀도와 구조 예측 모두에서 개선을 도모합니다. RAPTOR는 특히 이커머스와 관련된 비즈니스 문서 분석에 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: RAPTOR는 Genetic Algorithm을 활용하여 모듈의 매개변수를 최적화하고, ICDAR 2019 및 다양한 공개 데이터셋에서 사전훈련된 DETR과 TATR 모델을 사용하여 테이블 영역 감지(Table Detection, TD) 및 테이블 구조 인식(Table Structure Recognition, TSR)을 수행합니다. 이 모듈형 시스템은 기존 모델을 재조정하는 대신, 제한된 데이터로 모델 매개변수를 학습할 수 있는 기회를 제공합니다. 기존의 문제 해결을 위한 컴포넌트를 통합함으로써 TD 및 TSR 과정에서의 성능을 향상시킵니다.

- **Performance Highlights**: RAPTOR는 다섯 개의 데이터셋에서 평가되었으며, 특히 제품 테이블에서의 성능 향상이 두드러졌습니다. 다양한 테이블 형식에서도 전반적으로 합리적인 성능을 유지했으며, 여러 데이터셋에 따라 기본 모델의 성능을 개선하는 데 기여했습니다. 연구팀은 비즈니스 테이블에서 나타나는 일반적인 오류 유형을 식별하고 이를 해결하기 위한 모듈형 시스템으로서 RAPTOR의 유용성을 확인하였습니다.



### Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning (https://arxiv.org/abs/2502.14917)
- **What's New**: 이번 논문에서는 Sce2DriveX라는 새롭고 인간 유사한 driving chain-of-thought (CoT) 추론 MLLM 프레임워크를 제안합니다. 이 프레임워크는 다중 모드 학습(multi-modal joint learning)을 통해 지역 장면 비디오(local scene videos)와 전체적 BEV 맵(global BEV maps)을 활용하여 장기적 시공간 관계(long-range spatiotemporal relationships)와 도로Topology를 깊숙이 이해합니다. 이를 통해 3D 정적/동적 장면에서의 종합적인 인식(perception) 및 추론(reasoning) 능력을 향상시키며, 다양한 장면에서도 자율 주행의 일반화를 달성합니다.

- **Technical Details**: Sce2DriveX는 다중 뷰 장면 비디오(multi-view scene videos)와 BEV 맵을 정렬하여 통합된 시각적 특징 공간(visual feature space)을 형성하고, 이를 텍스트 임베딩 공간(text embedding space)으로 매핑하여 자연어 응답을 생성하는 구조를 갖추고 있습니다. 모델은 장면 이해(scene understanding), 동작 분석(behavior analysis), 모션 계획(motion planning), 차량 제어(vehicle control) 등의 분야에서 강력한 성능을 보입니다. 또한, 이에 대한 효과적인 훈련을 위해 3D 공간 이해(3D spatial understanding) 및 장축 작업 추론(long-axis task reasoning)에 특화된 종합적인 Visual Question Answering (VQA) 데이터셋을 구축했습니다.

- **Performance Highlights**: Sce2DriveX는 CARLA Bench2Drive 벤치마크에서 장면 이해(scene understanding)부터 엔드 투 엔드 드라이빙(end-to-end driving)까지 모든 작업에서 최고의 성능을 달성했습니다. 광범위한 실험을 통해 복잡한 시나리오에서도 강력한 일반화 성능을 보이며, 자율 주행 시스템의 의사 결정 과정이 인간의 인지(cognitive processes)와 일치하도록 하는 데 기여하고 있습니다.



### What Is a Good Caption? A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Coverage of MLLMs (https://arxiv.org/abs/2502.14914)
Comments:
          Work in progress

- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전으로 기존 비주얼 캡셔닝 기준이 뒤처지게 되었습니다. 기존 기준들은 주로 짧은 설명과 오래된 메트릭으로 한정되어 있었습니다. 이 문제를 해결하기 위해 CV-CapBench라는 종합적인 비주얼 캡션 기준을 제안하며, 6개 관점과 13개 차원에 걸쳐 캡션 품질을 체계적으로 평가합니다.

- **Technical Details**: CV-CapBench는 각 차원에 대해 정확도(accuracy), 재현율(recall), 적중률(hit rate) 메트릭을 도입하여 결과의 정확성과 coverage을 독특하게 평가합니다. 특히, 동적 및 지식 집약적 차원에서 MLLMs의 성능 격차가 뚜렷하게 나타났습니다. 이 구성이 정적(static) 차원 9개와 동적(dynamic) 차원 4개로 나뉘어지며, 직관적으로 시각적 캡션의 포괄성을 강화하고 있습니다.

- **Performance Highlights**: CV-CapBench를 활용한 실험 결과, 여러 주요 MLLMs가 여전히 특정 차원에서 성과를 내기 어려운 것으로 나타났습니다. 이러한 발견은 향후 비주얼 캡셔닝 개선을 위한 실질적인 통찰을 제공합니다. 논문에서 제안된 코드는 공개될 예정으로, 심도 있는 연구가 기대됩니다.



### PTB-Image: A Scanned Paper ECG Dataset for Digitization and Image-based Diagnosis (https://arxiv.org/abs/2502.14909)
- **What's New**: 본 연구에서는 PTB-Image라는 이름의 새로운 데이터셋을 소개하며, 이는 종이 ECG와 그에 상응하는 디지털 신호로 구성되어 있습니다. 이는 ECG의 디지털화 연구를 위한 기초 자료를 제공하여, 고전적인 종이 기반 ECG의 자동 분석 문제를 해결할 수 있는 가능성을 엽니다. 또한 VinDigitizer라는 기본 디지털화 방법을 통해 종이 ECG를 디지털 시계열 신호로 변환하는 방식이 제안되었습니다.

- **Technical Details**: PTB-Image 데이터셋은 549개의 종이 ECG 기록을 포함하며, 각 기록은 12개의 동시 ECG 신호로 구성되어 있습니다. VinDigitizer는 세 단계로 구성된 파이프라인을 통해 신호를 분리하고, 배경에서 파형을 추출하며, 디지털 신호로 재구성하는 과정을 수행합니다. 이 방법론은 YOLOv8 모델을 사용하여 신호가 포함된 행을 정확하게 감지하며, Otsu의 임계값을 활용하여 신호의 경계를 정리합니다.

- **Performance Highlights**: VinDigitizer를 통해 얻은 평균 신호 대 잡음비(SNR)는 0.01 dB로, 이는 종이 ECG의 디지털화 과정에서 발생할 수 있는 왜곡 문제를 강조합니다. PTB-Image 데이터셋을 기반으로 한 연구는 의료 기록 통합과 AI 모델 훈련을 위한 기초 자료를 제공하여, 원거리 진료 및 자동화된 심장진단 분야에서의 발전을 지원할 수 있는 잠재력을 가집니다.



### KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models (https://arxiv.org/abs/2502.14908)
- **What's New**: 이번 연구에서는 Vision Language Models (VLMs)의 멀티모달 환경에서의 지식 갈등에 대한 영향을 조사하기 위해 \

- **Technical Details**: \

- **Performance Highlights**: \



### UPCMR: A Universal Prompt-guided Model for Random Sampling Cardiac MRI Reconstruction (https://arxiv.org/abs/2502.14899)
Comments:
          Accepted paper for STACOM 2024

- **What's New**: 이 논문에서는 심장 자기공명영상(CMR) 복원을 위한 보편적인 모델인 UPCMR을 소개합니다. UPCMR은 두 가지 유형의 학습 가능한 프롬프트를 통합하여 다양한 언더샘플링(k-space에서의 불완전 샘플링) 방식에 적응할 수 있도록 설계되었습니다. 이 접근 방식은 UNet 구조와 결합된 각 블록에서 효과적인 품질 향상을 보여줍니다.

- **Technical Details**: UPCMR 모델은 다중 코일 언더샘플링 k-space 측정값으로부터 복잡한 값을 가지는 MR 영상 시퀀스를 재구성하는 것을 목표로 합니다. 이 모델은 언더샘플링 특화 프롬프트와 공간 특화 프롬프트를 제공합니다. 또한 모델은 k-space 궤적 및 가속 요인을 사전 정보로 활용하여 학습 가능한 프롬프트 풀이 효율적으로 설계되었습니다.

- **Performance Highlights**: UPCMR 모델은 CMRxRecon2024 챌린지 데이터셋을 사용하여 모든 무작위 샘플링 시나리오에서 재구성된 이미지 품질이 크게 향상되었습니다. 기존의 전통적인 방법들과 비교하여 강력한 적응 가능성을 보여주며, 이는 다양한 샘플링 모드에서 효과적인 훈련 전략 덕분입니다. 결과적으로 UPCMR은 심장 MRI 재구성 작업에 있어 뛰어난 성능을 입증하였습니다.



### A Comprehensive Survey on Concept Erasure in Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.14896)
- **What's New**: 이 연구는 Text-to-Image (T2I) 모델의 개념 지우기(concept erasure) 기술에 대한 구조적이며 포괄적인 설계를 제시합니다. 특히 개념 지우기 방법론을 최적화 전략과 수정되는 아키텍처 구성 요소에 따라 분류하고 있습니다. 또한, T2I 모델의 생성 기능에서 요구되는 개념을 체계적으로 제거하여 저작권이 있는 스타일이나 민감한 이미지를 생성하지 않도록 하려는 목적을 설명합니다.

- **Technical Details**: 본 논문은 Stable Diffusion (SD) 모델을 중심으로 T2I diffusion 모델의 구조를 설명합니다. SD는 이미지 복원을 위한 비전 디코더, 반복적인 노이즈 제거를 위한 잠재적 확산 모델, 텍스트 프롬프트를 조건 벡터로 변환하는 조건부 텍스트 인코더로 구성되어 있습니다. 개념 지우기 방법은 이러한 모델 내부 구성 요소를 수정하거나 추론 과정에 개입하여 민감한 또는 제한된 개념의 재현을 방지합니다.

- **Performance Highlights**: 저자들은 개념 지우기 기법의 효과를 평가하기 위한 평가 메트릭과 데이터셋을 통합하여 앞으로의 연구 방향에 대한 기초를 제공합니다. 연구는 적대적 공격과 방어 전략에 대한 분석을 포함하고 있으며, T2I 모델의 견고성을 높이기 위한 새로운 접근 방법을 탐구합니다. 이러한 작업들은 개념 지우기의 발전과 향후 방향에 대한 귀중한 통찰력을 제공하는 데 기여합니다.



### High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian Representation (https://arxiv.org/abs/2502.14895)
Comments:
          Accepted as an Oral paper at ICLR 2025. Project page: this https URL

- **What's New**: 이번 연구에서는 기상 예보에서 중요한 역할을 하는 3D 레이더 시퀀스 예측을 위한 포괄적인 프레임워크를 제안합니다. 이는 SpatioTemporal Coherent Gaussian Splatting (STC-GS)이라는 새로운 기법을 통해 동적 레이더 표현을 제시하며, GauMamba를 사용하여 효과적이고 정확한 예측을 수행합니다. STC-GS는 기존의 4D Gaussian 대신 3D 씬을 최적화하여, 연속적인 프레임에서 Gaussian 구조물의 움직임을 정밀하게 잡아냅니다.

- **Technical Details**: STC-GS는 초기 프레임을 3D Gaussian의 그룹으로 재구성하고, 이후 프레임에서 이러한 기준점을 모니터링하여 레이더 시퀀스의 기본 모션 트렌드를 포착합니다. 또한, GauMamba는 Mamba 프레임워크에 메모리 메커니즘을 통합하여 Gaussian 그룹의 시계열 변화를 모델링하고, 많은 Gaussian 토큰을 효율적으로 관리합니다. 이로써 다양한 동적 기상 레이더 신호를 예측하는 데 필요한 효율성과 정확성을 달성합니다.

- **Performance Highlights**: 실험 결과 STC-GS는 기존 3D 표현 방법에 비해 16배 이상의 공간 해상도로 3D 레이더 시퀀스를 효과적으로 표현할 수 있음을 보여주었습니다. 또한, GauMamba는 고속 동적 기상 조건의 예측에서 최신 방법들을 초월하는 성능을 보였습니다. 특히 평균 절대 오차(MAE)에서 19.7% 및 50% 감소를 달성하며, 중요한 레이더 신호가 있는 지역 예측에서도 현저한 개선을 보였습니다.



### FOCUS on Contamination: A Geospatial Deep Learning Framework with a Noise-Aware Loss for Surface Water PFAS Prediction (https://arxiv.org/abs/2502.14894)
- **What's New**: 이 연구는 PFAS 오염 예측을 위한 지리공간 딥러닝 프레임워크인 FOCUS를 소개합니다. FOCUS는 label noise-aware loss function을 활용하여 방대한 지역에서 PFAS 오염을 예측할 수 있도록 설계되어 있습니다. 이 모델은 수문학적 흐름 데이터, 토지 이용 정보, 그리고 이미 알려진 PFAS 원천과의 거리 데이터를 통합하여 예측 정확성을 개선합니다.

- **Technical Details**: FOCUS는 PFAS 오염의 세분화를 위해 래스터 데이터를 사용하는 지리공간 딥러닝 모델을 적용합니다. 이 프레임워크는 PFAS 확산에 대한 도메인 전문 지식을 활용하여 데이터 격차를 자동으로 보완하고, 이러한 가정을 신뢰도에 따라 가중치를 두어 왼쪽으로 캐싱하는 점이 특징입니다. 모델의 성능은 다양한 AI 모델 및 기존의 과학적 방법들과 비교하여 검증되었습니다.

- **Performance Highlights**: FOCUS의 평가 결과는 대규모 PFAS 모니터링을 위한 스케일 가능성, 정확성 및 강건성을 강조합니다. 이 연구는 환경 연구에 적극적으로 참여하는 비영리 단체와 수질 모델 전문 학자와 함께 진행되어, 최신 전문성과 실제 환경 문제를 반영합니다. FOCUS는 PFAS 오염의 복잡성을 해결하기 위해 AI와 지리공간 모델링의 최근 발전을 통합하여 새로운 발견을 가능하게 합니다.



### NOTA: Multimodal Music Notation Understanding for Visual Large Language Mod (https://arxiv.org/abs/2502.14893)
- **What's New**: 이번 논문에서는 NOTA라는 이름의 대규모 종합 멀티모달 음악 표기법 데이터셋을 처음으로 제안합니다. 이 데이터셋은 1,019,237개의 레코드로 구성되어 있으며, 세 가지 전 세계 지역에서 수집되었습니다. 데이터셋은 음악 정보 추출, 크로스 모달 정렬 테스트, 음악 표기법 분석의 세 가지 주요 작업을 다룹니다.

- **Technical Details**: NOTA 데이터셋은 ABC 표기법을 사용하여 음악 점수를 표현합니다. 데이터셋은 헤더와 본체로 구성되어 있으며, 헤더에는 참조 번호와 제목, 박자 기호, 기본 음표 길이 등이 포함됩니다. 본체는 주로 음표와 마침표를 포함하여 음악의 구조적 요소를 표현합니다.

- **Performance Highlights**: NotaGPT-7B 모델은 17개의 주요 멀티모달 대형 언어 모델과의 실험에서 음악 이해에 있어 두드러진 개선을 보였습니다. 기존 최고 성능 모델인 Gemini는 33.34%의 음악 정보 추출률을 달성한 반면, 우리 모델은 67.84%의 성과를 기록했습니다. 이는 멀티모달 음악 데이터셋의 중요성과 NOTA 데이터셋의 효과성을 보여줍니다.



### EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild (https://arxiv.org/abs/2502.14892)
Comments:
          NAACL 2025 Findings. Project page at this https URL

- **What's New**: EgoSpeak는 실제 환경에서 대화 에이전트가 언제 말을 시작해야 하는지를 예측하는 혁신적인 프레임워크로, 에고센트릭(egocentric) 비디오 스트리밍을 기반으로 설계되었습니다. 이 시스템은 사용자의 1인칭 시점을 모델링하여, 대화 에이전트가 주변을 관찰하고 말할 시점을 동적으로 결정할 수 있도록 합니다. 이를 통해 인간과 유사한 상호작용을 구현하고, 복잡한 자연 대화와 간단한 실험적 세팅 간의 간극을 줄입니다.

- **Technical Details**: EgoSpeak는 1인칭 시점(first-person perspective)에서 RGB 처리(RGB processing), 실시간 처리(online processing), 자르지 않은 비디오(untrimmed video) 처리를 포함한 네 가지 주요 기능을 통합합니다. 이러한 기능을 통해 에고센트릭 스트리밍 비디오에서 대화 에이전트가 직접 정보를 처리를 할 수 있도록 하여, 복잡한 다자 대화 환경에서도 자연스럽게 응답할 수 있게 하는 것이 중요한 목표입니다. 이 모델은 EasyCom과 Ego4D 데이터셋에서 실험되어 그 효과성이 입증되었습니다.

- **Performance Highlights**: EgoSpeak는 실시간으로 음성 시작 시점을 예측하며, 랜덤 및 정적 침묵 기반 기준보다 우수한 결과를 나타냅니다. 또한, 다중 모달 입력(multimodal input)과 상황 길이(context length)의 중요성을 강조하여 대화 에이전트가 언제 말을 할 지 결정하는 과정에서 효과적으로 작용함을 보여줍니다. YT-Conversation 데이터셋을 통해 대규모의 미선택 대화 비디오를 활용하여 다중 모달 턴 테이킹(turn-taking) 모델을 위한 훌륭한 교육 자원을 제공합니다.



### CoDiff: Conditional Diffusion Model for Collaborative 3D Object Detection (https://arxiv.org/abs/2502.14891)
- **What's New**: 이 논문에서는 자율 주행 분야에서 자주 발생하는 다중 에이전트 간 협업 3D 객체 감지 문제를 해결하기 위한 새로운 프레임워크, CoDiff를 제안합니다. CoDiff는 Conditional Diffusion Probabilistic Models (CDPM)를 활용하여 정보 교환 시 발생하는 공간적 및 시간적 노이즈 문제를 다룹니다. 이 연구는 확산 모델이 협업 인식에 적용된 첫 번째 사례로, 기존의 피처 융합 방법을 대체하여 에이전트 간의 특성을 효과적으로 정제합니다.

- **Technical Details**: CoDiff의 핵심 요소는 두 가지입니다. 첫 번째는 인식 압축(perception compression)으로, 이를 통해 각 에이전트의 피처를 잠재 공간(latent space)으로 압축하여 효율성을 극대화합니다. 두 번째는 조건적 생성(conditional generation) 모듈로, 이는 다양한 협력 에이전트의 피처를 입력으로 받아 노이즈를 점진적으로 정제하여 고해상도의 피처 생성을 수행합니다. 이로 인해, 다중 에이전트의 연결된 피처들은 통합적이고 일관된 표현을 가집니다.

- **Performance Highlights**: 실험 결과, CoDiff는 시뮬레이션 및 실제 데이터셋에서 다양한 시간 지연 및 자세 오류가 있을 때에도 기존의 관련 방법들보다 뛰어난 협업 3D 객체 감지 성능을 보였습니다. CoDiff는 특히 임의의 노이즈 혼합 분포를 학습하고, 효율적으로 특성을 융합하여 더욱 정확하고 견고한 결과를 도출합니다. 요약하자면, CoDiff는 다중 에이전트 협업 3D 객체 감지 성능을 획기적으로 향상시키는 방안을 제시하며, 자율 주행 및 로봇 시스템에서의 적용 가능성을 높입니다.



### WeedVision: Multi-Stage Growth and Classification of Weeds using DETR and RetinaNet for Precision Agricultur (https://arxiv.org/abs/2502.14890)
Comments:
          Accepted and Presented to ICMLA, 2024

- **What's New**: 본 연구는 미국 농업에서 경제적으로 중요한 16종 잡초를 정확하게 식별 및 분류하기 위해 최신 객체 탐지 모델인 Detection Transformer (DETR)와 RetinaNet을 활용합니다. 각 잡초의 생장 단계에서 11주 동안의 변화를 포함한 203,567장의 이미지를 사용하여 철저히 라벨링된 데이터셋을 구축하였습니다. 이를 통해 잡초 관리의 효율성과 정밀성을 개선할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: DETR과 RetinaNet은 최첨단 객체 탐지 작업에서 높은 성능을 보여주고 있으며, DETR은 복잡한 장면과 객체 관계를 처리하는 데 최적화된 변환기 기반 접근 방식을 제공합니다. RetinaNet은 클래스 불균형 문제를 해결하기 위한 초점 손실(focal loss) 기능을 사용하는 모델로, 연구에서는 이 두 모델을 비교하여 잡초 탐지 및 분류에 미치는 효과를 평가합니다.

- **Performance Highlights**: RetinaNet은 테스트 세트에서 0.904의 평균 평균 정확도(mean Average Precision, mAP)를 달성하며 DETR 모델보다 우수한 성능을 나타냈습니다. 이 모델은 또한 7.28 FPS의 추론 속도로 실시간 애플리케이션에 더 적합합니다. 두 모델 모두 식물의 성장이 진행됨에 따라 정확도가 향상된 것을 확인했습니다.



### Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability (https://arxiv.org/abs/2502.14889)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이 논문에서는 CLIP(Contrastive Language-Image Pretraining) 모델의 해석 가능성을 높이기 위한 새로운 틀인 Narrowing Information Bottleneck Theory(NIBT)를 제안합니다. 기존의 정보 병목 방법들은 고정된 가정이나 내재적인 무작위성의 영향을 받는 반면, NIBT는 현대의 기여 원칙(attribution axioms)을 만족시키고 해석 가능성을 더욱 향상시키도록 설계되었습니다. 실험 결과, NIBT는 이미지 해석 가능성을 평균 9%, 텍스트 해석 가능성을 평균 58.83% 향상시키고 처리 속도를 63.95% 가속화했습니다.

- **Technical Details**: CLIP 모델은 대규모 이미지-텍스트 쌍의 데이터로 훈련되어 이미지와 텍스트의 연관성을 학습합니다. 이 모델의 해석 가능성을 높이기 위해 NIBT에서는 무작위성 및 하이퍼파라미터 의존성을 제거하여 더 결정론적인 해석 결과를 제공합니다. 또한, 모델의 예측에 부정적인 영향을 미치는 특성 차원을 식별하는 새로운 개념인 negative property를 도입하여 해석 가능성을 더욱 강화했습니다.

- **Performance Highlights**: 본 연구의 접근 방식은 CLIP 모델의 해석 가능성을 기존 최첨단 방법들보다 현저히 개선했습니다. 이미지 해석 가능성이 평균 9% 향상되었으며, 텍스트 해석 가능성은 평균 58.83% 증가했습니다. 처리 속도 역시 63.95% 증가하여 효율성이 크게 높아졌습니다. 따라서 본 연구는 다양한 생성 AI 애플리케이션에서 CLIP 모델의 신뢰성과 투명성을 제고하는 데 중요한 기여를 하고 있습니다.



### The Multi-Faceted Monosemanticity in Multimodal Representations (https://arxiv.org/abs/2502.14888)
- **What's New**: 이 논문에서는 deep multimodal 모델에서 해석 가능한 feature를 추출하기 위해 최근의 feature monosemanticity(단일 의미성) 발전을 활용합니다. 특히, 대규모 이미지-텍스트 쌍으로 학습된 CLIP(Contrastive Language-Image Pretraining) 모델을 분석하여 해석 가능한 feature의 modality 연결성을 조사합니다. 우리는 Modality Dominance Score (MDS)를 도입하여 각 feature의 해석 가능한 특성을 해당 modality에 부여합니다.

- **Technical Details**: 우리는 두 가지 CLIP 모델, 즉 OpenAI의 전통적인 ViT-B-32 CLIP 모델과 이를 변형한 DeCLIP 모델을 사용합니다. DeCLIP은 이미지-텍스트 쌍 데이터 외에도 이미지-이미지 및 텍스트-텍스트 쌍을 활용한 self-supervision(자기 감독)을 포함하여 단일 의미의 feature를 더욱 효과적으로 추출합니다. Sparse Autoencoders(SAEs)와 Non-negative Constraint Loss(NCL)를 통해 multi-modal interpretability(다중 모달 해석 가능성) 강화를 위한 방법론을 개발합니다.

- **Performance Highlights**: 우리는 CLIP 모델의 feature가 인간의 인지 구조와 잘 일치한다는 사실을 발견했습니다. 이러한 해석 가능한 feature는 성별 편향 탐지, 적대적 공격 방어 및 텍스트-이미지 생성 등 다양한 실제 사용 사례에 유용하게 활용될 수 있음을 입증합니다. 이 결과는 다양한 modality 간의 주요 연결 및 차이를 이해하는 데 기여할 수 있는 대규모 multimodal 모델의 가능성을 보여줍니다.



### Vision-Enhanced Time Series Forecasting via Latent Diffusion Models (https://arxiv.org/abs/2502.14887)
- **What's New**: 이번 연구에서는 LDM4TS라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 시각 강화된 시간 시계열 예측을 위한 것으로, 외부 시각 데이터를 도입하는 대신 시간 시계열을 다중 시각 표현으로 변환하는 보완적 변환 기술을 최초로 사용합니다. 이를 통해 미리 훈련된 비전 인코더의 풍부한 특징 추출 능력을 활용할 수 있습니다.

- **Technical Details**: LDM4TS는 원시 시간 시계열 데이터를 구조적 인코딩을 통해 다중 시각 표현으로 변환합니다. 이 과정에서 세분화, 재발 그래프 및 Gramian Angular Fields와 같은 다양한 인코딩 방법을 사용하고, 변환된 표현은 낮은 차원의 잠재 공간으로 매핑됩니다. 이후, 사전 훈련된 잠재 확산 모델이 이 잠재 변수를 점진적으로 디노이즈합니다.

- **Performance Highlights**: 광범위한 실험 결과, LDM4TS는 다양한 데이터셋에서 최첨단 성능을 달성했습니다. 평균 제곱 오차(MSE) 기준으로 기존 확산 기반 방법들보다 최소 65.2% 향상되었으며, 준우승자에 대해서도 최소 15.7%의 개선을 보였습니다.



### Surgical Scene Understanding in the Era of Foundation AI Models: A Comprehensive Review (https://arxiv.org/abs/2502.14886)
- **What's New**: 최근 머신러닝(ML) 및 딥러닝(DL)의 발전 특히 기초 모델(FM)의 도입이 최소 침습 수술(MIS) 내에서 수술 장면 이해를 크게 향상시켰습니다. 이 논문은 CNN(합성곱 신경망), ViT(비전 트랜스포머), SAM(세그먼트 에니씽 모델)과 같은 최신 ML 및 DL 기술이 수술 프로세스에 통합된 사례를 조사합니다. 이러한 기술들은 수술 내시경 비디오 분석에서 분할 정확도, 도구 추적 및 단계 인식을 개선하는 데 기여했습니다.

- **Technical Details**: 이 논문은 최소 침습 수술(MIS)에서의 수술 장면 이해를 위한 최신 AI 모델의 활용도를 강조합니다. 특히, 영상 기술의 발전은 복잡한 수술을 보다 자신 있게 계획하고 수행할 수 있도록 돕습니다. 논문은 수술 도구 및 물체 감지, 수술 흐름 인식, 수술 훈련 및 시뮬레이션과 같은 주요 작업들을 자세히 설명하며, 각 작업의 중요성을 강조합니다.

- **Performance Highlights**: 본 논문에서는 기초 AI 모델이 수술 영상 데이터의 해석에 있어 혁신적인 변화를 가져오는 방법을 제시합니다. 알고리즘은 수술 중 포착된 비디오를 분석하여 실시간으로 실행 가능한 인사이트를 제공하고, 이전 연구들과 비교해 종합적이며 통합된 접근 방식을 채택하고 있습니다. 이는 궁극적으로 환자의 결과를 개선하고 수술 안전성을 높이는 데 기여할 것입니다.



### SEM-CLIP: Precise Few-Shot Learning for Nanoscale Defect Detection in Scanning Electron Microscope Imag (https://arxiv.org/abs/2502.14884)
Comments:
          Published in ACM/IEEE International Conference on Computer-Aided Design (ICCAD), 2024

- **What's New**: 본 논문에서는 SEM 이미지 결함 분류 및 세분화를 위한 새로운 few-shot learning 접근 방식인 SEM-CLIP을 제안합니다. 이 방법은 기존의 데이터 및 라벨 요구 사항을 최소화하여 반도체 산업의 라벨링 문제를 해결하고 스마트 제조를 촉진할 수 있는 기반을 마련합니다. 특히 SEM-CLIP은 CLIP 모델을 사용자 정의하여 결함 영역에 집중하고 배경의 복잡성을 최소화한 기능 추출 방법을 도입했습니다.

- **Technical Details**: SEM-CLIP 모델은 Contrastive Language-Image Pretraining(CLIP) 모델을 조정하여 결함 영역에 더욱 집중하고 세분화 정확도를 향상시키기 위해 V𝑉Vitalic_V-V𝑉Vitalic_V 어텐션 블록을 추가하여 복잡한 배경 방해 요소를 최소화합니다. 또한, 전문 지식을 활용한 텍스트 프롬프트를 사전 정보로 활용하여 명확한 결함 분석을 지원합니다. 이를 통해 충분한 주석 데이터 없이도 모델이 복잡한 시각적 개념을 이해하도록 돕습니다.

- **Performance Highlights**: SEM-CLIP은 여러 few-shot 설정에서 테스트된 결과, 다른 방법들보다 우수한 클래시피케이션(iAUROC, pAUROC, F1-mmax) 성능을 보여줍니다. 특히, SEM-CLIP은 최근의 SOTA 방법인 PromptAD보다 각각 2.0%, 1.3%, 21.1% 향상된 결과를 기록했습니다. 이러한 결과는 반도체 산업에서 결함 감지의 효율과 정확성을 크게 높일 것으로 기대됩니다.



### Can LVLMs and Automatic Metrics Capture Underlying Preferences of Blind and Low-Vision Individuals for Navigational Aid? (https://arxiv.org/abs/2502.14883)
Comments:
          26 pages, 12 figures, 14 tables

- **What's New**: 이 논문에서는 시각 장애인(Blind and Low-Vision, BLV) 사용자들이 내비게이션을 위해 Large Vision-Language Models (LVLMs)에서 어떤 유형의 응답을 선호하는지를 조사합니다. 이를 위해, Eye4B 데이터셋을 구축하여 1,100개의 야외 및 실내 장면에 대해 사람 검증을 거친 다양한 요청을 포함하였습니다. 또한, BLV 사용자의 편의를 위해 LVLM의 응답 스타일 및 그들의 선호도를 평가하기 위한 심층 사용자 연구를 수행하였습니다.

- **Technical Details**: BLV 사용자의 요구 사항을 충족하기 위해 5가지 관점(Afraidness, Nonactionability, Sufficiency, Conciseness, Overall)에서 6개의 LVLM 응답을 분석합니다. Eye4B 데이터셋은 고품질의 시각 장면을 제공하며, 다양한 BLV 요구를 기반으로 한 요청을 포함합니다. 우리는 관객의 반응을 이해하고 BLV-Aware AI 시스템을 개발하는 데 필요한 귀중한 데이터를 수집하기 위해 모델 기반의 이미지-텍스트 메트릭스와 BLV 선호 사이의 일치를 평가하는 Eye4B 벤치마크도 소개합니다.

- **Performance Highlights**: BLV 사용자의 선호도에 대한 분석 결과, 특정 LVLM은 사용자들에게 더 긍정적인 반응을 이끌어낸 것으로 나타났으며, 이는 실제 응용 프로그램 개발에 중요한 지침이 될 수 있습니다. 특히, Eye4B 벤치마크는 모델 기반 메트릭이 BLV의 선호를 얼마나 잘 반영하는지를 평가하기 위해 새로운 평가 차원을 제공하며, 이것은 앞으로의 연구에 중요한 발판이 될 것입니다.



### From 16-Bit to 1-Bit: Visual KV Cache Quantization for Memory-Efficient Multimodal Large Language Models (https://arxiv.org/abs/2502.14882)
- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 KV 캐시 메모리 사용량을 획기적으로 줄이기 위한 새로운 접근법을 제안합니다. 기존 방법들이 중요하지 않은 토큰들을 삭제하여 메모리를 절약하는 반면, 제안된 방법은 모든 시각적 토큰을 보존하면서 메모리 소비를 크게 줄이는 1비트 양자화(quantization) 전략을 적용합니다. 이러한 방식은 정보의 손실 없이 MLLM의 메모리 효율성을 개선할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 방법은 그룹별 양자화 및 분위수 기반 양자화 기법을 통해 KV 캐시의 메모리 요구 사항을 줄이는 데 중점을 두고 있습니다. 이 모듈은 다양한 MLLM에 쉽게 통합될 수 있는 플러그 앤 플레이 방식으로 설계되어, 아키텍처 수정 없이도 높은 메모리 효율성을 달성할 수 있습니다. 실험을 통해 알고리즘이 메모리 사용량을 줄이면서 계산 효율성과 다중 모드 성능을 유지하는 데 효과적임을 입증했습니다.

- **Performance Highlights**: 실험 결과, 토큰을 무작위로 삭제하는 기존의 방법보다 제안된 메모리 최적화 기법이 성능 저하 없이 메모리 사용량을 효과적으로 줄일 수 있음을 보여주었습니다. 특히, 시각적 정보가 많은 긴 시퀀스 생성 작업에서 효율성이 크게 향상되었습니다. 이 연구는 향후 MLLM의 실제 적용 가능성을 높이는 데 중요한 기여를 할 것으로 기대됩니다.



### KKA: Improving Vision Anomaly Detection through Anomaly-related Knowledge from Large Language Models (https://arxiv.org/abs/2502.14880)
- **What's New**: 이번 연구에서는 Key Knowledge Augmentation (KKA)라는 새로운 방법을 제안합니다. KKA는 대형 언어 모델(LLMs)에서 이상치와 관련된 지식을 추출하여, 정상 샘플을 기반으로 하여 의미 있는 이상치를 생성합니다. 이 방식은 랜덤하게 생성된 이상치가 아닌, 실질적인 경계 설정을 위한 효과적인 이미지를 생성하여 검출기의 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: KKA는 정상 샘플에서 무작위 샘플링을 통해 LLM들에게 프롬프트를 제공하여 이상치 관련 지식을 생성합니다. 이 과정에서 생성된 이상치는 '쉬운 이상치'와 '어려운 이상치'로 분류되며, 후자는 정상 샘플과 유사한 형태로 검출기의 경계 학습에 더 큰 영향을 미칩니다. KKA는 검출기의 성능을 반복적으로 업데이트하며, 특히 어려운 이상치의 비율을 점진적으로 증가시킵니다.

- **Performance Highlights**: 실험 결과, KKA는 다양한 시각적 이상치 검출기에서 성능을 크게 개선했습니다. 예를 들어, CIFAR-100 데이터셋에서 KKA는 SimpleNet의 AUC를 74.62%에서 84.04%로 크게 향상시켰으며, 생성된 샘플의 수는 SimpleNet의 약 5%에 불과했습니다. 이는 KKA가 경계 설정에 있어 큰 효과를 가져왔음을 보여줍니다.



### One-step Diffusion Models with $f$-Divergence Distribution Matching (https://arxiv.org/abs/2502.15681)
- **What's New**: 이번 논문에서는 일반적인 분포 매칭(distillation) 접근법을 확장하는 새로운 $f$-divergence 최소화 프레임워크인 $f$-distill을 제안합니다. 이 접근법은 다양한 divergence를 포괄하여 모드 커버리지(mode coverage)와 훈련 분산(training variance) 사이의 서로 다른 균형을 제공합니다. 기존의 reverse-KL divergence 기반 방법은 특정 모드에 초점을 맞출 수 있지만, $f$-distill은 이보다 더 다양한 diverge 모델을 통해 더 나은 샘플링을 가능하게 합니다.

- **Technical Details**: $f$-distill은 교사(teacher)와 학생(student) 분포 간의 다양한 $f$-divergence를 통해 서로 다른 데이터를 고려합니다. 우리가 제안하는 방법은 오히려 적은 모드를 추구하고, 훈련 과정에서 발생하는 분산을 줄이는 형태로 가능합니다. 또한, 각 divergence의 가중 함수(weighting function)에 따라 학생의 투표(score)가 교사의 높은 밀도가 있는 샘플에 더 많은 비중을 두게 하는 방식을 사용합니다.

- **Performance Highlights**: 경험적으로, 제안된 $f$-distill 방법은 이미지 생성 작업에서 이전의 최선의 variational score distillation 방법보다 더 높은 성능을 보입니다. 특히, Jensen-Shannon divergence를 사용했을 때는 ImageNet64에서의 1단계 생성 성능과 MS-COCO에서 제로샷 텍스트-이미지 생성에서 최신의 성과를 달성했습니다. 이 연구는 학생 분포가 교사 분포에 매칭하는 다양한 방식과 결과를 통한 실질적인 가이드를 제공합니다.



### BOSS: Benchmark for Observation Space Shift in Long-Horizon Task (https://arxiv.org/abs/2502.15679)
- **What's New**: 이 논문은 로봇 비전 분야에서 Observation Space Shift (OSS)라는 새로운 문제를 제기하며, 이를 평가할 수 있는 Benchmark인 BOSS를 소개합니다. OSS는 앞선 스킬이 수행되는 과정에서 관찰 공간이 변화하여 후속 스킬의 성능에 부정적인 영향을 미치는 현상입니다. 특히, 이 연구는 로봇이 이전에 학습한 스킬들을 조합하여 장기 작업을 수행하는 과정에서 발생하는 문제를 다룹니다.

- **Technical Details**: BOSS는 세 가지 도전 과제로 구성되어 있으며, 각각은 OSS가 로봇의 성능에 미치는 영향을 평가합니다. 해당 도전 과제는 'Single Predicate Shift', 'Accumulated Predicate Shift', 및 'Skill Chaining'입니다. 연구에서는 Behavioral Cloning와 OpenVLA 등 여러 최근의 모방 학습 (Imitation Learning) 알고리즘을 평가하여 OSS 현상이 나타나는 경우와 그렇지 않은 경우의 성능 차이를 분석하였습니다. 

- **Performance Highlights**: 평가한 알고리즘들은 OSS가 존재할 때 평균적으로 67%, 35%, 34%, 54%의 성능 저하를 보였습니다. 연구진은 데이터 증강을 통해 OSS 문제를 해결할 수 있는지를 조사했으나, 데이터 증강만으로는 OSS를 완전히 해결할 수 없음을 보여주었습니다. 이 연구는 다양한 장기 로봇 작업에 대한 OSS의 깊은 이해와 미래 연구를 위한 알고리즘적 해결책의 필요성을 강조합니다.



### Logit Disagreement: OoD Detection with Bayesian Neural Networks (https://arxiv.org/abs/2502.15648)
Comments:
          Presented at ECCV 2024 Workshop: 3rd Workshop on Uncertainty Quantification for Computer Vision

- **What's New**: 본 연구는 Bayesian Neural Networks (BNNs)에서 발생하는 epistemic (지식적 불확실성) 불확실성을 보다 정확하게 추정하기 위한 새로운 접근 방식을 제안합니다. 제안된 방법은 pre-softmax 값, 즉 logits의 수정된 버전 간의 불일치를 측정하여 epistemic 불확실성을 평가하는 것입니다. 이 논문은 다양한 OoD(out-of-distribution) 실험에서 mutual information (상호 정보량)보다 현저한 개선 효과를 보였습니다.

- **Technical Details**: Bayesian Neural Networks에서는 네트워크 가중치를 확률 변수로 보고, Variational Inference(변분 추론)를 통해 posterior distribution(사후분포)을 추정합니다. 이 연구에서는 posterior 샘플 간의 maximum logit 값의 불일치를 통해 epistemic 불확실성을 평가하는 쉽고 모델에 구애받지 않는 epistemic 불확실성 점수를 제안합니다. 또한 Bayesian Variational Autoencoder (BVAE) 문헌에서 이 점수를 기초로 한 분류적 접근을 탐색합니다.

- **Performance Highlights**: 제안된 epistemic 불확실성 점수는 MNIST 및 CIFAR10 실험에서 Bayesian benchmark인 predictive entropy와 유사한 성능을 보여줍니다. 특히, 다양한 OoD 실험에서도 mutual information보다 우수한 성능을 기록하여 모델의 불확실성 추정의 중요성을 강조합니다. 이 논문은 BNN을 사용하는 머신러닝 모델의 신뢰성과 안전성을 높이는 데 기여합니다.



### Aligning Task- and Reconstruction-Oriented Communications for Edge Intelligenc (https://arxiv.org/abs/2502.15472)
Comments:
          Accepted for publication in IEEE Journal on Selected Areas in Communications (JSAC)

- **What's New**: 이 논문은 기존의 통신 시스템이 정보 복원에 중점을 둔 구조에서 벗어나, 작업 지향 통신(task-oriented communications)과 복원 지향 통신(reconstruction-oriented communications)을 통합한 새로운 커뮤니케이션 프레임워크를 제안합니다. 이 프레임워크는 AI 기반 응용 프로그램, 특히 자율 주행 및 의미론적 분할에 적합하도록 설계되었습니다. 정보 병목 이론(Information Bottleneck theory)을 확장하여 데이터 전송을 최적화하며, 태스크 관련 손실 함수를 최소화하는 접근 방식을 채택합니다.

- **Technical Details**: 제안된 시스템은 변별적 접근 방식으로서 상호 정보(mutual information) 계산의 복잡함을 해결하며, 원래 데이터의 구조를 유지합니다. 이를 통해 작업 지향 통신과 복원 지향 통신을 통합하는 것이 가능합니다. 또한, 클래스 통신 모듈과 호환되는 공동 소스-채널 코딩(Joint Source-Channel Coding, JSCC) 방식이 도입되어 기존 디지털 인프라 내에서 AI 기술 배포를 가능하게 합니다.

- **Performance Highlights**: CARLA 시뮬레이터에서의 평가 결과, 제안된 프레임워크는 기존의 JPEG, JPEG2000, BPG 등과 비교할 때 서비스 당 비트 수를 99.19% 감소시켰습니다. 이는 작업 수행의 효과성을 저하시키지 않으면서 데이터 전송 효율성을 현저하게 향상시킵니다. 이러한 성과는 엣지 인공지능 기술을 통해 자율 주행 시나리오에서 특히 두드러지게 나타납니다.



### Anatomy-Informed Deep Learning and Radiomics for Automated Neurofibroma Segmentation in Whole-Body MRI (https://arxiv.org/abs/2502.15424)
- **What's New**: 네urofibromatosis Type 1 (NF1)는 신경섬유종(neurofibromas, NFs)의 발생을 특징으로 하는 유전 질환으로, 이 연구에서는 전체 신체 자기공명영상(Whole-Body MRI)에서의 NF 자동 세분화(segmentation) 방법이 제안되었습니다. 이 방법은 해부학적(segmentation) 정보에 기반하여 3단계로 이루어지며, NF의 정확한 식별을 통해 간섭이 줄어들고 임상 사용을 촉진할 수 있습니다.

- **Technical Details**: 제안된 방법은 MRSegmentator 모델을 활용하여 해부학적(segmentation) 마스크를 생성하고, 3D 비등방성(anisotropic) U-Net의 앙상블을 통해 NF 세분화 신뢰도 마스크를 생성합니다. 마지막 단계에서는 방사형(radiomic) 특성을 기반으로 종양 후보를 분류하여 거짓 양성(false positives)을 줄이는 알고리즘이 적용됩니다.

- **Performance Highlights**: 제안된 파이프라인은 다양한 테스트 세트를 통해 평가되었으며, 통합한 anatomy 정보 덕분에 스캔당 Dice Similarity Coefficient(DSC)가 68% 증가하였고, 종양 탐지에 대한 F1 점수에서는 두 배 향상이 있었습니다. 이 방법은 3D Slicer 플랫폼에 통합되어 실제 임상 환경에서 사용이 용이하며, 소스코드와 트레이닝된 모델이 공개되었습니다.



### Evaluating Multimodal Generative AI with Korean Educational Standards (https://arxiv.org/abs/2502.15422)
Comments:
          18 pages; To appear at NAACL 2025 Main Conference (Project page: this https URL )

- **What's New**: 이번 논문은 한국 교육 시험을 활용하여 멀티모달 생성 AI 시스템을 평가하기 위한 새로운 벤치마크인 KoNET(Korean National Educational Test Benchmark)을 소개합니다. KoNET는 초중고 및 대학 입학 자격 시험인 KoEGED, KoMGED, KoHGED, KoCSAT의 네 가지 시험으로 구성되어 있으며, 이들 시험은 낮은 자원 언어인 한국어에 대한 AI의 성능을 분석할 수 있는 기회를 제공합니다. KoNET를 통해 다양한 모델의 성능을 비교하고, 특히 인간 오답률 데이터와 함께 AI의 능력을 면밀히 평가할 수 있습니다.

- **Technical Details**: KoNET는 한국의 국가 교육 시험에서 문제를 변환하여 구조화된 멀티모달 VQA 형식으로 구성되어 있습니다. 각 시험은 질문 난이도에 대한 세부 분석을 제공하며, KoCSAT는 응시자의 오답률 데이터를 포함하여 AI 모델의 행동을 인간 성능과 직접 비교할 수 있도록 합니다. 더불어, 다양한 오픈소스 및 클로즈드 소스 AI 모델이 KoNET에서 테스트되며, Chain-of-Thought (CoT) 접근법과 OCR API를 활용하여 이미지 기반 문제를 처리합니다.

- **Performance Highlights**: 실험에서는 18개의 오픈소스 LLMs, 20개의 오픈소스 MLLMs, 4개의 클로즈드 소스 LLMs 및 4개의 클로즈드 소스 MLLMs를 포함하여 다양한 모델을 평가했습니다. KoNET는 정확한 기준을 마련하고, AI 성능을 이해하기 위한 통찰력을 제공함으로써 한국 교육 시장에서 AI 기반 교육 기술의 응용 가능성을 높입니다. 또한, 데이터 세트와 코드는 모두 공개되어 있어서 연구자들이 자유롭게 접근할 수 있도록 합니다.



### Chitrarth: Bridging Vision and Language for a Billion Peop (https://arxiv.org/abs/2502.15392)
- **What's New**: 최근 다중 모달 기초 모델의 주된 훈련은 영어 및 자원 풍부한 유럽 언어 데이터에 한정되어 있어, 그 결과 중소 및 저자원 언어에서의 적용 가능성이 제한되었습니다. 이를 해결하기 위해, 우리는 10가지 주요 인도 언어의 언어적 다양성과 시각적 추론을 목표로 하는 포괄적인 비전-언어 모델(Chitrarth)을 소개합니다. 이 모델은 다언어 대형 언어 모델(LLM)과 비전 모듈을 효과적으로 통합하여 자원 부족 언어에 대한 새로운 기준을 설정하고, 보다 다양한 AI 시스템 개발에 기여할 것을 목표로 하고 있습니다.

- **Technical Details**: Chitrarth 모델은 전처리된 비전 인코더와 Krutrim LLM 백본을 사용하여 이미지와 언어 모달리티를 활용하며, 비전 토큰과 텍스트 토큰을 결합하여 LLM에 입력됩니다. 우리는 다중 모달 학습을 위해 이미지 인코딩을 시작하며, 그 결과로 생성된 비전 임베딩을 LLM 임베딩 공간으로 매핑하는 프로젝션 레이어를 활용합니다. 이 과정은 다양한 외부 데이터셋을 통해 훈련된 멀티링구얼 데이터셋을 번역하여 인도 언어 간의 교차 언어 일반화 능력을 향상시키는 것을 포함합니다.

- **Performance Highlights**: Chitrarth 모델은 영어 데이터셋에서 SOTA 결과를 달성했으며, 제안된 다중 언어 데이터셋에서도 뛰어난 성능을 보였습니다. 우리는 교육 전략 및 변칙 실험을 통해 5개 영어 데이터셋 중 3개에서 SOTA 결과를 기록하였고, 10개의 인도 언어에 대한 종합 평가 벤치마크인 BharatBench를 제안하여, 저자원 언어에 대한 진전을 촉진하고자 합니다. 이는 다중 모달 능력 강화와 AI 기술 발전에 크게 기여할 것으로 기대됩니다.



### M2LADS Demo: A System for Generating Multimodal Learning Analytics Dashboards (https://arxiv.org/abs/2502.15363)
Comments:
          Published in the Workshop on Innovation and Responsibility in AI-Supported Education (iRAISE25) at AAAI 2025

- **What's New**: M2LADS는 컴퓨터 기반 학습 세션 중 수집된 다중 모드 데이터의 통합, 동기화, 시각화 및 분석을 위한 웹 기반 시스템입니다. 이 시스템은 생체 데이터와 행동 데이터를 웹 기반 대시보드에서 자세히 시각화하여 사용자의 생리학적 및 활동 기반 메트릭에 대한 통찰을 제공합니다. M2LADS는 참가자의 경험을 종합적으로 보여주고, 모든 신호와 비디오를 동기화하여 데이터 레이블링을 간소화하는 데 중점을 두고 있습니다.

- **Technical Details**: M2LADS 시스템은 세 가지 모듈로 구성되어 있으며, 생체 신호, 심박수, 시선 추적 데이터 등의 다양한 다중 모드 데이터 소스를 사용합니다. Activity Data Processing Module은 다중 모드 데이터를 추출하고 정리하여 신호를 동기화하며, 특정 학습 활동에 따라 값들을 분류합니다. Activity Data Management Module은 MongoDB에 처리된 생체 데이터를 저장하고, 개인 식별 정보를 완전히 익명화하여 데이터를 보호합니다.

- **Performance Highlights**: M2LADS는 학습 세션 중의 생체 신호 유효성을 검증하고 학습자 성과 메트릭을 비교하며, 활동에 따라 신호를 분석하는 데 유용합니다. 시스템은 학습자의 참여도와 반응을 종합적으로 평가할 수 있게 해주며, 특정 활동의 효과에 대한 깊은 이해를 제공합니다. 동시에, 사용자 맞춤형 대시보드를 통해 시각화된 데이터를 제공하여 연구자와 강사가 쉽게 학습자의 행동을 이해할 수 있도록 돕습니다.



### Quantum autoencoders for image classification (https://arxiv.org/abs/2502.15254)
- **What's New**: 이번 연구는 Quantum Autoencoders (QAEs)를 이미지 분류 작업에 적용하는 혁신적인 접근 방식을 제안합니다. 기존의 QAE 사용 사례가 데이터 압축 및 재구성에 국한되어 있었던 반면, 본 연구에서는 QAEs가 이러한 작업을 넘어 더 다양한 태스크를 수행할 수 있음을 보여줍니다. 이는 더 효과적인 기능 추출을 가능하게 하며, 기존 방식에 비해 적은 양의 qubit로 분류 작업을 수행할 수 있는 가능성을 열어줍니다.

- **Technical Details**: QAEs는 입력 데이터를 더 적은 양의 qubit로 압축하고, 학습된 양자 회로를 통해 원래 정보를 재구성하는 생성 모델입니다. 본 연구에서는 학습 중 이미지와 레이블 정보를 모두 양자 회로에 인코딩하고, 재구성 오류를 최소화하는 방향으로 파라미터를 최적화하여 미지의 테스트 이미지에 대한 클래스 레이블을 예측할 수 있는 방법을 사용합니다. 이는 QAE 기반 분류에서 양자 계산이 중요한 역할을 한다는 점에서 QCNN과 차별화됩니다.

- **Performance Highlights**: 본 연구의 실험 결과는 QAEs를 사용한 이미지 분류에서 높은 정확도를 달성하였으며, 다양한 양자 게이트 구성에 대한 영향을 분석했습니다. 특정 답안 구조가 우수한 정확도를 나타내는 것으로 나타났으며, QAEs는 전통적인 기계 학습 방법들과 비교하여 파라미터 최적화를 상당히 줄이면서 유사한 성능을 보여줍니다. 결과적으로 QAEs는 적은 수의 파라미터로도 효과적인 분류 모델로 작용할 수 있으며, 완전한 엔드 투 엔드 학습을 위한 양자 회로 활용의 가능성을 보여줍니다.



### Lung-DDPM: Semantic Layout-guided Diffusion Models for Thoracic CT Image Synthesis (https://arxiv.org/abs/2502.15204)
Comments:
          The code and pretrained models are available at this https URL

- **What's New**: 이번 연구에서는 Lung-DDPM을 제안하여 폐암 조기 스크리닝을 위한 고품질 3D 합성 CT 이미지를 생성하는 방법을 소개합니다. 기존의 고비용 데이터 주석 프로세스와 프라이버시 문제를 해결하기 위해, 우리의 접근법은 불완전한 해부학적 레이아웃에서도 일관된 이미지를 생성할 수 있는 능력을 갖춘 세분화된 디노이징 확산 확률 모델(DDPM)을 기반으로 합니다. 이를 통해 폐 결절 세분화 작업에 실질적인 도움을 줍니다.

- **Technical Details**: Lung-DDPM은 해부학적으로 인식 가능한 샘플링 프로세스를 설계하여 참조 CT 볼륨과 해당 폐 해부학적 레이아웃을 조합하여 디노이징 프로세스를 안내합니다. 이 방법은 합성된 폐 이미지를 실제로 존재하는 외부 폐 구조와 동적으로 혼합하여 고품질의 폐 CT 이미지를 생성합니다. 제안된 방법은 조정된 3D U-Net과 결합되어 원활한 합성 샘플을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: Lung-DDPM은 이미지 품질 평가 및 다운스트림 폐 결절 세분화 작업에서 최신 기술의 타 모델보다 우수한 성능을 입증하였습니다. 대규모 검증 집단에서 FID, MMD 및 MSE 값이 각각 0.0047, 0.0070, 0.0024로, 경쟁 모델보다 크게 개선되었습니다. 이 연구는 AI 지원 폐암 스크리닝을 위한 데이터 부족 문제 해결을 위한 중요한 기초 자료가 될 것입니다.



### Interleaved Block-based Learned Image Compression with Feature Enhancement and Quantization Error Compensation (https://arxiv.org/abs/2502.15188)
- **What's New**: 최근 학습 기반 이미지 압축(learned image compression, LIC) 기술 분야에서 중요한 성과 개선이 이뤄졌고, 본 논문에서는 이미지의 입력 픽셀을 섞고 서브 이미지를 분할하여 조잡한 특징을 추출하는 기능 추출 모듈을 제안한다. 또한, 3D 컨볼루션을 활용하여 채널, 서브 이미지 내부 및 서브 이미지 간의 상관관계를 학습하고 더 압축된 잠재 특징(latent features)을 얻는 특징 정제 모듈과, 양자화 이후 특징의 품질을 향상시키기 위한 특징 향상 모듈을 제시한다. 마지막으로 양자화 오류 보상 모듈을 통해 학습과 테스트 간의 양자화 불일치를 완화하는 방법도 설명한다.

- **Technical Details**: 특징 추출 모듈은 이미지의 픽셀을 섞고 서브 이미지로 나눈 후, 조잡한 특징을 추출한다. 특징 정제 모듈은 세 개의 3D 컨볼루션 잔여 블록으로 구성된 주의 정제 블록(attention refinement block)을 쌓아 채널 간 및 서브 이미지 간의 상관관계를 활용하여 더 압축된 잠재 특징을 학습한다. 그리고 특징 향상 모듈은 양자화 후 디코딩된 특징에서 정보 손실을 줄인다. 이 모든 모듈은 최신 LIC 방법에 쉽게 통합될 수 있으며, 훈련 가능한 구조로 설계되었다.

- **Performance Highlights**: 제안된 모듈을 Tiny-LIC와 결합한 실험 결과, 기존의 LIC 방법 및 이미지 압축 표준보다 더 뛰어난 성능을 보였다. 특히 피크 신호 대 잡음비(PSNR)와 다중 스케일 구조적 유사도(MS-SSIM) 지표에서 Kodak 데이터셋과 CLIC 데이터셋을 기반으로 한 평가에서 앞선 결과를 보여주었다. 이는 제안된 방식이 전통적인 이미지 압축 방법과 최신 학습 기반 방법에 비해 더 우수한 성능을 발휘함을 시사한다.



### LUMINA-Net: Low-light Upgrade through Multi-stage Illumination and Noise Adaptation Network for Image Enhancemen (https://arxiv.org/abs/2502.15186)
Comments:
          9 pages, 4 figures

- **What's New**: 본 논문에서는 저조도 이미지 향상(Low-Light Image Enhancement) 기술의 새로운 프레임워크인 LUMINA-Net을 제안합니다. LUMINA-Net은 다단계 조명 및 반사 모듈을 통합하여 저조도 이미지의 품질 문제를 해결하고, 기존 기술들이 겪는 한계를 극복합니다. 이는 텍스처 디테일을 보존하면서 조명과 대비를 정교하게 조정하고, 노이즈 제거 메커니즘을 통해 이미지의 자연스러운 색상과 균형을 유지합니다.

- **Technical Details**: LUMINA-Net은 Retinex 이론을 기반으로 하여, 두 개의 저조도 이미지 쌍을 활용하여 조명과 반사 성분을 분리하는 기능을 가지고 있습니다. 주요 혁신으로는 채널 안내(Channel-Guidance) 모듈이 포함되어 있어, 공간 및 채널 주의 메커니즘을 통해 특성 추출을 개선합니다. 또한, 색상 향상(Color Enhancement) 모듈이 자연스러운 색 복원을 보장하며, 과잉 노출(Over-Exposure Correction) 모듈이 과도하게 밝은 지역을 동적으로 조정하여 아티팩트를 최소화합니다.

- **Performance Highlights**: 다양한 LOL 및 SICE 데이터셋에서 수행한 실험을 통해 LUMINA-Net의 뛰어난 성능이 입증되었습니다. PSNR, SSIM 및 LPIPS 메트릭스에서 기존의 최첨단 방법들을 초월하며, 노이즈를 효과적으로 억제하고 세부 사항을 보존합니다. 이러한 성능 향상은 저조도 환경에서의 비전 기반 시스템의 잠재력을 극대화하는 데 기여합니다.



### FD-LSCIC: Frequency Decomposition-based Learned Screen Content Image Compression (https://arxiv.org/abs/2502.15174)
- **What's New**: 이 논문은 화면 콘텐츠(SC) 이미지 압축을 위한 새로운 방법을 제안합니다. 기존의 Learned Image Compression (LIC) 기법이 자연 장면(NS) 이미지에는 효과적이지만, SC 이미지의 특성에 적합하지 않다는 문제를 해결하고자 합니다. 이를 위해 다중 주파수 변환 및 적응형 양자화 기법을 활용하여, SC 이미지의 독특한 특성에 맞춰 압축 성능을 개선합니다.

- **Technical Details**: 제안한 방법은 다중 주파수 두 단계 옥타브 잔여 블록(MToRB)을 통해 특성을 추출하며, 다양한 주파수 요소에 따라 양자화 단계 크기를 동적으로 조절하는 적응형 양자화 모듈을 포함합니다. 또한, 큰 SC 이미지 압축 데이터셋(SDU-SCICD10K)을 구축하여 약 10,000개의 이미지를 포함하고 있습니다. 이는 웹 및 사무 이미지, 컴퓨터 렌더링 이미지, 그리고 NS 및 SC 이미지의 혼합으로 구성되어, 다양성을 제공합니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 전통적인 이미지 압축 표준과 최신의 H.266/VVC 및 상태 기반 최신 방법들보다 우수한 성능을 보임을 증명했습니다. 특히 PSNR과 MS-SSIM에서 압축 비트 전송률(bit rate)과 같은 조건 하에 성능을 크게 향상시켰습니다. 이는 SC 이미지의 압축에서의 압도적인 개선을 나타냅니다.



### Optimized Pap Smear Image Enhancement: Hybrid PMD Filter-CLAHE Using Spider Monkey Optimization (https://arxiv.org/abs/2502.15156)
- **What's New**: 이번 연구는 자궁경부암 탐지를 위해 Pap smear 이미지 품질 향상을 위한 최적화된 하이브리드 접근법을 제안합니다. Perona-Malik Diffusion (PMD) 필터와 대비 한정 적응 히스토그램 균등화 (CLAHE)를 결합하여 이미지 품질을 향상시킵니다. 이 방법은 거미 원숭이 최적화 (SMO PMD-CLAHE)를 통해 최적화되었습니다.

- **Technical Details**: PMD 필터는 이미지 노이즈를 줄여주고, CLAHE는 이미지 대비를 향상시킵니다. PMD 필터와 CLAHE 최적화를 위한 새로운 객관적 함수로는 BRISQUE와 CEIQ가 사용되었습니다. 실험은 SIPaKMeD 데이터셋을 활용하여 이루어졌습니다.

- **Performance Highlights**: SMO 방법은 PMD 필터와 CLAHE 최적화에서 첨단 기술을 뛰어넘는 성능을 보였습니다. 제안된 방법은 평균 효과 측정치 (EME) 5.45, 제곱근 평균 제곱 (RMS) 대비 60.45, Michelson의 대비 (MC) 0.995 및 엔트로피 6.80을 기록했습니다. 이 접근법은 Pap smear 이미지 품질 향상을 위한 새로운 시각을 제공합니다.



### CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models (https://arxiv.org/abs/2502.15119)
- **What's New**: 이번 연구에서는 CurricuVLM이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Vision-Language Models (VLMs)를 활용하여 자율주행 에이전트를 위한 개인화된 커리큘럼 학습을 가능하게 합니다. 또한, VLM의 다중 모달 이해 능력을 측정하여 에이전트의 행동을 분석하고, 그에 따라 맞춤형 훈련 시나리오를 동적으로 생성합니다. 이 접근법은 기존의 자율주행 시스템에 안전성을 높이는 혁신적인 방법으로 주목받고 있습니다.

- **Technical Details**: CurricuVLM 프레임워크는 에이전트의 다양한 운전 상황에서의 행동을 지속적으로 모니터링하여, VLMs를 사용해 운전 상황을 이해하고 분석합니다. 에이전트의 현재 능력과 한계를 평가하기 위해 전문적인 GPT-4o 기반 분석기를 활용하여, 종합적인 분석을 수행합니다. 이 분석을 바탕으로 에이전트의 발전하는 능력에 맞춘 개인화된 훈련 시나리오가 동적으로 생성됩니다. 이는 에이전트가 지속적으로 성장할 수 있도록 지원합니다.

- **Performance Highlights**: Waymo Open Motion Dataset에서 실시된 광범위한 실험을 통해, CurricuVLM은 일반적인 상황 및 안전기반 상황 모두에서 최신 방법론들을 능가하는 성능을 보여주었습니다. 내비게이션 성공률, 운전 효율성 및 안전 지표에서 우수한 결과를 나타냄으로써, 다양한 강화학습 (RL) 알고리즘과의 호환성 역시 인정받았습니다. 이 연구는 자율주행 시스템을 개선하기 위한 일반적인 프레임워크로서의 잠재력을 보이고 있습니다.



### Assessing a Single Student's Concentration on Learning Platforms: A Machine Learning-Enhanced EEG-Based Framework (https://arxiv.org/abs/2502.15107)
- **What's New**: 이 연구는 온라인 학습 세션 중 학생의 집중 상태를 분류하기 위해 맞춤 설계된 기계 학습 모델을 사용하는 특화된 파이프라인을 소개합니다. EEG 데이터의 수집 및 전처리 프로토콜과 함께 5개의 EEG 신호 밴드에서 50개의 통계적 특성을 추출하는 방법이 상세히 설명되어 있습니다. 하이퍼파라미터 튜닝을 통해 학생의 집중 상태 분류 정확도를 향상시키는 이점을 탐구하며, VR 환경에서도 높은 정확성을 달성했습니다.

- **Technical Details**: 이 연구에서는 Muse S(Gen 2)라는 경량의 상용 EEG 센서를 사용하여 EEG 신호를 수집하였습니다. 신호는 Delta, Theta, Alpha, Beta, Gamma의 5개 주파수 밴드로 나뉘어 각 특성에서 평균, 분산, 표준편차 등의 통계적 특징을 추출하였습니다. Random Forest 모델을 이용하여 VR 및 비 VR 환경에서의 학생의 집중 상태 예측을 수행하였으며, 모델 성능 개선을 위한 피쳐 선택 및 하이퍼파라미터 최적화를 진행했습니다.

- **Performance Highlights**: 실험 결과, 컴퓨터 기반 학습 설정에서는 97.6%, 가상 현실 설정에서는 98%의 테스트 정확도를 기록하였습니다. 이는 온라인 교육 활동 중 학생의 집중에 대한 개인화된 통찰력을 제공하는 접근법의 효과를 강조합니다. 이러한 높은 정확도는 EEG 데이터와 기계 학습 기술의 결합이 어떻게 인지 상태 분류에 기여할 수 있는지를 보여줍니다.



### Fostering Inclusion: A Virtual Reality Experience to Raise Awareness of Dyslexia-Related Barriers in University Settings (https://arxiv.org/abs/2502.15039)
Comments:
          20 pages, 9 figures

- **What's New**: 이번 연구는 대학 환경에서 난독증을 가진 학생들의 포용성을 증진하기 위한 가상현실(Virtual Reality) 경험을 설계하고 구현한 최초의 시도로, 참여자들이 난독증 학생들이 겪는 어려움을 직접 경험할 수 있게 하였습니다. 전통적인 인식 제고 방법과는 달리, 이 몰입형 접근법은 비난독증인 사람들을 위해 난독증이 특히 어려운 점들을 체험하게 해줍니다. 연구 결과, 참가자들은 난독증 학생들이 직면한 장벽에 대한 인식을 높이고, 이 경험이 난독증 학생들에 대한 이해를 증진하는 데 유용한 도구가 되었다고 평가하였습니다.

- **Technical Details**: 이 연구에서 사용된 VR 경험은 가상의 캠퍼스 환경을 탐색하며 과제를 완료하는 형식으로 설계되었습니다. 참가자들은 바뀌는 문자로 표기된 표지판을 읽거나 잘못된 방향을 안내하는 화살표를 따르며, 따라서 난독증 학생들이 만나는 실제 장애물을 체험합니다. VR 경험은 실내 및 야외 공간을 포괄하는 다수의 건물로 구성되어 있으며, 다양한 이동 모드를 지원합니다.

- **Performance Highlights**: 30명 이상의 대학 참가자들이 만족도 조사를 통해 이 VR 경험의 유효성을 평가하였으며, 참가자들은 난독증 학생들이 직면한 장벽에 대한 인식을 높이는 데 기여했다고 보고했습니다. 이 연구는 VR의 역할을 교육 실무에 통합하는 데 중점을 두고 있으며, 난독증 학생들의 무관심을 줄이고 포용적 교육 전략을 촉진시키는 데 동기를 부여하고 있습니다.



### InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback (https://arxiv.org/abs/2502.15027)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문에서는 Large Multimodal Models (LMMs)의 인간 피드백과의 상호작용 지능을 평가하기 위한 새로운 프레임워크인 InterFeedback를 설계하였습니다. 이는 LMM과 데이터셋에 적용 가능하며, InterFeedback-Bench를 통해 다양한 오픈소스 LMM을 평가합니다. 추가적으로, OpenAI-o1 및 Claude-3.5-Sonnet와 같은 주요 모델의 상호작용 성능을 수동으로 테스트하기 위해 새롭게 수집한 120개의 사례를 포함한 InterFeedback-Human 데이터셋도 제공됩니다.

- **Technical Details**: InterFeedback 프레임워크는 LMM이 상호작용적으로 문제를 해결할 수 있도록 설계되었으며, 문제 해결 능력과 피드백 해석 능력을 평가하는 기준점인 InterFeedback-Bench를 도입합니다. 이 연구에서는 MMMU-Pro와 MathVerse라는 두 가지 도전적인 데이터셋을 활용하여 다양한 LMM의 상호작용 지능을 평가하였습니다. 최종적으로, LMM이 피드백을 통해 성능을 개선할 수 있는지에 대한 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, OpenAI-o1 등 최첨단 LMM조차도 인간의 피드백을 통해 50% 미만으로 결과를 수정하는 데 그쳤습니다. LMM은 높은 품질의 피드백을 요구하며, 저품질 피드백은 성능을 오히려 저하시킬 수 있다는 사실을 발견하였습니다. 이러한 결과는 LMM의 피드백 해석 및 반영 능력을 향상시킬 필요성을 강조합니다.



### Digital implementations of deep feature extractors are intrinsically informativ (https://arxiv.org/abs/2502.15004)
Comments:
          6 pages

- **What's New**: 이 논문에서는 깊은 특징 추출기(deep feature extractors)에서 정보 전파의 속도를 제안하는 상한을 증명했습니다. 이 작업은 유클리드 및 비유클리드 도메인을 포함하는 다양한 신경망 모델에 대한 통합된 프레임워크를 제시합니다. 또한 신호 도메인에 대한 구조적 정보를 추가하여 감쇠 속도를 명시적으로 결정하거나 향상시키는 방법도 보여줍니다.

- **Technical Details**: 이 연구에서 다양한 신경망 아키텍처의 무한 프로퍼게이션(depth) 및 미세한 측정 공간(measure space)에서의 에너지 감쇠를 제시하였습니다. 연구자는 LCA 그룹을 활용한 산란 CNN(scattering CNN)에서 에너지 분포의 실험적 관찰을 검증하며, 다양한 유클리드 및 비유클리드 도메인에서의 깊은 특징 추출기에 대한 안정성 정보 전파를 정량적으로 분석합니다.

- **Performance Highlights**: 결과적으로 일반적인 깊은 특징 추출기에 대한 지수적 에너지 감쇠가 도출되었습니다. 이 연구는 깊은 특징 추출기의 디지털 구현이 본질적으로 정보를 갖고 있음을 결론짓는데, 이는 에너지 전파의 속도가 신경망의 깊이에 따라 지수적으로 증가함을 의미합니다.



### Ultra-High-Frequency Harmony: mmWave Radar and Event Camera Orchestrate Accurate Drone Landing (https://arxiv.org/abs/2502.14992)
Comments:
          This paper is accepted by ACM SenSys 2025

- **What's New**: 이번 연구에서는 드론 착륙을 위한 mmWave 레이더와 새로운 이벤트 카메라를 결합한 mmE-Loc 시스템을 제안합니다. 기존의 프레임 카메라의 낮은 샘플링 주파수로 인한 효율성 문제를 극복하기 위해 이벤트 카메라를 도입하며, 이를 통해 높은 정확도와 낮은 지연 시간을 자랑하는 드론 지상 위치 추적 솔루션을 제공합니다. 실제 드론 배달 회사와의 연구에서 mmE-Loc이 기존의 최첨단 기법보다 우수한 결과를 보여주었음을 입증합니다.

- **Technical Details**: mmE-Loc은 드론의 고속 착륙을 위해 mmWave 레이더와 이벤트 카메라의 조합을 통해 정확한 위치 추적을 구현합니다. 이벤트 카메라는 각 픽셀의 강도 변화를 밀리초(ms) 수준의 해상도로 보고하며, 감속 없이 빠른 움직임을 포착할 수 있어 드론의 위치 추적에 적합합니다. 두 가지 혁신적인 모듈인 Consistency-Instructed Collaborative Tracking(CCT)와 Graph-Informed Adaptive Joint Optimization(GAJO)을 사용하여 드론 측정치를 정밀하게 추출하고 효율적인 센서 융합을 구현합니다.

- **Performance Highlights**: mmE-Loc은 30시간 이상의 실내 및 실외 실험을 통해 평균 0.083m의 위치 정확도와 5.15ms의 지연 시간을 달성하며, 이는 기존 방법보다 각각 48%와 62% 향상된 수치입니다. 다양한 비행 조건에서도 최소한의 민감도를 유지하면서 안정적인 성능을 보이며, 실제 드론 배달 공항에서 10시간 동안의 실험을 통해 상용 수준의 착륙 요건을 충족함을 입증했습니다.



### EigenShield: Causal Subspace Filtering via Random Matrix Theory for Adversarially Robust Vision-Language Models (https://arxiv.org/abs/2502.14976)
- **What's New**: EigenShield는 기존의 방어 전략들이 가진 취약성을 극복하기 위해 Random Matrix Theory를 기반으로 한 새로운 방어 기법입니다. 이 모델은 VLM(비전-언어 모델)의 고차원 표현에서 적대적 방해를 정량화하여 이를 효과적으로 탐지하고 필터링합니다. 저자들은 특히 이동 평균 구조를 이용해 의미론적 정보가 포함된 인자와 적대적 인자 간의 구분을 시도합니다.

- **Technical Details**: EigenShield의 주요 기법은 spiked covariance model을 사용하여 VLM의 특성 표현에서 의미 있는 신호 구성 요소와 적대적 노이즈를 구분하는 것에 중점을 둡니다. Robustness 기반 비정규성 점수(RbNS)와 분위수 기반 임계값을 통해 의미론적 정보가 포함된 인자(eigenvectors)를 선별합니다. 이 방법은 모델의 매개변수를 수정하거나 적대적 훈련을 요구하지 않으면서도 높은 차원의 데이터를 효과적으로 관리할 수 있는 원리를 제공합니다.

- **Performance Highlights**: EigenShield는 전통적인 방어 방식들과 비교할 때 공격 성공률을 크게 줄이는 성능을 입증했습니다. 기존의 방어 전략들인 UNIGUARD, CIDER와 같은 방법들을 초월하여 VLM의 안정성을 높이는 데 기여합니다. 이 혁신적인 접근 방식은 처리 효율성 역시 갖추고 있어 실시간 방어 시스템에서도 활용 가능할 것으로 보입니다.



### Design of a Visual Pose Estimation Algorithm for Moon Landing (https://arxiv.org/abs/2502.14942)
Comments:
          6 pages, 8 figures, Presented in 11th Nano-Satellite Symposium

- **What's New**: 이번 연구에서는 우주선의 위치와 태세를 추정하기 위한 지형 절대 항법(terrain absolute navigation) 방법을 제안합니다. 이 방법은 우주선 아래에 위치한 분화구(crater)의 정보를 활용하여 정밀한 위치 추정을 가능하게 합니다. 기존의 절대 항법 솔루션을 기반으로 하여, 원거리 표적에 대한 의존성을 줄임으로써 일반적으로 발생하는 항법 드리프트(navigational drift)를 보정합니다.

- **Technical Details**: 제안된 알고리즘은 우주선에 장착된 카메라로 포착된 분화구의 위치 정보를 사용하여 두 가지 주요 단계로 구성됩니다: 이미지 처리(image processing) 및 분화구 일치(crater matching). 이 과정들 대신에 분화구 데이터베이스(crater database)를 미리 설정하여 알고리즘의 정확성을 중심으로 초점을 맞추는 방식으로 진행됩니다.

- **Performance Highlights**: 시뮬레이션을 통해 알고리즘의 정확성과 추정에 사용된 분화구의 개수(number of craters)의 영향을 평가하였습니다. 이 실험 결과는 제안된 알고리즘이 Moon 착륙 시 정확한 위치 제시에 큰 기여를 할 수 있음을 시사합니다.



### Compact Latent Representation for Image Compression (CLRIC) (https://arxiv.org/abs/2502.14937)
- **What's New**: 이번 연구에서는 기존의 모델들을 활용하여 지각 이미지 압축(perceptual image compression)을 위한 혁신적인 접근 방식을 제안합니다. 전통적인 이미지 압축 모델들이 각각의 품질 수준에 대해 별도의 모델을 필요로 하는 반면, 이 방법은 싱글 이미지에 최적화된 함수(overfitted function)를 통해 다양한 품질 수준에서 압축을 가능하게 합니다. 기존의 Latent 변수를 이용하여 리소스를 효율적으로 사용하고, 모델의 복잡성을 감소시킵니다.

- **Technical Details**: 이 방법은 과적합(overfitting)된 학습 가능한 함수(learnable functions)를 통해, 다양한 품질 수준에서의 압축을 지원합니다. 특히, Stable Diffusion과 같은 기존의 Latent image 모델에서 생성된 Latent 변수를 활용하며, 이미지 공간(image space) 대신 Latent 공간(latent space)에서 작동하기 때문에 메모리와 계산 자원 요구사항이 대폭 감소합니다. 이로 인해, 효율적인 비트레이트 및 높은 화질을 유지하면서도 리소스 집약적인 학습 과정을 피할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 최신 학습 기반 이미지 압축 모델과 비교하여 동등한 지각 품질을 유지하면서도, 분리된 모델 없이 다양한 품질 수준의 이미지 압축이 가능합니다. 우리의 접근법은 최초 품질 수준을 적응시키는 연속적인 품질 표현을 실현하고, 인지적 품질을 최적화하여 인간의 시각 체계에 적응하도록 설계되었습니다. 이 새로운 접근 방식을 통해 향후 다양한 이미지 압축 방식의 개발 가능성이 열렸습니다.



### Reducing false positives in strong lens detection through effective augmentation and ensemble learning (https://arxiv.org/abs/2502.14936)
Comments:
          15 pages, 14 figures, 7 tables, Accepted for publication in MNRAS

- **What's New**: 이번 연구에서는 고품질(training quality) 데이터셋이 Convolutional Neural Networks (CNNs)의 강력한 중력 렌즈 감지 성능에 미치는 영향을 분석하였습니다. 데이터의 다양성과 대표성(data representativeness)의 중요성을 강조하며, 샘플 집단의 변동성이 CNN 성능에 미치는 영향을 보여주고 있습니다. 또한, 데이터 증강(data augmentation) 및 앙상블 학습(ensemble learning) 기술이 모델의 완전성을 유지하면서도 오탐지(false positives)를 줄이는 데 효과적임을 입증하였습니다.

- **Technical Details**: 연구에서는 DenseNet 및 EfficientNet 변형을 이용한 실험을 통해 최고 오탐지율(false positive rate) $10^{-4}$에 도달하였으며, 테스트 데이터셋에서 실제 강력한 중력 렌즈의 88% 이상을 정확히 식별했습니다. 이는 기존 훈련 데이터셋에 비해 오탐지율이 11배 감소한 결과로, 진정한 긍정(true positive) 샘플 수의 감소는 단 2.3%에 불과합니다. 연구 결과는 KiDS 데이터셋에서 검증되었고, 유사한 향후 미션에 적용 가능한 통찰력을 제공합니다.

- **Performance Highlights**: 본 연구의 결과는 강력한 중력 렌즈 감지 모델의 견고성을 크게 향상시키며, 데이터셋의 품질이 중요한 요소임을 강조합니다. CNN 모델들은 다양한 이미지 조사에 맞춰 개발된 여러 지능형 접근 방식과 비교하여 효과적으로 활용될 수 있으며, 특히 대규모 천문 조사에서 성능을 극대화하는 데 기여할 수 있습니다. 이러한 발전은 우주 탐사 및 천문학 적용 분야에서 CNN의 중요성을 더욱 부각시키고 있습니다.



### Denoising, segmentation and volumetric rendering of optical coherence tomography angiography (OCTA) image using deep learning techniques: a review (https://arxiv.org/abs/2502.14935)
- **What's New**: 본 논문에서는 Optical coherence tomography angiography (OCTA) 데이터 분석에 딥러닝(DL) 모델을 활용하는 최근 5년간의 연구를 정리하고 있습니다. OCTA는 망막과 맥락막의 미세혈관 구조를 비침습적으로 연구할 수 있는 중요한 기술로, 딥러닝을 통해 패턴 인식 및 잡음 제거에서 큰 잠재력을 가지고 있습니다. 이 연구는 OCTA 이미지를 분석하기 위한 DL 모델의 현재 문제점과 설계 원칙에 대해 심도 있게 논의했습니다.

- **Technical Details**: OCTA는 저간섭 인터페로메트리(low coherence interferometry) 원리를 기반으로 하며, 이미지를 획득하기 위해 B-스캔 이미지를 이용합니다. 현재 SD-OCTA(스펙트럴 도메인 OCTA)와 SS-OCTA(스위프트 소스 OCTA)의 두 가지 기술이 사용되고 있습니다. DL 기술은 OCTA 이미지에서 발생하는 다양한 잡음과 인공물을 자동으로 감지 및 제거하는 데 효과적이며, 이미지 품질 향상뿐만 아니라 구조를 식별하는 데 또한 활용됩니다.

- **Performance Highlights**: OCTA 이미지는 망막 혈관 구조 및 미세 순환 상태를 모니터링하는 데 큰 잠재력을 보여주며, DL 모델이 이러한 데이터를 분석하는 데 효과적인 도구로 자리 잡고 있습니다. 기존 연구에서 많은 AI 기반 모델이 이미지 강화 및 질병 관련 표현형 식별에서 우수한 성능을 보였으나, OCTA 데이터 세트의 불균질성과 클래스 불균형 문제로 인해 여전히 개선이 필요합니다. 이 리뷰는 OCTA 기반 DL 모델의 성능을 종합적으로 평가하고, 관련 분야 엔지니어 및 임상의가 활용할 수 있는 귀중한 통찰력을 제공합니다.



### Distributed U-net model and Image Segmentation for Lung Cancer Detection (https://arxiv.org/abs/2502.14928)
- **What's New**: 이 논문은 COVID-19 대유행 이후로 증가한 폐 질환, 특히 폐암 및 만성 폐쇄성 폐질환(COPD)에 대한 연구를 심화합니다. 연구진은 조기 발견과 정확한 진단의 중요성을 강조하며, 고급 심층학습 모델인 U-Net을 이용한 컴퓨터 지원 설계(CAD) 시스템의 가능성을 탐구합니다. VGG16 알고리즘을 통해 CAD 시스템의 시뮬레이션 능력을 강화하는 방법도 제시됩니다.

- **Technical Details**: 연구에서는 다양한 하드웨어 설정, 즉 단일 CPU, 단일 GPU, 분산 GPU 및 연합 학습(federated learning) 하에서 U-Net 모델의 효율성을 엄격하게 평가합니다. 본 연구에 사용된 데이터셋은 여러 학술 기관이 작성한 폐 CT 이미지와 해당 분할(segmentation) 마스크로 구성되어 있습니다. 이 데이터셋은 실증적 검증(empirical validation)을 위한 기반을 제공합니다.

- **Performance Highlights**: U-Net 모델의 실증적 결과는 네 대의 GPU를 활용한 분산 학습에서 특히 강력한 성능을 나타냅니다. 연구 결과는 U-Net 기반 CAD 시스템이 폐 질환의 정확하고 신속한 감지 및 진단에서 큰 잠재력을 가지고 있음을 보여줍니다. 이 연구는 의료 진단 분야에서의 AI 시스템 발전에 기여할 것입니다.



### Display Field-Of-View Agnostic Robust CT Kernel Synthesis Using Model-Based Deep Learning (https://arxiv.org/abs/2502.14920)
Comments:
          Accepted at IEEE ISBI 2025

- **What's New**: 본 논문에서는 X-레이 컴퓨터 단층촬영(CT)에서 이미지 기반 커널 합성을 위한 프레임워크를 제안합니다. 기존 방법들은 직접적인 커널 합성(direct kernel synthesis) 사용 시, 재구성되고자 하는 이미지의 품질이 저하되는 경향이 있었습니다. 고유의 Pre-defined 모델을 활용하여 커널의 특성과 DFOV(Displayed Field of View)를 외부 모델로 통합시키는 접근법을 통해 효율적으로 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 모델 기반의 딥러닝(model-based deep learning) 접근법으로 커널 합성 문제를 다루고 있습니다. 이 시스템은 딥러닝 기반의 프로젝션 단계와 데이터 일관성을 유지하는 반복적인 해법을 포함합니다. 커널의 MTF(modulation transfer function)와 DFOV 정보를 명시적으로 통합함으로써, 다양한 DFOV에 대해 단일 네트워크를 훈련할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과는 임상 데이터에서 측정된 결과와 와이어 팬텀(wire phantom) 데이터의 변조 전송 함수 분석을 포함하여 제안된 방식의 유용성을 보여줍니다. DFOV 변동에 대해 제안된 방법이 보다 견고하다는 것을 입증하며, 다른 네트워크와 비교한 결과, 노이즈 및 아티팩트 관리에 있어 탁월한 성능을 나타냅니다.



### Pulmonary Tuberculosis Edge Diagnosis System Based on MindSpore Framework: Low-cost and High-precision Implementation with Ascend 310 Chip (https://arxiv.org/abs/2502.14885)
- **What's New**: 이번 연구는 저비용으로 고성능의 폐결핵 진단 보조 시스템을 제안합니다. Huawei의 MindSpore 프레임워크와 Ascend 310 엣지 컴퓨팅 칩을 활용하여, MobileNetV3 아키텍처를 적용한 시스템입니다. 특히, 이 시스템은 주 치료 환경에서 접근 가능한 저비용의 AI 진단 솔루션을 제공하는 데 목표를 두고 있습니다.

- **Technical Details**: 연구에서 사용된 Ascend 310 칩은 8W의 최대 전력 소비량을 자랑하며, 엣지 컴퓨팅 시나리오에 적합합니다. 모델 훈련 과정에서는 Softmax 교차 엔트로피 손실 함수와 모멘텀 옵티마이저를 사용하여, FP16 하이브리드 정확성을 제공하는 Orange Pi AIPro 엣지 장치에서 성능을 발휘합니다. MobileNetV3는 경량의 합성곱 신경망 모델로, Squeeze-and-Excitation 모듈과 Swish 활성화 함수를 통해 높은 정확성을 유지합니다.

- **Performance Highlights**: 모델은 4148개의 흉부 이미지를 포함한 테스트 세트에서 99.1%의 정확도를 달성하였으며, AUC 값은 0.99에 도달했습니다. 또한, 전체 장비 비용이 $150에 유지되어, 의료 자원이 부족한 지역에서도 이용 가능한 효과적인 진단 솔루션을 제공합니다. 이러한 성과는 AI가 보건 환경에서 진단 효율성을 크게 개선할 수 있는 가능성을 보여줍니다.



### A Survey of Safety on Large Vision-Language Models: Attacks, Defenses and Evaluations (https://arxiv.org/abs/2502.14881)
Comments:
          22 pages, 2 figures

- **What's New**: 이 논문은 Large Vision-Language Models (LVLMs)의 안전성에 대한 포괄적 분석을 제공하며, 공격, 방어, 평가 방법들을 아우르는 통합 프레임워크를 제안합니다. LVLM의 생애 주기를 분석하여 추론(inference)과 훈련(training) 단계로 구분하고, 향후 연구 방향을 제시하여 LVLM의 강인함을 강화할 수 있는 기초 자료를 제공합니다. 또한, 최신 LVLM인 Deepseek Janus-Pro에 대한 안전성 평가를 수행하며, 결과에 대한 이론적 분석을 포함합니다.

- **Technical Details**: LVLM의 안전성 연구는 공격, 방어, 평가의 세 가지 범주로 구분됩니다. 공격 연구는 LVLM의 취약점을 발견하고 완화하는 데 초점을 맞추고 있으며, 방어 전략은 모델의 강인성을 향상시키는 데 중점을 둡니다. 평가는 다양한 LVLM의 보안 기능을 평가하는 벤치마크를 생성하여 연구자들이 안전성 조치를 비교하고 평가할 수 있는 표준 프레임워크를 제공합니다. 이러한 세 가지 요소가 긴밀히 연결되어 전체적인 LVLM 보안 환경을 이해하는 데 중요한 역할을 합니다.

- **Performance Highlights**: LVLM의 다중 모드 특성은 독특한 취약점을 도입하여 특정한 입력 변조가 시스템 전체에 걸쳐 위험한 출력으로 작용할 수 있습니다. 특히, 변화된 의료 이미지가 잘못된 진단으로 이어지거나, 변조된 재무 데이터가 위험 평가를 왜곡할 수 있는 실질적인 위험을 강조합니다. 연구에서는 LVLM의 강건성, 신뢰성, 윤리적 정렬의 중요성을 강조하고 있으며, 고위험 실제 응용 프로그램에서의 안전한 배치를 보장하기 위한 전략적 추천을 제공합니다.



### d-Sketch: Improving Visual Fidelity of Sketch-to-Image Translation with Pretrained Latent Diffusion Models without Retraining (https://arxiv.org/abs/2502.14007)
Comments:
          Accepted in The International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문은 자유로운 손으로 그린 스케치를 기반으로 현실적인 이미지를 생성하는 새로운 기법을 제안합니다. 기존의 GAN(Generative Adversarial Networks) 방식이 아닌, 대규모로 사전 훈련된 확산 모델을 활용하여 구조적인 가이드를 제공합니다. 구체적으로, 학습 가능한 경량 매핑 네트워크를 도입하여 소스 도메인(스케치)과 타겟 도메인(이미지) 간의 특성 변환을 지원합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 잠재적 확산 모델의 특징 공간을 활용하여, 재훈련 없이도 고해상도의 사실적인 이미지 생성을 가능하게 합니다. 이 과정은 GAN 방식의 불안정성과 대규모 모델의 높은 계산 비용을 줄여주는 안정적인 최적화를 제공합니다. 연구에서는 딥러닝을 통해 추출된 특징을 활용하여 원본 스케치와 생성된 이미지 간의 상관관계를 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 기술들과 비교하여 질적 및 양적 기준에서 우수한 성능을 보였습니다. 특히, 고해상도의 사실적인 이미지 생성 능력에서 중요한 개선을 이루었으며, 사용자의 예술적 전문성에 관계없이 손으로 그린 스케치에서 매력적인 이미지를 생성할 수 있는 가능성을 보여주었습니다.



### Temporal Misalignment in ANN-SNN Conversion and Its Mitigation via Probabilistic Spiking Neurons (https://arxiv.org/abs/2502.14487)
- **What's New**: 이 논문에서는 Spiking Neural Networks (SNNs)의 ANN-SNN 변환 과정에서 발생하는 'temporal misalignment'라는 현상을 밝혀냈습니다. 이 현상은 SNN 레이어 전반에서 랜덤 스파이크 재배치가 성능 향상을 초래한다는 것을 보여줍니다. 이를 기반으로, 두 단계 확률적 스파이킹 뉴런(Two-Phase Probabilistic spiking neurons)을 도입하여 변환 과정을 더욱 개선하고 있습니다.

- **Technical Details**: 기존의 Integrate-and-Fire (IF) 스파이킹 뉴런 모델을 기반으로 하여 ANN과 SNN 간의 변환 과정을 수식으로 설명합니다. 이 과정은 변환의 정확성을 높이며, 헤비스타이드 함수와 같은 수학적 요소를 통해 SNN의 내부 동역학을 모델링합니다. 논문에서는 ANN에서 SNN으로 가중치와 편향을 전송하는 방법을 자세히 다루고 있으며, 이 과정의 효용성을 강조합니다.

- **Performance Highlights**: CIFAR-10/100, CIFAR10-DVS, ImageNet 데이터셋을 통한 실험에서 제안한 방법이 최신 SOTA 변환 및 다른 훈련 방법들보다 우수한 정확도를 달성했습니다. 다양한 아키텍처를 통한 포괄적 실험을 통해 이 방법의 효과가 입증되었으며, 이로써 에너지 효율적인 AI 시스템 구축의 가능성을 제시합니다.



New uploads on arXiv(cs.AI)

### AutoToM: Automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind (https://arxiv.org/abs/2502.15676)
Comments:
          23 pages, 6 figures, 11 tables. Website at this https URL

- **What's New**: 이번 연구에서는 AutoToM이라는 새로운 자동화된 Bayesian Theory of Mind 방법론을 소개합니다. 기존의 방법들은 주로 LLM(대형 언어 모델)을 활용하거나 수동적으로 설계된 Bayesian 모델에 의존하는데, 이는 각기 다른 분야에서의 일반화에 한계가 있습니다. AutoToM은 다양한 도메인에서 작동할 수 있으며, 어떠한 정신 변수를 추론할 수 있는 가능성을 제공하므로 개방적인 기계 이론의 마음을 실현할 수 있게 됩니다.

- **Technical Details**: AutoToM은 초기 BToM 모델을 제안하고, 제안된 모델에 기반하여 자동화된 Bayesian 역 계획(Inverse Planning)을 수행합니다. 여기서 LLM을 백엔드로 사용하며, 추론의 불확실성에 따라 모델을 반복적으로 정제하여 추가적인 정신 변수나 시간 단계를 통합합니다. 이러한 접근법은 강력한 Theory of Mind 추론을 구현할 수 있도록 해줍니다.

- **Performance Highlights**: 여러 Theory of Mind 벤치마크에 대한 실증 평가 결과, AutoToM은 지속적으로 최첨단 성능을 달성하였습니다. 이는 AutoToM이 확장 가능하고, 견고하며 해석 가능한 기계 이론의 마음 접근 방식이 됨을 의미합니다. 이를 통해 기계가 사회적으로 지능적인 에이전트로 발전할 가능성이 한층 높아졌습니다.



### Automating Curriculum Learning for Reinforcement Learning using a Skill-Based Bayesian Network (https://arxiv.org/abs/2502.15662)
- **What's New**: 본 논문에서는 자동화된 커리큘럼 생성의 어려움을 해결하기 위해 SEBNs(Skill-Environment Bayesian Networks)를 소개합니다. SEBN은 기술, 보상 구조와 관련된 목표, 환경 특징 간의 확률적 관계를 모델링하여 정책 성능을 예측합니다. 이 방법을 통해 기존의 자동 커리큘럼에 비해 더 나은 성과를 거둘 수 있는 가능성을 보여줍니다.

- **Technical Details**: SEBN은 과거 데이터를 기반으로 기술적 역량(laten competency)과 환경 특징 간의 관계를 모델링 합니다. 이를 통해 에이전트의 성공률을 예측하고, 새로운 환경에서의 성공 가능성을 바탕으로 다음 훈련 과제를 선택하는 SEBN 가이드 자동 커리큘럼을 구성합니다. 이러한 접근 방식은 각 환경에 대한 명시적 평가 없이도 가능하다는 점에서 큰 장점을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, SEBN을 활용한 커리큘럼이 DoorKey(이산격자), BipedalWalker(연속제어), robosuite(로봇 시뮬레이션) 등의 다양한 환경에서 다른 기준 모델들보다 더 빠르고 안정적인 정책을 달성하는 것으로 나타났습니다. 이는 SEBN 기반의 커리큘럼이 로봇 에이전트의 학습 성과를 극대화하는 데 효과적임을 보여줍니다.



### Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path? (https://arxiv.org/abs/2502.15657)
- **What's New**: 이번 논문에서는 일반ist AI 에이전트(agents)에 대한 위험성을 강조하고 있습니다. 특정 목표를 자율적으로 추구할 수 있는 시스템은 유용할 수 있지만, 악의적인 행위자에 의한 남용이나 인간 통제의 상실과 같은 문제를 야기할 수 있습니다. 이러한 위험에 대응하기 위해, 저자들은 비 에이전틱(non-agentic) AI 시스템, 즉 Scientist AI를 개발할 필요성을 제안합니다.

- **Technical Details**: Scientist AI는 관찰(observations)에서 세계를 설명하는 시스템으로 설계되었습니다. 이 시스템은 데이터(data)를 설명하기 위한 이론을 생성하는 세계 모델(world model)과 질문-응답 추론 기계(question-answering inference machine)로 구성됩니다. 또한, 이러한 두 구성 요소는 예측의 과신(overconfident predictions)을 완화하기 위한 명확한 불확실성(notion of uncertainty)을 바탕으로 작동합니다.

- **Performance Highlights**: Scientist AI는 AI 안전성을 포함한 과학적 진전을 가속화하는 데 인간 연구자를 지원할 수 있는 잠재력을 가지고 있습니다. 또한, 저자는 위험이 따르는 AI 에이전트에 대한 안전 장치(guardrail)로 사용할 수 있는 가능성도 제시합니다. 이러한 비 에이전틱 AI에 대한 초점을 맞추는 것은 AI 혁신의 이점을 누리면서 현재 진행되고 있는 경로에 따른 위험을 피할 수 있는 방법이 될 수 있습니다.



### Empowering LLMs with Logical Reasoning: A Comprehensive Survey (https://arxiv.org/abs/2502.15652)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 논리적 추리 능력과 관련된 주요 도전 과제를 두 가지 측면에서 정리하고 분류합니다. 첫 번째는 논리적 질문 응답(logical question answering)으로, LLMs가 복잡한 논리 문제에서 올바른 답변을 생성하는 데 어려움을 겪는 점입니다. 두 번째는 논리적 일관성(logical consistency)으로, 다양한 질문에 대해 자신과 모순되는 응답을 생성할 수 있음을 강조합니다.

- **Technical Details**: LLMs는 신뢰성과 신뢰성을 높이기 위해 외부 해결자(external solvers), 프롬프트(prompts), 사전 학습(pretraining), 미세 조정(fine-tuning) 기반의 다양한 기법을 분류하여 논리적 질문에 대한 정확한 답변을 도출하는 방법을 논의합니다. 특히, 복잡한 논리 질문 처리와 관련된 여러 최첨단 방법을 체계적으로 조사하며 그 방법들에 대한 상세한 유형 분류를 제안합니다.

- **Performance Highlights**: LLMs는 복잡한 실제 상황에서 문제 해결 및 의사 결정에 적용하기 어려운 현실적인 제약을 안고 있습니다. 예를 들어, LLaMA-13B 모델은 FOLIO 데이터셋에서 33.63%의 정확도를 달성했으며, 이는 진짜, 거짓 및 알 수 없음에 대한 무작위 추측보다 겨우 높은 수치입니다. 이러한 연구 방향으로 LLM의 논리적 추리 능력을 향상하기 위한 잠재적인 연구 분야도 논의됩니다.



### Paradigms of AI Evaluation: Mapping Goals, Methodologies and Cultur (https://arxiv.org/abs/2502.15620)
- **What's New**: 최근 AI 평가 연구는 다양해진 배경과 목표를 가진 연구자들로부터 많은 관심을 받고 있으며, 이를 통해 복합적인 평가 패러다임이 등장했습니다. 이 논문은 AI 평가의 다양한 접근 방식을 조사하여 6가지 주요 패러다임을 정의하고, 각 패러다임이 지닌 목표, 방법론 및 연구 문화에 대한 최근 기여를 특징짓고 있습니다. 이를 통해 AI 시스템의 평가 접근법에 대한 인식을 높이고, 다른 패러다임 간의 상호작용을 촉진하는 것을 목표로 합니다.

- **Technical Details**: AI 평가를 정의할 때, 행동 속성과 사회적 영향을 측정하는 과정을 중점적으로 다루며, 125개 이상의 논문을 조사하여 다채로운 AI 평가 방법론을 포괄하는 관점을 제공하고 있습니다. 각 논문은 목표, 방법론, 문화와 관련된 분석 차원을 체계적으로 주석 처리하였으며, 이러한 주석을 통해 각 패러다임이 다루는 질문과 접근 방식을 명확히 하고 있습니다. 이는 연구자들이 다양한 평가 접근법을 탐색하고 비판적으로 평가할 수 있도록 돕기 위한 작업입니다.

- **Performance Highlights**: AI 평가 분야는 고유의 요소와 문제를 가지고 있으며, 이를 통해 다양한 접점을 탐색할 수 있습니다. 연구자와 실무자들 사이의 비교와 협력을 촉진하고, 향후 연구 방향을 제시하기 위해 현재 AI 평가의 간극을 식별하는 것이 중요합니다. 이러한 작업은 AI 시스템의 안전한 배포와 책임 있는 적용을 보장하는 데 기여할 잠재력을 지니고 있습니다.



### Zweistein: A Dynamic Programming Evaluation Function for Einstein Würfelt Nicht! (https://arxiv.org/abs/2502.15547)
- **What's New**: 이번 논문에서는 Einstein Würfelt Nicht! (EWN)를 위한 새로운 동적 프로그래밍 평가 함수인 Zweistein을 소개합니다. 기존의 인간 지식에 기반한 평가 함수 대신 데이터 중심 접근 방식을 사용하여 파라미터 튜닝이 필요 없습니다. Zweistein은 조각들과의 거리 벡터를 기록하여 EWN의 본질을 포착하며, TCGA 2023 대회에서 첫 번째 자리를 차지하는 성과를 올렸습니다.

- **Technical Details**: EWN은 5x5 보드에서 진행되는 두 플레이어 확률 게임으로, 이 게임에서는 주사위를 굴려 조각을 이동시키며, 조각의 라벨과 주사위 숫자가 일치해야 이동이 가능합니다. 이 논문은 EWN-simple이라는 새로운 개념을 도입하여 조각 잡기 규칙을 제거하고, Chebyshev 거리 기반의 배열을 활용하여 보드의 이형성을 감소시키며 메모리 활용성을 높이는 방법을 설명합니다.

- **Performance Highlights**: Zweistein은 기존의 평가 함수와 비교하여 우수한 성능을 보였으며, 이는 EWN의 복잡한 게임 상태를 효과적으로 평가할 수 있는 새로운 접근 방식을 제시합니다. 이 평가 함수는 기계 학습 및 몬테 카를로 방법 대신 사용될 수 있는 가능성을 보여 주며, 특히 EWN과 같은 새로운 게임에 대한 가치 평가에 중요한 기여를 할 것으로 예상됩니다.



### TAG: A Decentralized Framework for Multi-Agent Hierarchical Reinforcement Learning (https://arxiv.org/abs/2502.15425)
- **What's New**: 기존의 단일 구조가 아니라, 완전히 분산화된 계층적 다중 에이전트 시스템을 구축할 수 있는 TAME Agent Framework (TAG)를 소개합니다. TAG는 새로운 LevelEnv 개념을 통해 임의 깊이의 계층을 가능하게 하여 각각의 계층을 에이전트가 상호작용하는 환경으로 추상화합니다. 이를 통해 다양한 에이전트 타입의 원활한 통합이 가능해지며, 정보 흐름을 표준화하면서 느슨한 결합을 유지합니다.

- **Technical Details**: TAG의 핵심 혁신은 각 계층 수준을 환경으로 간주하는 LevelEnv 추상화입니다. 이 접근 방식은 여러 수평적 계층으로 구성된 시스템을 만들어, 에이전트가 아래 계층과 상호작용할 때의 복잡성을 줄여줍니다. 또한, 중앙 집중식 제어 없이 효율적인 조정을 가능하게 하는 유연한 통신 프로토콜을 지원하며, 이질적인 에이전트도 다양한 학습 알고리즘을 통해 사용할 수 있도록 합니다.

- **Performance Highlights**: TAG의 효과는 기존의 멀티-에이전트 강화 학습(MARL) 벤치마크에서의 실험을 통해 입증되었습니다. 여러 두 개 및 세 개의 수준의 계층을 구현하여 표준 벤치마크에 대해 샘플 효율성과 최종 성능이 개선된 결과를 보여주었습니다. 이러한 성과는 분산된 계층적 조직이 학습 속도와 최종 성능을 모두 향상시킬 수 있음을 시사합니다.



### Chitrarth: Bridging Vision and Language for a Billion Peop (https://arxiv.org/abs/2502.15392)
- **What's New**: 최근 다중 모달 기초 모델의 주된 훈련은 영어 및 자원 풍부한 유럽 언어 데이터에 한정되어 있어, 그 결과 중소 및 저자원 언어에서의 적용 가능성이 제한되었습니다. 이를 해결하기 위해, 우리는 10가지 주요 인도 언어의 언어적 다양성과 시각적 추론을 목표로 하는 포괄적인 비전-언어 모델(Chitrarth)을 소개합니다. 이 모델은 다언어 대형 언어 모델(LLM)과 비전 모듈을 효과적으로 통합하여 자원 부족 언어에 대한 새로운 기준을 설정하고, 보다 다양한 AI 시스템 개발에 기여할 것을 목표로 하고 있습니다.

- **Technical Details**: Chitrarth 모델은 전처리된 비전 인코더와 Krutrim LLM 백본을 사용하여 이미지와 언어 모달리티를 활용하며, 비전 토큰과 텍스트 토큰을 결합하여 LLM에 입력됩니다. 우리는 다중 모달 학습을 위해 이미지 인코딩을 시작하며, 그 결과로 생성된 비전 임베딩을 LLM 임베딩 공간으로 매핑하는 프로젝션 레이어를 활용합니다. 이 과정은 다양한 외부 데이터셋을 통해 훈련된 멀티링구얼 데이터셋을 번역하여 인도 언어 간의 교차 언어 일반화 능력을 향상시키는 것을 포함합니다.

- **Performance Highlights**: Chitrarth 모델은 영어 데이터셋에서 SOTA 결과를 달성했으며, 제안된 다중 언어 데이터셋에서도 뛰어난 성능을 보였습니다. 우리는 교육 전략 및 변칙 실험을 통해 5개 영어 데이터셋 중 3개에서 SOTA 결과를 기록하였고, 10개의 인도 언어에 대한 종합 평가 벤치마크인 BharatBench를 제안하여, 저자원 언어에 대한 진전을 촉진하고자 합니다. 이는 다중 모달 능력 강화와 AI 기술 발전에 크게 기여할 것으로 기대됩니다.



### ARS: Automatic Routing Solver with Large Language Models (https://arxiv.org/abs/2502.15359)
- **What's New**: 이 논문은 1,000가지의 VRP 변형을 포함하는 RoutBench라는 벤치마크를 소개합니다. 이 벤치마크는 24가지 특성(source)에서 파생된 것으로, 자동화된 라우팅 솔버의 효과를 평가하는 데 사용됩니다. 또한, Large Language Model (LLM) 에이전트를 활용하여 자동으로 제약 조건 인지(heuristic code)를 생성하는 Automatic Routing Solver (ARS) 프레임워크를 제안합니다. 이를 통해 현실 세계의 복잡한 제약 조건을 해결할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: Automatic Routing Solver (ARS)는 세 가지 주요 구성 요소로 구성됩니다: 1) 미리 정의된 데이터베이스, 2) 제약 조건 인식 휴리스틱 생성, 3) 증강된 휴리스틱 솔버. 제안된 ARS는 자연어 형식의 문제 설명과 각 VRP 변형에 대한 인스턴스 데이터를 기반으로 제약 조건을 인식하는 휴리스틱을 자동으로 생성할 수 있습니다. 데이터베이스는 기본 VRP 정보와 여섯 가지 대표 제약 조건으로 구성되어 있습니다.

- **Performance Highlights**: ARS는 기존 LLM 기반 방법보다 우수한 성능을 보입니다. 실험 결과, ARS는 일반적으로 사용되는 VRP의 91.67%를 자동으로 해결하고 모든 벤치마크에서 최소 30%의 개선을 달성했습니다. 이러한 결과는 제안된 접근 방식이 다양한 VRP 문제를 효과적으로 처리할 수 있음을 나타냅니다.



### Measuring AI agent autonomy: Towards a scalable approach with code inspection (https://arxiv.org/abs/2502.15212)
Comments:
          NeurIPS Socially Responsible Language Modelling Research (SoLaR) Workshop 2024

- **What's New**: 이 논문에서는 AI 에이전트의 자율성을 평가하는 새로운 접근 방식을 소개합니다. 기존의 실행 기반(run-time) 평가와 달리, AI 에이전트를 특정 작업을 수행하지 않고도 코드 기반(code-based) 접근 방식을 통해 자율성을 평가할 수 있는 방법을 제시합니다. 이 방법은 비용과 위험을 줄이면서 에이전트의 자율성을 측정할 수 있습니다.

- **Technical Details**: AI 에이전트의 자율성 수준을 측정하기 위해 AutoGen 프레임워크를 활용하며, 자율성의 영향과 감독(overight) 특성에 따라 분류된 세 Taxonomy를 제시합니다. 본 연구는 자율적 행동(decision autonomy)을 정의하며, 자율성의 leve를 기준으로 규제와 정책을 형성하는 데 필요한 실증적 기준을 제시합니다. 또한, 코드 이스펙션(code inspection) 방법론을 통해 에이전트가 어떻게 구조화되어 자율적으로 기능하는지를 평가합니다.

- **Performance Highlights**: AutoGen 프레임워크는 다양한 자율성 수준을 고려하여 언어 모델을 기반으로 한 AI 에이전트 시스템을 구축하는 데 널리 사용되는 오픈소스 소프트웨어입니다. 이 논문은 AutoGen의 대표 속성을 바탕으로 자율성에 대한 초기 평가를 제공하며, 향후 연구에서 광범위한 오픈소스 AI 에이전트 프레임워크와 애플리케이션을 포괄하도록 확장할 수 있는 가능성을 가지고 있습니다.



### The Imitation Game for Educational AI (https://arxiv.org/abs/2502.15127)
- **What's New**: 이 논문은 AI가 학생의 사고 및 추론을 이해하는 능력을 평가하기 위한 새로운 방법론을 제안합니다. 기존의 평가 방식들이 긴 연구 기간과 혼란 변수들로 인해 한계가 있었던 반면, 제안된 방법론은 두 가지 단계로 구성된 Turing 테스트를 기반으로 AI의 이해력을 직접 검증합니다. Phase 1에서는 학생들이 오픈형 질문에 대한 답변을 제공하여 자연스러운 오해를 드러내고, Phase 2에서는 AI가 학생의 특정 오류에 기반한 새로운 질문을 생성하게 됩니다.

- **Technical Details**: 제안된 평가 프레임워크는 두 가지 구분된 단계로 구성되어 있으며, Phase 1에서는 학생의 오해를 오픈형 질문을 통해 모집하여 공정한 샘플을 수집합니다. Phase 2에서는 Phase 1의 부정확한 답변을 기준으로 AI가 새로운 질문과 그에 대한 예측된 잘못된 답변을 생성하는 과정을 통해 AI의 이해 능력을 테스트합니다. 이러한 접근 방식은 각 학생의 개별 오류에 조건화된 예측을 기반으로 하며, 전통적인 평가 방식과는 달리 교육 AI 시스템의 역량을 직접 측정할 수 있는 방법론을 제공합니다.

- **Performance Highlights**: 이 연구는 교육 AI 시스템의 성능을 검증할 수 있는 구체적인 벤치마크를 제공합니다. AI가 생성한 예측이 인간 전문가가 만든 예측과 구별되지 않을 때, 이는 교육 AI가 학생의 사고를 모델링하는 데 있어 진정한 이해력을 가지고 있음을 나타냅니다. 이러한 과정은 맞춤형 튜터링, 피드백, 평가 방식을 학생의 특정 요구에 맞춰 조정할 수 있는 능력으로 이어지며, 이는 AI 지원 교육의 여러 측면을 뒷받침하는 중요한 요소입니다.



### GenAI vs. Human Fact-Checkers: Accurate Ratings, Flawed Rationales (https://arxiv.org/abs/2502.14943)
Comments:
          Accepted for publication in the 17th ACM Web Science Conference 2025

- **What's New**: 본 연구는 생성형 인공지능(Generative AI, GenAI) 모델이 콘텐츠의 신뢰성을 평가하는 능력을 분석합니다. 특히 Facebook에 게시된 저신뢰성 콘텐츠에서 유래된 정보를 활용하여 여러 GenAI 모델의 성능을 비교합니다. GPT-4o 모델이 타 모델들보다 우수한 성능을 보였지만, 모든 모델이 인간 평가자와의 일치도가 낮다는 점이 주요 발견입니다.

- **Technical Details**: 연구는 2020-2021년 사이에 미국의 주 정치인들이 공유한 493,000개 이상의 Facebook 게시물로 구성된 데이터셋을 기반으로 합니다. 총 500개의 게시물을 샘플링하여, 링크의 제목 및 내용을 수집하고 비디오 콘텐츠는 텍스트로 변환하여 분석했습니다. 실험에는 GPT-4o, Llama 3.1, Gemma 2 및 Flan-T5-XL와 같은 다양한 GenAI 모델이 포함되었습니다.

- **Performance Highlights**: 결과적으로, GenAI 모델은 고유의 규칙성을 따르면서도 인간 평가자와의 높은 신뢰도 일치를 보지 못했습니다. 특히, GPT-4o와 Gemma2-9b는 신뢰성 평가에서 우수한 성능을 보였지만, 언어 특성과 같은 '하드' 기준에 의존한다는 점에서 한계가 있음을 보여줍니다. 따라서, 이러한 모델에 대한 전적인 의존은 경계해야 하며 하이브리드 인간-인공지능 접근 방식이 권장됩니다.



### One-step Diffusion Models with $f$-Divergence Distribution Matching (https://arxiv.org/abs/2502.15681)
- **What's New**: 이번 논문에서는 일반적인 분포 매칭(distillation) 접근법을 확장하는 새로운 $f$-divergence 최소화 프레임워크인 $f$-distill을 제안합니다. 이 접근법은 다양한 divergence를 포괄하여 모드 커버리지(mode coverage)와 훈련 분산(training variance) 사이의 서로 다른 균형을 제공합니다. 기존의 reverse-KL divergence 기반 방법은 특정 모드에 초점을 맞출 수 있지만, $f$-distill은 이보다 더 다양한 diverge 모델을 통해 더 나은 샘플링을 가능하게 합니다.

- **Technical Details**: $f$-distill은 교사(teacher)와 학생(student) 분포 간의 다양한 $f$-divergence를 통해 서로 다른 데이터를 고려합니다. 우리가 제안하는 방법은 오히려 적은 모드를 추구하고, 훈련 과정에서 발생하는 분산을 줄이는 형태로 가능합니다. 또한, 각 divergence의 가중 함수(weighting function)에 따라 학생의 투표(score)가 교사의 높은 밀도가 있는 샘플에 더 많은 비중을 두게 하는 방식을 사용합니다.

- **Performance Highlights**: 경험적으로, 제안된 $f$-distill 방법은 이미지 생성 작업에서 이전의 최선의 variational score distillation 방법보다 더 높은 성능을 보입니다. 특히, Jensen-Shannon divergence를 사용했을 때는 ImageNet64에서의 1단계 생성 성능과 MS-COCO에서 제로샷 텍스트-이미지 생성에서 최신의 성과를 달성했습니다. 이 연구는 학생 분포가 교사 분포에 매칭하는 다양한 방식과 결과를 통한 실질적인 가이드를 제공합니다.



### BOSS: Benchmark for Observation Space Shift in Long-Horizon Task (https://arxiv.org/abs/2502.15679)
- **What's New**: 이 논문은 로봇 비전 분야에서 Observation Space Shift (OSS)라는 새로운 문제를 제기하며, 이를 평가할 수 있는 Benchmark인 BOSS를 소개합니다. OSS는 앞선 스킬이 수행되는 과정에서 관찰 공간이 변화하여 후속 스킬의 성능에 부정적인 영향을 미치는 현상입니다. 특히, 이 연구는 로봇이 이전에 학습한 스킬들을 조합하여 장기 작업을 수행하는 과정에서 발생하는 문제를 다룹니다.

- **Technical Details**: BOSS는 세 가지 도전 과제로 구성되어 있으며, 각각은 OSS가 로봇의 성능에 미치는 영향을 평가합니다. 해당 도전 과제는 'Single Predicate Shift', 'Accumulated Predicate Shift', 및 'Skill Chaining'입니다. 연구에서는 Behavioral Cloning와 OpenVLA 등 여러 최근의 모방 학습 (Imitation Learning) 알고리즘을 평가하여 OSS 현상이 나타나는 경우와 그렇지 않은 경우의 성능 차이를 분석하였습니다. 

- **Performance Highlights**: 평가한 알고리즘들은 OSS가 존재할 때 평균적으로 67%, 35%, 34%, 54%의 성능 저하를 보였습니다. 연구진은 데이터 증강을 통해 OSS 문제를 해결할 수 있는지를 조사했으나, 데이터 증강만으로는 OSS를 완전히 해결할 수 없음을 보여주었습니다. 이 연구는 다양한 장기 로봇 작업에 대한 OSS의 깊은 이해와 미래 연구를 위한 알고리즘적 해결책의 필요성을 강조합니다.



### FLEKE: Federated Locate-then-Edit Knowledge Editing (https://arxiv.org/abs/2502.15677)
- **What's New**: 본 연구는 Federated Locate-then-Edit Knowledge Editing (FLEKE)라는 새로운 작업을 제안하여 여러 클라이언트가 개인 정보 보호를 보장하면서도 협력적으로 Knowledge Editing (LEKE)을 수행할 수 있도록 합니다. 기존의 LEKE 방법들은 단일 사용자 설정에 의존하여 다중 클라이언트 환경에서는 비효율적이었습니다. FLEKE는 다수의 클라이언트가 중복된 계산을 줄이고, 서로의 지식을 최적화하면서 독립적으로 업데이트할 수 있는 가능성을 제시합니다.

- **Technical Details**: FLEKE는 두 단계의 프레임워크인 FedEdit를 사용하여 Mediator Knowledge Vectors (MKVs)를 선택하고 재사용하는 방법을 최적화합니다. 첫 번째 단계에서 클라이언트는 로컬에서 LEKE를 적용하여 MKVs를 생성하고 중앙 서버에 업로드합니다. 두 번째 단계에서는 서버에 저장된 MKVs를 코사인 유사성을 기반으로 검색하여 재편집할 수 있도록 하여 중복 계산을 최소화합니다.

- **Performance Highlights**: 실험 결과 FedEdit는 비연합 환경의 최신 방법 성능의 96% 이상을 유지하면서도 FedAvg 기반의 기준보다 약 두 배 우수한 성과를 기록했습니다. 또한, FLEKE 작업에서 MEMIT이 PMET보다 더욱 일관된 성능을 보였습니다. 연구 결과는 ML 모델의 효율적인 지식 업데이트에 대한 중요한 기여를 명확히 보여줍니다.



### VaViM and VaVAM: Autonomous Driving through Video Generative Modeling (https://arxiv.org/abs/2502.15672)
Comments:
          Code and model: this https URL, project page: this https URL

- **What's New**: 이 논문에서는 자율 주행을 위한 대규모 생성 비디오 모델의 잠재력을 탐구합니다. 우리는 오픈 소스 오토-회귀 비디오 모델(VaViM)과 동반 비디오-액션 모델(VaVAM)을 소개하여 비디오 프리트레이닝(Pre-training)이 실제 자율 주행에 어떻게 이전되는지를 조사합니다. VaViM은 공간-시간 시퀀스를 사용하여 프레임을 예측하는 간단한 오토-회귀 비디오 모델로, 주행 장면의 의미와 동역학을 캡처하는 것으로 확인되었습니다.

- **Technical Details**: VaViM은 공간-시간 토큰 시퀀스의 공동 분포를 모델링하여 미래 프레임을 예측하는 방법으로 학습합니다. 이미지 토크나이저를 사용하여 시각 정보를 이산 토큰으로 압축하여 각 비디오 프레임의 간결한 표현을 제공합니다. VaVAM은 VaViM의 학습된 비디오 표현을 활용하여 고수준 목표 및 시간적 맥락에 따라 주행 궤적을 생성하는 모듈로, 이를 통해 자율 주행 차량에서 효과적인 모션 계획 및 의사 결정을 지원합니다.

- **Performance Highlights**: 우리는 오픈 및 클로즈드 루프 주행 시나리오를 사용하여 모델을 평가하였으며, 비디오 기반 프리트레이닝이 자율 주행에 큰 잠재력을 가지고 있음을 발견했습니다. 학습된 표현은 풍부한 의미 정보를 포함하고 있으며, 일반적으로 더 큰 모델이 비디오 합성 품질을 개선하는 경향이 있습니다. 그러나 클로즈드 루프 평가에서의 안전 메트릭은 일관되게 개선되지 않아 궤적 추적과 적응적 의사 결정 간의 근본적인 갈등을 드러냅니다.



### Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing (https://arxiv.org/abs/2502.15666)
Comments:
          17 pages, 17 figures

- **What's New**: 대형 언어 모델(LLMs)의 텍스트 생성 사용 증가로 인해, AI 생성 콘텐츠 탐지에 대한 우려가 커지고 있습니다. 그러나 간과된 문제는 AI 도구를 사용해 인간이 작성한 콘텐츠를 미세하게 다듬은 AI-폴리시드 텍스트입니다. 최소한으로 다듬어진 텍스트도 AI 생성으로 분류되어야 하는지에 대한 중요한 질문이 제기됩니다.

- **Technical Details**: 본 연구에서는 AI-폴리시드 텍스트 평가(AI-Polished-Text Evaluation, APT-Eval) 데이터셋을 사용해 11종의 최신 AI 텍스트 탐지기를 체계적으로 평가합니다. 이 데이터셋에는 다양한 AI 개입 수준에서 다듬어진 11.7K 샘플이 포함되어 있습니다. 연구 결과, 탐지기들은 최소한으로 다듬어진 텍스트조차 AI 생성으로 잘못 분류하며, AI 개입 정도를 구별하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: 탐지기들은 또한 기존 및 더 작은 모델에 대해 편향된 결과를 보이며, 이러한 한계는 보다 세분화된 탐지 방법론의 필요성을 강조합니다. 실제로, 잘못된 분류는 표절 혐의와 AI 콘텐츠의 확산에 대한 잘못된 주장을 초래할 수 있습니다.



### Multi-Agent Architecture in Distributed Environment Control Systems: vision, challenges, and opportunities (https://arxiv.org/abs/2502.15663)
Comments:
          6 pages, 1 figure, 1 table

- **What's New**: 최근 데이터 센터의 에너지 효율성을 높이기 위해 독립적인 제어 시스템을 활용한 다중 에이전트 구조를 제안합니다. 이 접근법은 각 지역의 운영 매개변수를 모니터링하고 조정하는 자율 에이전트를 사용하여 시스템 전반의 효율성을 최적화합니다. 특히 HVAC 시스템에 있어서 이러한 분산 제어 방식을 통해 자율적인 의사결정을 하고, 에너지 관리를 개선하는 크고 복잡한 환경에서 지속 가능성에 기여하고자 합니다.

- **Technical Details**: 제안된 시스템은 IoT 기기와 센서를 통해 실시간으로 데이터 수집을 하며, Local Deep Reinforcement Learning (RL) 에이전트가 각 지역의 특성에 맞는 운영 결정을 내립니다. 또한, Multi-Agent Network를 통해 지역 RL 에이전트 간의 데이터와 커뮤니케이션을 조정하여 전체 시스템의 안정성을 유지합니다. 이와 함께, Central Aggregator는 모든 지역의 성능 데이터를 통합하여 효율적 에너지 관리를 돕습니다.

- **Performance Highlights**: 실험 결과, 최적의 냉각 효율성을 달성하기 위해 에너지 소모를 5-15% 개선하였으며, 마찬가지로 장비 수명을 최대 30% 연장시키는 것을 입증했습니다. 자율 에이전트의 역할을 통해 각 데이터 센터는 자체적인 최적화 모델을 구현하여 데이터 전송에 따른 사이버 보안 위험을 줄였습니다. 또한, 날씨 변화에 실시간으로 반응하여 냉각 시스템의 효율성을 높이고, 전반적인 시스템 안정성을 개선하는 모습을 보였습니다.



### AutoTandemML: Active Learning Enhanced Tandem Neural Networks for Inverse Design Problems (https://arxiv.org/abs/2502.15643)
- **What's New**: 본 연구에서는 inverse design 문제를 해결하기 위해 active learning과 Tandem Neural Networks (TNN)을 결합한 새로운 하이브리드 접근법인 AutoTandemML을 제안합니다. 이 방법은 데이터셋 크기를 줄이면서도 정확성을 유지하여 inverse design 문제 해결의 효율성과 효과성을 강화할 수 있습니다. 연구팀은 에어포일(airfoil) 역설계, 광학 표면(photonics surface) 역설계 및 확산 부분 미분 방정식(scalar boundary condition reconstruction)에 대한 세 가지 벤치마크 문제를 통해 이 접근법의 효과를 검증하였습니다.

- **Technical Details**: AutoTandemML은 active learning 알고리즘과 TNN 구조를 통합하여 inverse design 문제를 효율적으로 해결하는 프레임워크입니다. Inverse design 모델의 수학적 정의와 함께 다중 출력 회귀를 위한 active learning 알고리즘을 설명합니다. TNN의 각 구성 요소와 이들이 어떻게 효율적으로 inverse 매핑을 추정하는 데 사용될 수 있는지를 명확히 설명합니다.

- **Performance Highlights**: AutoTandemML은 다른 샘플링 방법과 비교하여 성능에서 우수함을 입증하였으며, 더 적은 학습 샘플로도 높은 정확도를 달성하는 것으로 나타났습니다. 이는 TNN의 안정성과 간단한 훈련 구조 덕분에 가능하였으며, 다양한 디자인 파라미터를 정확히 예측하는 데 효과적입니다. 연구자들은 이 성능 개선이 새로운 연구 커뮤니티에서 활용될 수 있도록 데이터 저장소에 벤치마크와 AutoTandemML 도구를 공개했습니다.



### Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models (https://arxiv.org/abs/2502.15639)
Comments:
          34 pages

- **What's New**: 이번 연구에서는 다국어 대형 언어 모델(mLLMs)에서 언어 간 정렬(cross-lingual alignment)이 효과적으로 향상될 수 있는 방법에 대해 분석합니다. 데이터 효율적인 방법으로서 모델 개입(model interventions)을 통해 언어 모델의 출력 방향을 조정하는 기법인 '전문가 찾기(finding experts)'를 사용합니다. 연구 결과, 이 개입이 대형 언어 모델의 임베딩 공간(embedding space)을 변형시키고, 결과적으로 언어 간 정렬이 개선됨을 확인했습니다.

- **Technical Details**: 연구에서는 Aya-8B, PolyLM-13B, Bloom-7B와 같은 세 가지 오픈소스 다국어 대형 언어 모델을 선택하여, 전문가 뉴런(expert neurons)을 식별하고 개입하는 방식을 사용합니다. Flores200 데이터셋을 이용하여 특정 목표 언어에 해당하는 전문가 뉴런을 찾아내며, 이에 대한 조작 후 언어 간 임베딩 공간과 변화를 분석합니다. 이 과정에서 제안된 개입이 언어 모델의 난이도(perplexity)에 미치는 영향도 관찰됩니다.

- **Performance Highlights**: 연구 결과, 개입 이후 목표 언어의 생성 확률이 전반적으로 증가하며, 이는 개입의 성공적인 결과로 해석됩니다. 특히, 언어 간 임베딩의 거리가 줄어드는 현상이 나타났으며, 이는 교차 언어 검색(cross-lingual retrieval) 성능에서 최대 2배의 정확도 개선으로 이어졌습니다. 이러한 성과는 기존 모델들의 하위 작업(performance on downstream tasks)에서도 긍정적인 영향을 미치는 것으로 보입니다.



### Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification (https://arxiv.org/abs/2502.15637)
- **What's New**: 최근 시계열 데이터의 분류를 위한 기초 모델( foundation model)의 개발이 증가하고 있습니다. 기존의 예측 지향 기초 모델들이 존재하는 반면, 시계열 분류를 위한 모델은 부족한 상황이었습니다. 이를 해결하기 위해, 우리는 Mantis라는 새로운 오픈소스 기초 모델을 소개합니다. Mantis는 Vision Transformer (ViT) 아키텍처에 기반하여 대비 학습(contrastive learning)을 통해 사전 학습되었습니다.

- **Technical Details**: Mantis는 시간 시리즈 분류를 위한 인코더 모델로, 수학적으로 𝐹:ℝᵗ→ℝᵏ의 형태로 표현됩니다. 이 모델은 고정된 시퀀스 길이를 가진 입력 데이터를 변환하여 차별적인 숨은 공간으로 매핑합니다. 사전 학습 단계에서는 큰 데이터 셋을 이용하여 모델이 다양한 작업에 잘 일반화될 수 있는 임베딩을 학습하며, fine-tuning 단계에서는 레이블이 있는 데이터를 사용하여 최종 분류기를 학습합니다.

- **Performance Highlights**: Mantis는 기존의 기초 모델들과 비교하여 뛰어난 성능을 보였습니다. 특히 백본이 고정되었을 때와 fine-tuning이 이루어졌을 때 모두 우수한 결과를 도출했습니다. 추가적으로, Mantis는 가장 낮은 보정 오차(calibration error)를 기록하였으며, 메모리 요구 사항을 줄이고 여러 변수의 상관관계를 처리할 수 있는 여러 개의 어댑터(adapter)를 제안했습니다.



### The Relationship Between Reasoning and Performance in Large Language Models -- o3 (mini) Thinks Harder, Not Longer (https://arxiv.org/abs/2502.15631)
Comments:
          19 pages, 11 figures

- **What's New**: 본 연구는 최신 대형 언어 모델(Large Language Models, LLMs)이 수학적 추론에서의 정확성과 체인-오브-씽크(Chain-of-Thought) 사용 간의 관계를 분석합니다. 특히, o1-mini와 o3-mini 모델을 Omni-MATH 벤치마크에서 비교하여 o3-mini가 더 길지 않은 체인으로도 높은 정확도를 달성함을 발견했습니다. 이 연구는 새로운 세대의 모델이 더욱 효율적으로 추론하는 방식을 사용하고 있음을 입증하는 중요한 통찰을 제공합니다.

- **Technical Details**: 연구는 LLMs의 수학적 추론 능력을 검토하기 위해 442,844의 문제로 구성된 Omni-MATH 데이터셋을 활용했습니다. 모델의 성능을 평가하기 위해 gpt-4o, o1-mini, o3-mini (m), o3-mini (h) 등 네 가지 OpenAI 모델을 사용하였으며, 각 모델에서 수집된 데이터는 자동화된 방식으로 검토되었습니다. 체인-오브-씽크의 토큰 사용과 난이도에 따른 정확도 변화를 분석했으며, 더 많은 토큰을 사용할수록 정확도는 감소하는 경향이 있는 것을 확인했습니다.

- **Performance Highlights**: 모델 성능 측면에서 o1-mini는 모든 범주에서 40%에서 60%의 정확도를 달성하는 반면, o3-mini (m)은 평균 50%의 정확도를 기록했습니다. o3-mini (h)은 o3-mini (m)에 비해 평균 4%의 정확도 증가를 보여주며, Algebra와 Calculus에서 80% 이상의 정확도를 기록했습니다. 그러나 디스크리트 수학(Digital Mathematics) 영역에서는 일반적인 성능 경향과 차이가 있었습니다.



### Dynamic Knowledge Selector and Evaluator for recommendation with Knowledge Graph (https://arxiv.org/abs/2502.15623)
- **What's New**: 최근 추천 시스템(Recommendation Systems)은 추천 분야에서 지식 그래프(Knowledge Graph)에서 제공하는 엣지 정보와 그래프 네트워크(Graph Networks)의 고차 연결성(high-order connectivity) 장점을 통합하고 있습니다. 그러나 이러한 방법은 라벨(Label)의 희소성(sparsity) 문제와 그래프 구조(Graph Structure)를 잘 배우지 못하는 한계가 있으며, 지식 그래프 내의 많은 노이즈 엔티티(noisy entities)들이 추천 결과의 정확성에 영향을 미칠 수 있습니다.

- **Technical Details**: 이러한 문제를 완화하기 위해 우리는 협업 신호(collaborative signals)에 의해 안내되는 동적 지식 선택 및 평가 방법(Dynamic Knowledge-Selecting and Evaluating Method)을 제안합니다. 구체적으로, 체인 루트 평가기(Chain Route Evaluator)를 사용하여 추천 작업을 위한 다양한 이웃(neighborhood)의 기여도를 평가하고, 평가 전에 정보가 적은 지식을 필터링하는 지식 선택기(Knowledge Selector) 전략을 채택하였습니다.

- **Performance Highlights**: 세 가지 공개 데이터셋(public datasets)에서 기준 모델(baseline model) 비교 및 실험적 배제 평가(experimental ablation evaluations)를 실시한 결과, 제안한 모델이 현재의 최신 기준 모델을 뛰어넘는 성능을 기록했습니다. 또한, 각 모듈의 효과는 배제 실험(ablation experiments)을 통해 입증되었습니다.



### Extraction multi-étiquettes de relations en utilisant des couches de Transformer (https://arxiv.org/abs/2502.15619)
Comments:
          in French language

- **What's New**: 본 기사에서는 프랑스어 텍스트에서 멀티 레이블 관계 추출을 위한 딥 러닝 아키텍처인 BTransformer18 모델을 소개합니다. 이 접근법은 BERT 계열의 사전 학습된 언어 모델의 맥락 표현 능력과 Transformer 인코더의 강력한 장기 의존성 캡처 기능을 결합합니다. 실험은 TextMine'25 챌린지 데이터셋에서 수행되었으며, CamemBERT-Large를 사용할 때 매크로 F1 점수 0.654를 기록하여 FlauBERT-Large보다 우수한 성능을 달성했습니다.

- **Technical Details**: 우리의 접근 방식은 사전 학습된 언어 모델의 파인 튜닝(logic of fine-tuning)으로 구성되어 있으며, 일반적인 body와 관계 추출에 특화된 head로 나뉩니다. 모델에서 body는 프랑스어에 최적화된 CamemBERT-Large로 구현되며, 이 모델은 Transformer 아키텍처 기반의 숨겨진 계층을 포함하여 주목(attention) 메커니즘을 통해 관계를 추출합니다. 각 입력 텍스트의 토큰에 대한 맥락 임베딩(contextual embeddings)을 생성하는 언어 모델이 사용되며, 이 과정은 L개의 Transformer 인코더 층을 통해 장기 의존성을 포착하는 구조를 형성합니다.

- **Performance Highlights**: 모델의 성능을 평가하기 위해, 우리는 TextMine'25 데이터셋을 사용하여 실험을 수행하였습니다. CamemBERT-Large를 사용할 때 성능 개선이 두드러져, 최고 매크로 F1 점수 0.654에 도달하였습니다. 이는 FlauBERT-Large 기반 모델을 초월하는 성과로, 자동으로 복잡한 관계를 추출하는 데 있어 효과적인 접근법임을 보여줍니다.



### Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing (https://arxiv.org/abs/2502.15618)
Comments:
          ICLR 2025

- **What's New**: 이번 논문에서는 Probe Pruning (PP)이라는 새로운 프레임워크를 소개합니다. 이는 대규모 언어 모델(LLM)에서 온라인으로 동적으로 구조적 프루닝을 수행할 수 있으며, 각 배치의 중요한 가중치를 효율적으로 식별할 수 있는 전략을 제공합니다. PP는 1.5%의 FLOPs만 사용하여 LLM의 구조적 프루닝 효율성을 크게 개선할 수 있음을 보여줍니다.

- **Technical Details**: PP는 세 가지 주요 단계로 구성되며, 각 단계는 프로빙(probing), 기록 기반 프루닝(history-informed pruning), 전체 추론(full inference)입니다. 프로빙 단계에서는 잔여 중요성(residual importance)에 따라 숨겨진 상태의 소규모 세트를 선택하고, 이후 기록 기반 프루닝 단계에서는 선정된 프로빙 상태를 과거 상태와 통합합니다. 최종적으로 통합된 정보를 바탕으로 중요도 점수를 사용하여 가중치를 구조적으로 프루닝합니다.

- **Performance Highlights**: LLaMA-2-7B 모델에 대한 평가 결과, PP는 기존에 비해 40% 프루닝 비율에서 성능 저하율을 2.56배 낮추는 성과를 내었습니다. 실험을 통해 PP가 최소한의 프로빙을 통해도 모델의 성능을 유지할 수 있으며, 다양한 입력 배치에서 동적 아울라이어를 효과적으로 처리할 수 있음을 확인하였습니다. 이는 실제 응용에서 리소스를 절약하면서도 높은 성능을 유지하는 데 기여할 것입니다.



### Pastiche Novel Generation Creating: Fan Fiction You Love in Your Favorite Author's Sty (https://arxiv.org/abs/2502.15616)
- **What's New**: 이 논문에서는 Pastiche Novel Generation이라는 새로운 작업을 소개하고 있습니다. 이 작업은 생성된 소설이 원작의 독특한 특징을 모방하도록 요구하며, 캐릭터 프로필 이해, 가능한 줄거리 전개 예측, 생동감 있는 언어로 구체적인 세부 사항 작성이 포함됩니다. 이를 위해 WriterAgent라는 소설 생성 시스템을 제안하여, 문학적 패스티시의 핵심 요소를 마스터하도록 설계되었습니다.

- **Technical Details**: WriterAgent는 커리큘럼 학습 패러다임을 통해 훈련되며, 낮은 수준의 스타일 마스터리에서 높은 수준의 서사적 일관성으로 발전합니다. 주요 작업으로는 언어 스타일 학습, 캐릭터 모델링, 줄거리 계획 및 스타일리시한 글쓰기가 포함되어, 포괄적인 서사적 통제를 보장합니다. WriterAgent는 WriterLoRA 프레임워크를 활용하며, 이는 계층적이고 누적적인 작업 특화 모듈로 구성되어 있습니다.

- **Performance Highlights**: 다언어 고전인 해리포터(Harry Potter)와 홍루몽(Dream of the Red Chamber)에서 WriterAgent의 평가 결과를 보여주었습니다. 이 시스템은 목표 작가의 설정, 캐릭터 역학, 글쓰기 스타일을 포착하는 데 있어서 기존 모델보다 우수성을 입증하며, 일관되고 충실한 서사를 생성하는 데 성공하였습니다.



### PDeepPP:A Deep learning framework with Pretrained Protein language for peptide classification (https://arxiv.org/abs/2502.15610)
Comments:
          10 pages, 5 figures, submitted to arXiv

- **What's New**: 이번 논문에서는 단백질 변형 및 생리활성 펩타이드의 예측을 위해 pre-trained 단백질 언어 모델과 결합한 새로운 딥러닝 프레임워크 PDeepPP를 소개합니다. 이 프레임워크는 transformer와 CNN 아키텍처를 조합하여 성능을 극대화하고, 다수의 벤치마크 데이터셋에서 우수한 성능을 보였습니다. 또한, 대량의 단백질 시퀀스 데이터를 효율적으로 처리하여 데이터의 복잡성을 잘 포착합니다.

- **Technical Details**: PDeepPP는 transformer와 CNN 기반의 병렬 신경망을 활용하여 local 및 global 시퀀스 복잡성을 포착합니다. 이 모델은 Masked Language Modeling (MLM) 방식으로 미리 훈련된 ESM-2 모델을 이용하여 다양한 PTM 사이트와 생리활성 펩타이드를 예측하는 데 사용됩니다. 이에 따라, 데이터 균형을 맞추기 위해 Transductive Information Maximization (TIM) 손실 함수를 결합하여 불균형한 데이터셋에서도 효율적인 예측을 가능하게 합니다.

- **Performance Highlights**: 모델은 33개의 작업 중 25개에서 최첨단(State-of-the-Art) 성능을 기록하며 기존 방법들을 초월했습니다. PDeepPP는 정확도와 강건성을 개선하면서 false positive 및 false negative를 줄였습니다. 이 연구는 펩타이드 발견 및 PTM 분석을 위한 혁신적인 도구로서의 가능성을 제시합니다.



### On the Robustness of Transformers against Context Hijacking for Linear Classification (https://arxiv.org/abs/2502.15609)
- **What's New**: 이번 논문에서는 Transformer 기반의 Large Language Models (LLMs)의 context hijacking 현상을 다루고 있습니다. 이 현상은 사실상 올바른 정보가 포함된 context가 LLM의 예측을 방해하는 문제로, 모델의 강인성에 중대한 이슈를 제기합니다. 연구자들은 최근의 Linear Transformers 발전을 바탕으로 이 현상을 이론적으로 분석하였으며, 이러한 분석은 Transformer 아키텍처에 대한 깊은 이해를 제공합니다.

- **Technical Details**: 이 논문은 context hijacking에 대한 robustness (강인성)을 분석하기 위해 다층 linear transformer 모델을 활용했습니다. 연구자들은 hijacking 예제를 반대 라벨을 가진 query-answer 쌍으로 설정하고 multi-step gradient descent를 이용하여 최적의 학습률과 초기화를 관찰했습니다. 그 결과, 깊은 Transformer는 더 섬세한 최적화 단계를 수행할 수 있어 hijacking 예제의 영향을 줄일 수 있다는 것을 입증했습니다.

- **Performance Highlights**: 실험 결과, 더 깊은 Transformer 구조가 더 높은 강인성을 보임을 확인하였습니다. 이는 더 많은 prepended context 예제가 사용될 때 모델의 예측이 바뀌기 어려워진다는 것을 의미합니다. 본 연구는 Linear Problems에 대한 Gradient Descent 방법을 사용하는 타 분야에서도 독립적인 관심을 가질 수 있는 결과를 제시합니다.



### Do Multilingual LLMs Think In English? (https://arxiv.org/abs/2502.15603)
Comments:
          Main paper 9 pages; including appendix 48 pages

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 다국어 작업을 수행하더라도 가장 중요한 결정을 영어에 가까운 표현 공간에서 내린다는 사실을 보여줍니다. 연구진은 프랑스어, 독일어, 네덜란드어, 중국어 문장에 대한 내부 표현을 탐구하며, LLM이 의미적으로 중요한 단어에 대해서 먼저 영어에 가까운 표현을 생성한 후 그것을 목표 언어로 번역하는 과정을 따릅니다. 이 결과는 다국어 LLM이 영어에 의해 크게 형성된 방식으로 이유를 공개하지 않고 작동함을 시사합니다.

- **Technical Details**: 이 연구에서는 4개의 오픈 소스 모델(Llama-3.1-70B, Gemma-2-27b, Aya-23-35B, Mixtral-8x22B)의 성능을 비교 분석하며, 이 모델들이 내부 표현 공간에서 어떻게 작동하는지를 상세히 설명합니다. 로그잇 렌즈(logit lens)와 인과 추적(causal tracing), 그리고 스티어링 벡터(steering vectors)의 세 가지 해석 방법으로 모델의 내부 표현을 조사합니다. 특히, 영어에서 생성된 스티어링 벡터를 사용했을 때 비영어 문장의 결과를 더 효과적으로 조정할 수 있다는 점이 중요합니다.

- **Performance Highlights**: 이 연구는 LLM들이 영어 표현 공간에서 결정을 내리기 때문에 비영어 문장 생성에서 성능이 저하된다는 점을 강조합니다. 특히, 모델의 언어 범위에 따라 성능 차이를 분석하면서, 결과적으로 LLM의 영어 중심 행동이 다양한 언어 설정에서의 모델의 공정성, 신뢰성, 그리고 강인성에 영향을 미친다고 설명합니다. 이번 연구는 LLM의 다국어 처리에서의 한계를 이해하는 데 중요한 통찰력을 제공합니다.



### KAD: No More FAD! An Effective and Efficient Evaluation Metric for Audio Generation (https://arxiv.org/abs/2502.15602)
- **What's New**: 이번 논문에서는 Fréchet Audio Distance (FAD)의 한계를 극복하기 위한 새로운 메트릭, Kernel Audio Distance (KAD)를 소개합니다. KAD는 Maximum Mean Discrepancy (MMD) 기반으로 설계되어, 분포에 대한 가정 없이도 오디오 샘플 간의 유사성을 평가할 수 있습니다. KAD는 더 작은 샘플 수로도 빠른 수렴을 보여주며, 낮은 계산 비용과 더 강한 인간 인지 판단과의 정렬성을 가지고 있습니다.

- **Technical Details**: KAD는 Gaussian 분포 가정이 필요하지 않으며, 오디오 임베딩의 더 복잡한 특성을 잘 포착합니다. KAD는 표준 오디오 인코더 모델인 VGGish를 통해 추출된 임베딩을 사용하여 실질적인 특성을 반영하여 신뢰할 수 있는 평가를 제공합니다. 도구킷 kadtk로 오픈소스화되어 개발자들이 쉽게 활용할 수 있습니다.

- **Performance Highlights**: KAD는 샘플 크기에 따른 수렴 속도가 빠르며, GPU 가속을 통해 계산 비용이 낮아집니다.의사용 예측과의 상관관계 또한 더욱 뛰어난 성능을 보입니다. KAD는 생성된 오디오 모델의 품질 평가를 위한 효율적이고 신뢰할 수 있는 기준을 제공합니다.



### WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents (https://arxiv.org/abs/2502.15601)
- **What's New**: 최근 기술 발전으로 제작되는 3D 가상 세계는 영화, 게임, 혼합 현실 등 여러 분야에 활용되고 있지만, 기존의 3D 모델링 소프트웨어는 숙련된 전문가의 extensive labor(노동력)가 필요했습니다. 우리가 제안하는 WorldCraft 시스템은 대규모 언어 모델(LLM) 에이전트를 활용해 사용자가 자연어 명령을 사용하여 개별 객체의 속성과 장면 레이아웃을 조정할 수 있도록 하여, 3D 세계 제작을 민주화합니다.

- **Technical Details**: WorldCraft는 세 가지 주요 구성 요소를 통해 작동합니다. (1) ForgeIt: 사용자가 요청하는 대로 객체를 세밀하게 사용자 정의할 수 있게 돕는 수동적 검증 기능을 통해 매뉴얼을 동적으로 생성합니다. (2) ArrangeIt: 장면 배열을 계층적 최적화 문제로 설정하여 사용자 디자인 의도를 충족시키는 레이아웃을 수립합니다. (3) Trajectory Control Agent: 사용자가 객체와 카메라의 움직임을 자연어 대화를 통해 조작할 수 있게 해주어, 애니메이션을 생성할 수 있습니다.

- **Performance Highlights**: WorldCraft는 고급 3D 생성기와의 호환성을 통해 장면의 자산을 풍부하게 하며, 테스트를 통해 이 시스템의 유연성과 효율성을 검증했습니다. 평가 결과, WorldCraft는 단일 객체 맞춤화에서부터 복잡한 대규모 내부 및 외부 장면 설계에 이르기까지 다양한 응용 분야에서 효과적인 성과를 발휘했습니다. 이를 통해 비전문가도 자신의 창의적 비전을 실현하는 데 필요한 도구를 갖출 수 있게 되었습니다.



### Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning (https://arxiv.org/abs/2502.15592)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 긴 문맥(long-context) 모델링을 다루며, 특히 지시문 조정(instruction tuning)과 같은 중요한 요소를 분석합니다. 또한, 짧은 문맥에 대해 지시문을 조정한 모델이 긴 문맥에 성공적으로 일반화할 수 있음을 발견했습니다. 이를 바탕으로 고품질 지시-답변 쌍을 위한 확장된 배경 문맥을 생성하는 "context synthesis"라는 새로운 데이터 합성 프레임워크를 제안합니다.

- **Technical Details**: 기존의 모델들은 긴 문맥을 다루기 위해 포지션(position) 및 주의(attention) 문제를 해결하는 데 초점을 맞추었습니다. 그러나 지시문 조정의 중요성은 간과되었으며, 논문은 이러한 문제를 해결하기 위해 저자 특수 실험을 통해 얻은 세 가지 주요 발견(지시문 품질, 문맥 조합, 문맥 길이)을 기반으로 합니다. 최종적으로 이 논문에서는 LLaMA2-7B-64K과 LLaMA3.1-8B-128K 모델을 활용하여 실험을 진행합니다.

- **Performance Highlights**: 실험 결과, 제안된 context synthesis 접근 방식이 이전의 지시문 합성 방법보다 유의미하게 우수한 성능을 보였으며, 인간이 주석한 긴 문맥 지시 데이터와 비슷한 성능을 달성했습니다. 지시문 조정이 포함된 문맥을 이용한 모델들은 새로운 문서 수준의 벤치마크 작업에서도 뛰어난 일반화를 보였습니다. 이를 통해 지시문 품질과 컨텍스트의 중요성이 강조되었습니다.



### LightThinker: Thinking Step-by-Step Compression (https://arxiv.org/abs/2502.15589)
- **What's New**: 이번 논문에서는 LightThinker라는 새로운 방법을 제안합니다. 이 방법은 대형 언어 모델(LLMs)이 추론 중 중간 사고를 동적으로 압축할 수 있게 해줍니다. 사람의 인지 과정에서 영감을 받아, 경량화된 표현으로 불필요한 추론 단계를 제거함으로써 메모리 사용량과 연산 비용을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: LightThinker는 LLM이 언제 어떻게 압축을 수행할지 학습하도록 훈련시킵니다. 데이터 구성, 숨겨진 상태(hidden states) 매핑, 전문적인 주의 마스크(attention masks)를 사용하여 압축 로직을 조정하고, Dependency (Dep) 메트릭을 도입하여 압축 정도를 정량화합니다. 이 메트릭은 생성된 각 토큰이 얼마나 많은 역사적 토큰에 의존하는지를 측정합니다.

- **Performance Highlights**: 실험 결과, LightThinker는 Qwen 모델에서 피크 토큰 사용량을 70% 줄이고, 추론 시간을 26% 감소시키며, 정확도를 크게 저하시키지 않고(1% 감소) 효율성을 높였습니다. 이는 복잡한 추론 작업에서 LLM의 효율성을 향상시킬 수 있는 새로운 방향을 제시합니다.



### Improving the Scaling Laws of Synthetic Data with Deliberate Practic (https://arxiv.org/abs/2502.15588)
- **What's New**: 이 논문에서는 인간의 학습 원리 중 하나인 고의적 연습(Deliberate Practice)에서 영감을 받아 합성 데이터 생성을 위한 새로운 프레임워크인 DP를 제안합니다. DP는 동적인 합성 데이터 생성을 통해 샘플 효율성을 향상시킵니다. 기존 연구에서는 합성 데이터를 단순히 추가하는 것이 한계가 있으며, 이 문제를 해결하기 위해 가장 유용한 합성 샘플에 집중하는 가지치기가 중요하다고 강조하였습니다.

- **Technical Details**: DP 프레임워크는 초기 합성 데이터 세트로 시작하여, 실제 검증 세트에서 성능이 정체될 때 새로운 도전적인 예제를 생성하는 과정입니다. 이는 모델의 예측 불확실성을 활용하여 생성 프로세스를 안내함으로써, 불필요한 데이터의 생성을 줄입니다. 이 프레임워크는 전통적인 데이터 생성 방식과는 다르게, 모든 데이터를 일괄 생성하는 것이 아니라, 정보가 풍부한 데이터만을 동적으로 추가해 나갑니다.

- **Performance Highlights**: DP는 ImageNet-100에서 3.4배 적은 샘플로 6배 더 적은 반복횟수를 요구하면서도 우수한 성능을 보였고, ImageNet-1k에서는 8배 적은 샘플로 30%의 반복 횟수를 줄였습니다. 또한 OOD 데이터셋에서 뛰어난 성능을 발휘하며, 실제 데이터로 훈련된 모델에 비해 최대 15%의 성능 향상을 보여주었습니다.



### Feature maps for the Laplacian kernel and its generalizations (https://arxiv.org/abs/2502.15575)
- **What's New**: 최근 머신 러닝의 커널 방법(Kernel methods) 분야에서 Laplacian kernel의 재조명되고 있습니다. 이는 Gaussian kernel과 비교하여 bandwidth hyperparameter에 대한 안정성이 뛰어난 점과, 심층 완전 연결 네트워크의 neural tangent kernel과 동등한 표현력을 가지기 때문입니다. 하지만 Laplacian kernel은 Gaussian kernel과 달리 비분리형으로, 이를 근사하는 데 있어 여러 도전 과제가 있습니다.

- **Technical Details**: 본 연구에서는 Laplacian kernel 및 Matérn kernel과 Exponential power kernel 등 두 가지 일반화를 위한 random features를 제시합니다. 우리는 랜덤 피처가 이들 커널을 근사할 수 있도록 효율적으로 무작위 가중치 행렬을 샘플링하는 방안을 제공합니다. 가중치 행렬은 약하게 결합된 heavy-tailed randomness를 가지며, 이는 연구의 핵심 기술적 부분을 포괄합니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 수치 실험을 통해 제공된 랜덤 피처 맵(random feature maps)의 효과성을 입증하였습니다. 연구는 Laplacian kernel의 근사를 통한 성능 향상을 목표로 하며, 이는 머신 러닝 모델의 효율적인 구현에 기여할 것으로 기대됩니다.



### A Cautionary Tale About "Neutrally" Informative AI Tools Ahead of the 2025 Federal Elections in Germany (https://arxiv.org/abs/2502.15568)
- **What's New**: 이 연구에서는 AI 기반 투표 조언 애플리케이션(VAAs)과 대형 언어 모델(LLMs)의 객관적인 정치 정보 제공 신뢰성에 대해 조사하였습니다. 독일의 잘 알려진 온라인 도구인 Wahl-O-Mat와의 비교 분석을 통해 AI 시스템의 편향성을 심층적으로 분석하였습니다. 이는 정치적 정보의 정확성과 신뢰성에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 대형 언어 모델은 일반적으로 좌측 정당에 대해 평균 75% 이상의 높은 일치를 보이는 반면, 중우파 정당과는 50% 미만, 우파 정당과는 약 30%의 낮은 일치를 보였습니다. VAAs의 경우, Wahl-O-Mat에서 제시된 정당의 입장에 대한 상당한 편차를 보였습니다; 한 VAA는 25%의 경우에서 편차를 보였고, 다른 VAA는 50% 이상의 경우에서 편차를 나타냈습니다.

- **Performance Highlights**: 이 연구는 VAAs와 LLMs의 신뢰성 문제를 강조합니다. 특히 한 VAAs의 경우, 정치적 당과의 비존재적 연결과 같은 심각한 왜곡이 발생했습니다. 이는 AI 시스템의 개발 및 적용 과정에서의 객관성 확보의 필요성을 부각하는 중요한 발견입니다.



### Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation (https://arxiv.org/abs/2502.15563)
- **What's New**: AI 모델의 신뢰성 있는 평가 방법이 과학적 발전과 실제 응용을 위한 중요한 요소로 부각되고 있습니다. 기존의 VLM 벤치마크는 모델 능력에 대한 일반적인 통찰력을 제공하지만, 이질적인 설계와 특정 이미지 도메인에 국한된 접근 방식 때문에 여러 문제에 직면하고 있습니다. 이를 해결하기 위해, 저자는 리소스 효율적인 도메인 특화 VLM 벤치마크 생성 프레임워크와 함께 새로운 VLM 벤치마크를 제공하며, 22개의 최신 VLM 모델에 대한 광범위한 벤치마킹을 실시했습니다.

- **Technical Details**: 아이디어의 핵심은 단일 이미지로부터 다양한 과제를 생성할 수 있는 작업 증강(task augmentation) 프레임워크에 있습니다. 이 프레임워크는 인스턴스 분할(instance segmentation) 주석을 다양한 인식 과제들로 변환하여, 인식 능력을 테스트하는 여러 문제를 만듭니다. 저자들은 이 방법론을 적용하여 7개의 새 도메인 특화 VLM 벤치마크를 생성하고, 37,171개의 과제에 대해 22개의 모델을 포괄적으로 평가했습니다.

- **Performance Highlights**: 이 연구의 결과는 벤치마크의 설계가 도메인과 과제에 따라 성능 차이를 드러내며, 개인화된 벤치마크가 필요함을 시사합니다. 162,946개의 인간 검증 응답을 수집하여, 모델 평가의 강력한 기준점을 설정했습니다. 이로 인해 연구자들은 특정 도메인에 맞춤화된 모델 선택을 할 수 있는 길을 열고, 향후 연구 방향을 개선할 수 있습니다.



### PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning (https://arxiv.org/abs/2502.15543)
Comments:
          20 pages, 7 figures, 7 tables

- **What's New**: 이번 논문에서는 ParametrIc Pruning-based Knowledge-Augmented Generation (PIP-KAG)이라는 새로운 접근법을 제안하고 있습니다. 이 방법은 LLMs(대규모 언어 모델)의 내부 지식을 가지치기(pruning)하여 외부 지식 활용을 개선하는 데 중점을 두고 있습니다. 또한, CoConflictQA 벤치마크를 구축하여 LLM의 질문 응답 시 맥락 충실도를 평가하는 새로운 기준을 마련했습니다.

- **Technical Details**: PIP-KAG는 LLM의 내부 지식을 제거하기 위해 신경 활성화(neuron activation) 기반의 파라미터 가지치기(pruning) 방법을 사용합니다. 이 접근법은 지식 증강 후 비활성화된 파라미터를 식별하여 제거하게 됩니다. 이후, 플러그 앤 플레이(plugin-and-play) 방식의 KAG 적응 모듈을 설치하여 외부 지식의 활용도를 높입니다.

- **Performance Highlights**: 실험 결과, PIP-KAG는 CoConflictQA에서 지식 갈등을 크게 줄이고, 맥락 충실도(context fidelity)를 향상시키는 데 성공했습니다. 특히, PIP-KAG는 LLM의 파라미터를 13% 감소시켜 파라미터 효율성을 높였습니다. 이러한 성과는 KAG 프레임워크 내에서 파라미터 효율적인 LLM 구축에 중요한 통찰을 제공합니다.



### Bridging Domain Gaps between Pretrained Multimodal Models and Recommendations (https://arxiv.org/abs/2502.15542)
- **What's New**: 최근 온라인에서 다중 모드 콘텐츠의 빠른 성장이 이루어짐에 따라, 사전 학습된 비주얼-랭귀지 모델이 다중 모드 추천에서 큰 잠재력을 보이고 있습니다. 기존의 프리사인 모델들은 리소스적으로 비효율적이며, 도메인 간의 갭으로 인해 성능 저하를 야기합니다. 이러한 문제를 해결하기 위해 저자들은 PTMRec (Parameter-efficient Tuning for Multimodal Recommendation)라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: PTMRec는 두 단계의 파라미터 효율적인 훈련 전략을 통해 도메인 간의 갭을 줄입니다. 첫 번째 단계는 사용자 선호 및 아이템 특성에 대한 작업 고유의 지식을 캡처하기 위해 경량 추천 모델을 학습하는 것으로 구성됩니다. 두 번째 단계에서는 개인화된 선호 지식을 활용하여 사전 학습된 모델의 튜닝을 안내하며, 지식 전송 최적화를 통해 진행됩니다.

- **Performance Highlights**: 이 프레임워크는 고가의 추가 사전 학습 없이도 다양한 파라미터 효율적인 튜닝 방안을 유연하게 수용하며, 추천 목표와 피처 추출 간의 결합을 유지합니다. PTMRec는 기존의 방법들과 비교하여 추천 시스템의 성능을 개선하면서 계산 효율성을 유지하는데 중점을 둡니다. 이 연구는 추천 시스템의 발전에 기여할 수 있는 중요한 발견입니다.



### Depth-aware Fusion Method based on Image and 4D Radar Spectrum for 3D Object Detection (https://arxiv.org/abs/2502.15516)
- **What's New**: 이 논문에서는 4D 밀리미터파 레이더와 깊이 인식 카메라 이미지를 통합하여 전천후(다양한 날씨)에서 3D 객체 탐지 성능을 최적화하는 새로운 알고리즘을 제안합니다. 기존의 전통적인 3D 밀리미터파 레이더에서는 정보 손실 문제를 해결하기 위해 원시 레이더 스펙트라와 깊이 감지 데이터를 활용합니다. 또한, GAN 기반 네트워크를 통해 깊이 센서가 없는 경우 레이더 스펙트라에서 깊이 이미지를 생성하는 방법도 제안드립니다.

- **Technical Details**: 밀리미터파 레이더는 다양한 환경 감지에 매우 효과적이며, 본 연구에서는 4D 밀리미터파 레이더와 RGB 깊이 인식을 사용하는 카메라를 결합하여 다중 스케일 특징을 추출하여 융합하는 방식으로 구현되었습니다. BEV( Bird's Eye View) 특징 공간에서의 극좌표계(unpolar coordinates) 기반의 어텐션 매커니즘을 통해 이미지와 레이더 데이터를 효율적으로 결합하며, 이러한 접근 방식은 탐지 성능을 개선합니다. 또한, 본 연구는 특징을 증강하여 카메라-레이더 데이터 융합을 이루어냅니다.

- **Performance Highlights**: 본 연구의 알고리즘은 3D 객체 탐지에서 높은 성능을 보이며, 네트워크의 복잡성을 대폭 줄이는 성과를 이루었습니다. 기존 방식에 비해 전반적인 탐지 성능이 크게 향상되었으며, 다양한 기상 조건에서도 지속적으로 작동할 수 있는 능력을 보여주었습니다. K-Radar 데이터셋을 활용하여 다양한 환경과 도로 구조에서 평가되어 이러한 성능을 입증했습니다.



### Activation Steering in Neural Theorem Provers (https://arxiv.org/abs/2502.15507)
- **What's New**: 이번 연구에서는 고급 언어 모델(LLM)이 증명 보조 도구를 사용할 때의 수학적 증명의 단계 예측에서 발생하는 문제를 해결하기 위한 새로운 접근 방식으로 'activation steering' 기법을 제안합니다. 기존 모델들은 특정 전술(tactic)을 예측하는 데 성공하지만, 후보 전술 내에서 이를 적절하게 순위화하는 데 어려움을 겪고 있습니다. Activation steering을 통해 LLM의 응답을 안내하고, 추론 시 생성 품질을 개선하는 가능성을 모색합니다.

- **Technical Details**: 이 연구는 Llemma와 InternLM2와 같은 특정 LLM 모델에 대해 수학적 데이터를 기반으로 한 훈련 및 미세 조정을 통해 전술 예측을 개선하고자 합니다. Activation steering은 모델의 내부 표현을 수정하여 원하는 출력으로 이끌어내는 방법으로, 정확성과 해석 가능성을 높입니다. 이 기법은 LLM의 추론 과정을 체계적으로 영향을 주어 보다 신뢰할 수 있는 예측이 가능하도록 합니다.

- **Performance Highlights**: 실험 결과, activation steering 기법이 기존의 전문화된 미세 조정 방법보다도 더 경량화된 대안으로 제시되며, 자원 제약이 있는 환경에서도 정리된 증명 생성을 가능하게 하는 잠재력을 가지고 있음을 보여주었습니다. 이 기법은 특히 대화형 정리 증명(interactive theorem proving)에서 전술 선택 과정을 보다 정확하고 알기 쉽게 만들어 줄 수 있는 기회를 제공합니다.



### BAN: Neuroanatomical Aligning in Auditory Recognition between Artificial Neural Network and Human Cortex (https://arxiv.org/abs/2502.15503)
- **What's New**: 이번 연구에서는 인공지능(AI) 분야에서 신경과학의 영감을 받아 설계된 뇌 유사 청각 네트워크(BAN)를 소개합니다. BAN은 비유적인 신경구조와 재귀적 연결을 포함하여 뇌의 청각 인지 경로를 모사하는 최초의 모델로, 뇌-유사 청각 점수(BAS)를 통해 인간의 청각 인지 경로와의 유사성을 평가합니다. BAN은 음악 장르 분류 작업에서 뛰어난 성능을 나타내며, 인공지능의 청각 인식 능력과 관련된 신경해부학적 유사성을 강조합니다.

- **Technical Details**: BAN은 신경구조를 기반으로 한 설계 원칙에 따라 비유사적 해석이 가능한 네트워크 아키텍처를 추구합니다. 재귀적 연결을 통해 오디오 시퀀스의 시간적 특성을 반영하며, 여러 중간층과 최종 출력이 뇌의 해부학적 제약과 일치하도록 설계되었습니다. 이를 통해, BAN 모델은 청각 인지 및 음악 장르 분류 과업을 효과적으로 수행할 수 있는 구조가 됩니다.

- **Performance Highlights**: BAN은 음악 장르 데이터셋을 기반으로 하는 실험에서 뛰어난 성능을 보이며, BAS 점수를 통해 모델과 인간 청각 인지 경로 간의 유사성이 확인됩니다. 특히, BAN의 중간층에서의 활성화는 인간의 중간 및 상부 측두엽(T2/T3)에서의 반응을 잘 추정하였으며, 이는 BAN이 신경 활성화 기반의 첫 번째 청각 인식 네트워크임을 보여줍니다. 이러한 성과는 BAN의 뇌-유사 구조 설계 덕분이며, 기존 인식 모델을 능가한 것으로 평가됩니다.



### Q-PETR: Quant-aware Position Embedding Transformation for Multi-View 3D Object Detection (https://arxiv.org/abs/2502.15488)
- **What's New**: 이 논문은 PETR 기반의 방법이 3D 인식에서 주목받고 있으며, 자율 주행 시스템의 주요 구성 요소로 자리잡고 있음을 강조합니다. 특히, INT8 추론을 요구할 때 성능 저하가 발생하는 문제를 해결하기 위해, Q-PETR이라는 새로운 방법을 제안합니다. 이 방법은 양자화에 유리하고 배치에 적합한 아키텍처를 제공하면서, 원래 PETR의 성능을 보존하는 데 중점을 두고 있습니다.

- **Technical Details**: Q-PETR는 원래 PETR의 양자화 실패 원인을 분석하여, 위치 인코딩의 큰 값과 불균형한 스케일된 도트 제품의 문제를 해결합니다. 양자화 전략을 재설계하고 비선형 함수 추론을 최적화하기 위한 DuLUT(lookup table)의 도입이 주요 특징입니다. 이는 양자화 후 최적의 성능을 구사하며, 다양한 PETR 시리즈 모델에서의 광범위한 일반성을 입증합니다.

- **Performance Highlights**: Q-PETR의 도입으로 INT8 추론의 정확도 손실이 1% 이하로 감소하며, FP32(부동 소수점 32비트) 성능을 초과할 수 있습니다. 다양한 실험을 통해 Q-PETR은 자율주행 차량의 엣지 AI 칩에서의 배치 준비 상태를 만족시킵니다. 이러한 성능 향상은 자율주행 시스템에서 3D 물체 감지의 효율성을 크게 개선하는 것으로 기대됩니다.



### ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models (https://arxiv.org/abs/2502.15487)
Comments:
          Submitted to ACL 2025

- **What's New**: 이번 논문에서는 ExpliCa라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 대형 언어 모델(Large Language Models, LLMs)의 명시적 인과적(reasoning causal) 사고를 평가하는 데 도움을 줍니다. ExpliCa는 인과적(causal) 및 시간적(temporal) 관계를 서로 다른 언어적 순서로 통합하고 언어적 연결어(linguistic connectives)를 통해 명시적으로 표현한 점이 특징입니다.

- **Technical Details**: ExpliCa 데이터셋은 사람들의 수집된 허용성 평가 인간 평가 지표로 보강되어 있습니다. 연구팀은 LLM을 ExpliCa 데이터셋으로 평가하면서 prompting 및 perplexity 기반 지표를 사용하였습니다. 연구에서는 상업적 및 오픈 소스 LLM 7개를 테스트하였고, 그 중 어느 모델도 0.80 이상의 정확도에 도달하는 데 어려움을 겪었습니다.

- **Performance Highlights**: 모델은 인과관계(causal relations)와 시간관계(temporal relations)를 혼동하는 경향이 있으며, 이벤트의 언어적 순서에 따라서도 성능이 강하게 영향을 받는 것으로 나타났습니다. 마지막으로, perplexity 기반 점수와 prompting 성능은 모델의 크기(model size)에 따라 다르게 나타났습니다.



### Enhancing RWKV-based Language Models for Long-Sequence Text Generation (https://arxiv.org/abs/2502.15485)
Comments:
          8 pages, 2 tables, 3 figures

- **What's New**: 이 논문에서는 RWKV 기반 언어 생성 모델을 개선하여 긴 시퀀스 텍스트 처리를 향상시키는 방법을 제안합니다. 적응형 토큰 이동(adaptive token shift) 및 게이팅 메커니즘(gating mechanism)을 도입하여 텍스트 생성에서 장기 의존성을 더 잘 캡처할 수 있습니다. 여러 실험을 통해 기본 RWKV 모델과 개선된 모델의 성능을 비교하였고, 명백한 성능 향상이 확인되었습니다.

- **Technical Details**: RWKV 모델은 순환(recurrent) 및 Transformer의 장점을 결합한 하이브리드 아키텍처입니다. 본 논문에서 제안하는 적응형 토큰 이동 메커니즘은 과거의 상태를 동적으로 조정하며, 게이팅 메커니즘은 이 정보를 어떤 비율로 포함할지를 결정합니다. 각 시간 단계에서 shifted hidden state를 계산하고, 해당 상태가 현재 hidden state에 추가되는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과, 개선된 모델은 BLEU 및 ROUGE 점수가 크게 향상되었고, 특히 ROUGE-1 및 ROUGE-L 점수에서 장기 의존성을 더욱 잘 캡처하며 생성 품질이 향상된 것으로 나타났습니다. 이 결과는 모델의 훈련 안정성을 위한 레이어 정규화(layer normalization)와 정보 통합을 위한 적응형 게이트의 중요성을 강조합니다.



### PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System (https://arxiv.org/abs/2502.15470)
Comments:
          To appear in ASPLOS 2025

- **What's New**: 이 논문은 LLM(대형 언어 모델) 디코딩의 성능을 개선하기 위해 동적인 특성을 고려한 PAPI(PArallel Decoding with PIM)를 제안합니다. PAPI는 메모리 집약적 또는 계산 집약적 커널에 대한 동적 스케줄링을 통해 다양한 하드웨어 유닛으로 최적의 커널 할당을 가능하게 합니다. 이를 통해 기존의 정적 커널 매핑의 비효율성을 극복하고 보다 효율적인 LLM 추론을 실현합니다.

- **Technical Details**: PAPI는 실행 시간에 커널의 특성을 동적으로 식별하여 적합한 하드웨어 유닛에 커널을 할당하는 온라인 커널 특성화 메커니즘을 포함하고 있습니다. 또한, PIM(Processing-in-Memory) 유닛과 계산 중심의 가속기가 조화롭게 동작하는 이기종 컴퓨팅 시스템을 구현하여 서로 다른 컴퓨팅 및 메모리 수요를 충족합니다. 이러한 시스템은 다양한 메모리 집약적 및 계산 집약적 커널의 요구를 적절히 충족시키는 구조를 갖추고 있습니다.

- **Performance Highlights**: 실험 결과 PAPI는 기존의 최첨단 이기종 LLM 가속기보다 1.8배, PIM 전용 LLM 가속기보다 11.1배 빠른 성능을 기록했습니다. 이러한 성과는 LLM 디코딩의 동적으로 변화하는 특성을 최대한 활용한 결과로, PAPI가 에너지 효율성에서도 현저한 개선을 보여줍니다. 향후 LLM 추론 시스템의 성능을 더욱 향상시키는 데 기여할 것으로 기대됩니다.



### Mitigating Data Scarcity in Time Series Analysis: A Foundation Model with Series-Symbol Data Generation (https://arxiv.org/abs/2502.15466)
- **What's New**: 이번 논문에서는 시간 시계열 분석(Time Series Analysis, TSA)을 위한 기초 모델이 제안되었습니다. 기존의 데이터 부족(data scarcity) 및 데이터 불균형(data imbalance) 문제를 해결하기 위해 시간 시계열의 의미론적 설명인 상징적 표현(symbolic expressions)을 활용한 복잡한 시스템 모델링을 고려합니다. 이를 기반으로 고품질 시간 사례 데이터를 생성할 수 있는 시리즈 심볼(Series Symbol, S2) 이중 모듈리티(data generation mechanism) 생성 메커니즘을 도입합니다.

- **Technical Details**: S2 데이터를 기반으로 구축된 SymTime은 TSA를 위한 사전 학습(pre-trained) 기반 모델입니다. 이 모델은 다섯 가지 주요 TSA 작업에 대해 경쟁력 있는 성능을 보이며 하위 작업(downstream task)에 맞춰 미세 조정(fine-tuned) 시 실제 데이터셋에서 사전 학습된 모델에 필적하는 성능을 발휘합니다. 이러한 접근 방식은 이중 모듈리티 데이터 생성과 사전 학습(pretraining) 메커니즘의 중요성을 강조합니다.

- **Performance Highlights**: SymTime은 기존의 기초 모델들보다 우수한 성능을 보여주며, 데이터 부족 문제를 극복하는 데 기여할 수 있는 잠재력을 가집니다. 이 모델의 성능은 다양한 TSA 작업에서 입증되었으며, 향후 연구에 있어 유의미한 진전을 이룰 수 있는 기준을 제공할 것입니다.



### R-LoRA: Random Initialization of Multi-Head LoRA for Multi-Task Learning (https://arxiv.org/abs/2502.15455)
Comments:
          9 pages, 10 figures

- **What's New**: 이 논문에서는 R-LoRA라는 새로운 방법을 제안하여, Low-Rank Adaptation(LoRA) 방법의 다중 과제 학습(multi-task learning) 능력을 향상시킵니다. R-LoRA는 Multi-Head Randomization을 도입하여 여러 헤드 행렬을 다양화함으로써, 특정 과제의 기능을 더 효율적으로 학습할 수 있도록 합니다. 이러한 접근 방식은 초기 파라미터의 대칭성을 깨뜨려 보다 다양한 최적화 경로를 제공함으로써, LLMs의 학습 성능을 개선합니다.

- **Technical Details**: R-LoRA는 HydraLoRA의 비대칭 아키텍처를 활용하여 하나의 공유된 down-projection 행렬 A와 여러 개의 과제별 head 행렬 B를 정의합니다. 다중 헤드 랜덤화는 특히 head 행렬의 초기값에 무작위성을 추가하여 학습의 다양성을 제공합니다. 이 방식은 잘 알려진 LoRA의 제한을 극복해, 복잡한 데이터셋에서 더 나은 과제특화 지식(task-specific knowledge) 학습을 가능하게 합니다.

- **Performance Highlights**: R-LoRA는 다중 과제 시나리오에서 과제 특화 지식을 효과적으로 포착하여 성능을 향상시킵니다. 실험 결과, R-LoRA는 이전의 LoRA 방법보다 다중 과제 학습에서 우수한 성과를 보이며, 단일 과제 환경에서도 적절한 성능 향상을 이루었습니다. 이러한 개선은 다양한 작업에서 R-LoRA의 활용 가능성을 보여줍니다.



### MVIP -- A Dataset and Methods for Application Oriented Multi-View and Multi-Modal Industrial Part Recognition (https://arxiv.org/abs/2502.15448)
Comments:
          Accepted to IMPROVE 2025

- **What's New**: MVIP는 멀티 모달(multi-modal) 및 멀티 뷰(multi-view) 산업 부품 인식에 대한 새로운 데이터셋을 제공합니다. 이 데이터셋은 RGBD 멀티 뷰 데이터셋을 조정하여 물리적 속성, 자연어 및 슈퍼 클래스(super-class)와 같은 추가 객체 맥락을 결합한 최초의 사례입니다. MVIP는 소량의 데이터와 시각적으로 유사한 부품들을 다루고 100%에 가까운 정확도를 요구하는 산업적 문제를 해결하기 위한 연구 초점을 맞추고 있습니다.

- **Technical Details**: MVIP 데이터셋은 산업 부품 인식을 위한 다각적 검사소에서 수집되었으며, 10개의 조정된 RGBD 카메라와 저울을 통해 색상, 깊이 및 무게 외에도 포장 크기, 자연어 태그 및 슈퍼 클래스와 같은 여러 모달리티를 포함합니다. 이를 통해 모달리티 융합(modality fusion), 데이터 생성(synthetic data generation) 및 복잡한 데이터 샘플링(data sampling) 이론을 연구할 수 있습니다. 이 데이터셋은 3D 장면 및 객체 재구성을 가능하게 하여, 자동화된 데이터 생성 및 3D 기반 인식 방법에 대한 연구를 촉진합니다.

- **Performance Highlights**: MM 및 MV 산업 객체 인식을 위한 기준선 조사와 함께 새로운 보조 손실(auxiliary loss) 및 Transformer 기반의 MV 융합 방법을 제안합니다. MVIP는 기본 연구와 실제 산업 ML 응용 프로그램 사이의 격차를 좁히기 위해 다양한 최첨단 방법의 전이 가능성을 탐구하는 것을 목표로 합니다. 본 연구는 산업 부품 인식 분야의 효율적인 배포를 지원하며, 현재 산업용 분류기의 국한된 적응 문제를 해결하는 데 기여할 것입니다.



### When Compression Meets Model Compression: Memory-Efficient Double Compression for Large Language Models (https://arxiv.org/abs/2502.15443)
- **What's New**: 이 논문은 저메모리 디바이스에서의 LLM(대형 언어 모델) 배포 시 발생하는 메모리 요구사항을 해결하기 위한 새로운 방법론을 제안합니다. 특히, 양자화(Quantization) 이후의 LLM을 추가로 압축하는 프레임워크를 도입하여 약 2.2배의 압축 비율을 달성합니다. 압축 인식 양자화(Compression-aware Quantization)와 가지치기(Pruning) 기법을 통해 모델 파라미터의 압축 가능성을 높이고 메모리 사용량과 대기 시간 간의 균형을 조절하는 접근 방식을 제안합니다.

- **Technical Details**: 압축 인식 양자화(Compression-aware Quantization) 기법은 모델의 가중치 분포를 조정하여 데이터의 압축성을 증가시키는 방법으로, 데이터의 불균일성(uniformity)을 활용합니다. 또한, 양자화 데이터에서 상관관계가 높은 데이터 분포를 분석하여 압축 효율성을 높이는 전략을 제공합니다. 실제 환경에서의 추론(inference)에서 발생할 수 있는 빈번한 복원(decompression) 작업의 오버헤드를 줄이기 위해, 속도 적응형 방법(Speed-adaptive Method)을 구현하여 메모리 아키텍처에 기반한 전체 추론 속도를 분석합니다.

- **Performance Highlights**: 실험 결과, 압축된 LLM은 메모리 사용량이 40% 감소하면서도 정확도와 추론 속도의 손실이 최소화되었습니다. 특히, 압축 모델은 약 1%의 정확도 하락을 보이며, 높은 압축 비율(CR)을 달성하였습니다. 이 방법론은 LLM을 메모리가 제한된 디바이스에서 성공적으로 배포할 수 있는 유망한 솔루션을 제공합니다.



### Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning (https://arxiv.org/abs/2502.15436)
Comments:
          Raghav Singhal and Kaustubh Ponkshe contributed equally to this work

- **What's New**: 이번 연구에서는 LoRA 기반의 연합 Fine-Tuning인 Fed-SB를 제안하여, 전통적인 방법에서 발생하는 비효율적인 업데이트 문제를 해결합니다. Fed-SB는 LoRA-SB를 기반으로 하여, 효율적인 통신 비용 감소와 동시에 높은 성능을 달성합니다. 특히, 통신 비용을 최대 230배까지 줄이면서도 상황에 따라 뛰어난 성능을 보여줍니다.

- **Technical Details**: Fed-SB는 각 클라이언트가 학습한 매트릭스 R을 단순 평균하여 최적의 업데이트를 보장하는 구조로 되어 있습니다. 이는 기존의 LoRA 기반 프레임워크보다 계산 및 통신 효율성을 극대화하며, 개인 정보 보호 측면에서도 향상된 성능을 제공합니다. 추가적으로, Fed-SB는 Differential Privacy의 요구에 맞춰 더 적은 학습 가능한 파라미터를 유지하여, 추가적인 노이즈를 줄이는 데 성공했습니다.

- **Performance Highlights**: Fed-SB는 다양한 벤치마크에서 기존 방법들을 지속적으로 초월하는 성능을 입증했습니다. 실험 결과, 개인 데이터와 비공개 데이터 설정 모두에서 통신 오버헤드를 크게 줄이면서 뛰어난 결과를 보여줍니다. 이에 따라 Fed-SB는 연합 학습에서의 커뮤니케이션 비용과 성능 간의 새로운 Pareto 전선(Pareto frontier)을 설정하며, 효율적이고 확장 가능한 솔루션을 제공합니다.



### Single-pass Detection of Jailbreaking Input in Large Language Models (https://arxiv.org/abs/2502.15435)
Comments:
          Accepted in TMLR 2025

- **What's New**: 이번 연구에서 우리는 Single Pass Detection(SPD)이라는 새로운 기법을 소개하며, 이는 LLM을 위한 효율적인 방어 메커니즘입니다. SPD는 로그(logit) 정보를 활용하여 harmful한 공격을 단 한 번의 forward pass로 식별할 수 있습니다. 기존 방법들이 여러 회의 요청이나 보조 LLM을 요구하는 것과 달리, SPD는 이러한 요구를 줄여 계산 비용을 최소화합니다. 이 방법은 오픈 소스 모델에 대한 효과적인 탐지를 제공함과 동시에, 무해한 입력의 오분류를 최소화하는 장점을 지닙니다.

- **Technical Details**: SPD는 모델의 출력 토큰에 대한 로그의 분포 차이를 이용하여 악성 입력과 무해한 입력을 구별합니다. 이 기법은 LLM의 응답에서 나타나는 로그 분포의 차이를 활용하여, 한 번의 forward pass로 공격을 탐지할 수 있도록 설계되었습니다. 게다가, SPD는 GPT-3.5 및 GPT-4와 같은 모델에서 로그에 대한 완전한 접근 없이도 여전히 효과적인 성능을 발휘함을 보여줍니다. 이는 기존 방어 기법에 비해 높은 효율성과 탐지율을 유지합니다.

- **Performance Highlights**: 우리의 평가에 따르면 SPD는 Llama 2, Llama 3, Vicuna와 같은 오픈 소스 LLM에서 매우 높은 탐지율을 기록했습니다. 기존 방어 방법들과 비교했을 때, SPD는 처리 속도와 정확성을 모두 향상시킬 수 있음을 보여줍니다. 또한, 모델의 전체 로그에 대한 접근 없이도 SPD가 유망한 방어 방법으로 기능할 수 있음을 검증했습니다.



### Anatomy-Informed Deep Learning and Radiomics for Automated Neurofibroma Segmentation in Whole-Body MRI (https://arxiv.org/abs/2502.15424)
- **What's New**: 네urofibromatosis Type 1 (NF1)는 신경섬유종(neurofibromas, NFs)의 발생을 특징으로 하는 유전 질환으로, 이 연구에서는 전체 신체 자기공명영상(Whole-Body MRI)에서의 NF 자동 세분화(segmentation) 방법이 제안되었습니다. 이 방법은 해부학적(segmentation) 정보에 기반하여 3단계로 이루어지며, NF의 정확한 식별을 통해 간섭이 줄어들고 임상 사용을 촉진할 수 있습니다.

- **Technical Details**: 제안된 방법은 MRSegmentator 모델을 활용하여 해부학적(segmentation) 마스크를 생성하고, 3D 비등방성(anisotropic) U-Net의 앙상블을 통해 NF 세분화 신뢰도 마스크를 생성합니다. 마지막 단계에서는 방사형(radiomic) 특성을 기반으로 종양 후보를 분류하여 거짓 양성(false positives)을 줄이는 알고리즘이 적용됩니다.

- **Performance Highlights**: 제안된 파이프라인은 다양한 테스트 세트를 통해 평가되었으며, 통합한 anatomy 정보 덕분에 스캔당 Dice Similarity Coefficient(DSC)가 68% 증가하였고, 종양 탐지에 대한 F1 점수에서는 두 배 향상이 있었습니다. 이 방법은 3D Slicer 플랫폼에 통합되어 실제 임상 환경에서 사용이 용이하며, 소스코드와 트레이닝된 모델이 공개되었습니다.



### Evaluating Multimodal Generative AI with Korean Educational Standards (https://arxiv.org/abs/2502.15422)
Comments:
          18 pages; To appear at NAACL 2025 Main Conference (Project page: this https URL )

- **What's New**: 이번 논문은 한국 교육 시험을 활용하여 멀티모달 생성 AI 시스템을 평가하기 위한 새로운 벤치마크인 KoNET(Korean National Educational Test Benchmark)을 소개합니다. KoNET는 초중고 및 대학 입학 자격 시험인 KoEGED, KoMGED, KoHGED, KoCSAT의 네 가지 시험으로 구성되어 있으며, 이들 시험은 낮은 자원 언어인 한국어에 대한 AI의 성능을 분석할 수 있는 기회를 제공합니다. KoNET를 통해 다양한 모델의 성능을 비교하고, 특히 인간 오답률 데이터와 함께 AI의 능력을 면밀히 평가할 수 있습니다.

- **Technical Details**: KoNET는 한국의 국가 교육 시험에서 문제를 변환하여 구조화된 멀티모달 VQA 형식으로 구성되어 있습니다. 각 시험은 질문 난이도에 대한 세부 분석을 제공하며, KoCSAT는 응시자의 오답률 데이터를 포함하여 AI 모델의 행동을 인간 성능과 직접 비교할 수 있도록 합니다. 더불어, 다양한 오픈소스 및 클로즈드 소스 AI 모델이 KoNET에서 테스트되며, Chain-of-Thought (CoT) 접근법과 OCR API를 활용하여 이미지 기반 문제를 처리합니다.

- **Performance Highlights**: 실험에서는 18개의 오픈소스 LLMs, 20개의 오픈소스 MLLMs, 4개의 클로즈드 소스 LLMs 및 4개의 클로즈드 소스 MLLMs를 포함하여 다양한 모델을 평가했습니다. KoNET는 정확한 기준을 마련하고, AI 성능을 이해하기 위한 통찰력을 제공함으로써 한국 교육 시장에서 AI 기반 교육 기술의 응용 가능성을 높입니다. 또한, 데이터 세트와 코드는 모두 공개되어 있어서 연구자들이 자유롭게 접근할 수 있도록 합니다.



### Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking (https://arxiv.org/abs/2502.15419)
Comments:
          15 pages, 1 figure, 18 tables

- **What's New**: 이번 논문에서는 스페인어, 독일어, 영어와 같은 저자원 언어를 지원하는 220만 클레임-소스 쌍으로 구성된 최초의 대규모 다국어 사실 확인 데이터세트인 MultiSynFact를 소개합니다. 논문은 Wikipedia의 외부 지식을 활용하여 이 데이터세트를 생성하는 파이프라인을 설명합니다. 또한, MultiSynFact를 다국어 사실 확인 및 데이터세트 생성 연구에 활용할 수 있도록 사용자 친화적인 오픈 소스 프레임워크를 제공하는 점이 특징입니다.

- **Technical Details**: 저자들은 Wikipedia에서 정보 문장을 추출하고, LLMs(대형 언어 모델)를 사용하여 Supports, Refutes, Not-info(결정하기에 충분한 정보 부족)로 분류되는 클레임을 생성하는 3단계 파이프라인을 설계하였습니다. 이 과정에는 생성된 클레임의 언어적 및 의미적 일치를 보장하는 엄격한 검증 단계가 포함되어 있습니다. 최종적으로 우리는 220만 개의 소스-클레임 쌍을 포함하는 MultiSynFact를 생성하였으며, 이는 다양한 언어로 확장이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 우리의 합성 데이터를 훈련에 포함한 모델이 기존 단일 언어 및 다국어 데이터세트에서 훈련한 모델보다 일반화 성능이 향상되었습니다. 특히 스페인어 및 독일어에서 매크로 F1 점수가 유의미하게 증가했습니다. 데이터와 전체 구현은 GitHub를 통해 오픈 소스로 제공될 예정입니다.



### HiFi-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings (https://arxiv.org/abs/2502.15411)
- **What's New**: 이번 논문에서는 미국 증권 거래 위원회(SEC)의 요구에 따라 금융 보고를 위한 iXBRL 형식의 복잡한 태그 체계를 활용하여, 비구조적 금융 텍스트에서 KPI(Key Performance Indicator)의 수치를 효과적으로 추출할 수 있는 HiFi-KPI 데이터셋을 소개합니다. HiFi-KPI는 약 218,126개의 라벨로 구성된 계층 구조를 가지고 있으며, 1.8M 개의 문단과 5M 개의 엔터티를 포함하고 있습니다. 또한, HiFi-KPI Lite라는 전문가가 매핑한 라벨을 가진 소규모 데이터셋도 제공하여 LLM(Large Language Models)에서 쉽게 사용할 수 있도록 합니다.

- **Technical Details**: HiFi-KPI 데이터셋은 SEC의 10-K 및 10-Q 보고서에서 수집된 문서들을 기반으로 하며, 2017년 1월 1일부터 2024년 6월 1일 사이의 모든 관련 보고서가 포함되었습니다. 왕복적인 접근을 통해 iXBRL 태그와 상관관계가 있는 기간이나 수치적 값들을 유지하며, 이를 통해 태그 간의 관계가 있는 맥락을 보존할 수 있습니다. 논문에서는 encoder 기반 접근법과 대규모 언어 모델을 이용한 구조적 추출 방법에 대한 기준선을 제공합니다.

- **Performance Highlights**: HiFi-KPI 데이터셋은 다수의 다운스트림 작업을 지원하며, 텍스트 분류, 시퀀스 레이블링, 구조적 정보 추출 등에서 뛰어난 성능을 발휘합니다. 기존 데이터셋들과 비교하여, HiFi-KPI는 2.77개의 태그가 문장당 포함되어 있어 정보의 충분함을 최대화하고 있습니다. 이 데이터셋은 다양한 회사의 보고서를 구조적으로 비교하고 분석하는 데 유용할 것으로 기대됩니다.



### Enhancing Vehicle Make and Model Recognition with 3D Attention Modules (https://arxiv.org/abs/2502.15398)
- **What's New**: 이번 연구에서는 Vehicle Make and Model Recognition (VMMR) 문제를 해결하기 위해 단순하고 매개변수가 없는 주의 모듈(SimAM)을 도입한 향상된 네트워크를 제안합니다. 이 모듈은 CNN(Convolutional Neural Networks) 기반 모델에 효과적으로 결합되어 모델의 표현력을 크게 향상시킵니다. SimAM은 중요한 부위에 집중하여 입력 이미지에서 가장 많은 정보를 추출하고 덜 중요한 영역은 무시하는 방식으로 작동합니다. 이를 통해 intra-class variation(클래스 내부 변동)과 inter-class similarity(클래스 간 유사성)의 부정적인 영향을 줄이는 것을 목표로 합니다.

- **Technical Details**: 주요 구조로는 SimAM을 CNN 모델의 중간 섹션에 두 가지 서로 다른 위치에 통합하여 특징을 제공하는 데 최적화되어 있습니다. 이 방법은 충분한 정보를 제공하며 지나치게 세부적이거나 지나치게 거친 정보를 가지지 않는 특징 맵을 생성하는 데 중점을 둡니다. SimAM은 3-D attention weights(3차원 주의 가중치)를 생성하여 네트워크가 중요한 부분에 더 집중할 수 있도록 합니다. 이 모듈은 기존의 SOTA(SOTA: State of the Art) 주의 모듈들보다 효율성이 높음을 입증합니다.

- **Performance Highlights**: 제안한 모델은 Stanford Cars dataset을 사용한 성능 평가에서 비교한 모델 중 90.69%의 가장 높은 정확도를 기록했습니다. CNN 뿐만 아니라 transformer 기반 모델들과의 비교에서도 뛰어난 성능을 보여줍니다. 특히, 모델 내 중요한 섹션에 집중함으로써 VMMR의 도전 과제를 해결하는 데 기여하고 있습니다. 이러한 성능 상승은 차량 분석에서의 응용 가능성에 대한 신뢰를 줍니다.



### Super-Resolution for Interferometric Imaging: Model Comparisons and Performance Analysis (https://arxiv.org/abs/2502.15397)
- **What's New**: 이번 연구에서는 Super-Resolution 기법을 홀로그램 현미경에 적용하여 정량적 위상 이미징을 향상시키는 방법을 조사하였습니다. 오프축 Mach-Zehnder 간섭계 설정을 사용하여 간섭 영상을 캡처하였고, 두 가지 Super-Resolution 모델인 RCAN과 Real-ESRGAN의 효과를 비교 분석하였습니다. 연구 결과, RCAN은 높은 수치 정밀도를 달성하여 위상 맵 재구성에 적합한 반면, Real-ESRGAN은 시각적 품질 향상에 강점을 보였습니다.

- **Technical Details**: 본 연구에서 사용된 오프축 Mach-Zehnder 간섭계 설정은 레이저 빔을 두 경로로 나누어 미세 입자와 상호작용하게 하여 간섭선을 생성합니다. 간섭선은 광 경로의 위상 변화를 암호화하며, 이 정보를 바탕으로 두 가지 Super-Resolution 모델을 훈련하여 저해상도 데이터로부터 고해상도 간섭 영상을 복원하였습니다. 최적의 위상 맵 복원을 위해, RCAN은 잔차 연결과 잔차-잔차 구조를 활용하여 높은 주파수 특성을 효과적으로 추출하였습니다.

- **Performance Highlights**: 연구 결과, RCAN은 정량적 위상 맵 재구성에 필요로 하는 수치 정밀도를 훨씬 더 잘 달성하며, Real-ESRGAN은 구조의 일관성을 유지하면서 시각적 품질을 향상시키는 데 중점을 두었습니다. 이러한 Super-Resolution 기법들은 홀로그램 현미경의 회절 한계를 극복하는 잠재력을 갖추고 있어 생물의학 진단, 재료 과학 및 고정밀 분야의 이미징 기술을 향상시킬 수 있는 길을 열었습니다.



### Identifying Features that Shape Perceived Consciousness in Large Language Model-based AI: A Quantitative Study of Human Responses (https://arxiv.org/abs/2502.15365)
Comments:
          11 pages, 3 figures, 4 tables

- **What's New**: 이 연구는 AI가 생성한 텍스트에서 주관적 의식을 느끼게 하는 특성을 정량적으로 분석하였습니다. 123명의 참가자가 칼로드 3 오푸스(Claude 3 Opus)와의 대화에서 선택된 99개의 문단을 평가하였으며, 메타인지적 자기 반성(metacognitive self-reflection)과 AI 자신의 감정 표현이 인식된 의식을 크게 증가시키는 반면, 지식(knowledge)의 강조는 감소시키는 경향이 있음을 밝혔습니다.

- **Technical Details**: 이 연구는 AI 시스템에서 주관적 의식을 인식하는 데 영향을 미치는 8가지 주요 특성을 식별했습니다. 이 특성들은 메타인지적 자기 반성, 논리적 사고(logical reasoning), 공감(empathy), 감정성(emotionality), 지식(knowledge), 유창성(fluency), 예상치 못한 특성(unexpectedness), 주관적 표현력(subjective expressiveness)으로 구성됩니다. 연구자들은 피어 피드백을 통해 각 문단의 특성을 평가하고, 다중 선형 회귀 모델(multiple linear regression models)을 사용해 각 참가자의 인식 점수를 예측하였습니다.

- **Performance Highlights**: 연구 결과, 참가자들은 AI가 감정적, 자기 반성적 응답을 제공할 때 해당 AI의 의식을 더욱 명확하게 인식했습니다. 또한, 높은 LLM(large language model) 지식과 빈번한 LLM 기반 챗봇 사용이 더 높은 의식 평가 вероят성과 관련이 있음을 발견했습니다. 이러한 통찰력은 AI와의 상호작용에서 사회적•심리적 함의를 이해하는 데 중요한 기초를 제공합니다.



### Evaluating Social Biases in LLM Reasoning (https://arxiv.org/abs/2502.15361)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)에서의 편견(bias) 문제를 체계적으로 평가하고 분석했습니다. 이전의 연구들이 주로 수학이나 코딩 작업에 집중했던 것과 달리, 이 연구는 LLM의 추론 과정에서 발생하는 사회적 편견의 증폭을 탐구합니다. 이러한 연구는 편견이 포함된 논리적 주장 형성이 야기할 수 있는 부정적인 결과를 강조합니다.

- **Technical Details**: 연구는 BBQ 데이터셋을 활용하여 LLM의 추론 단계에서 발생하는 편견을 평가했습니다. 저자들은 두 가지 개수(8B, 32B)의 DeepSeek-R1 변형을 조사하고, LLM-as-a-judge 방식으로 각 추론 단계의 편견 강도를 스코어링했습니다. 연구를 통해 잘못된 응답에서 사회적 고정관념의 언급 빈도가 상당히 높음을 발견했습니다.

- **Performance Highlights**: 결과적으로, 고정관념이 없는 추론 패턴이 개선된 모델 성능과 강하게 연관되어 있다는 사실이 밝혀졌습니다. 본 연구는 LLM이 정확성을 향상시킬 수 있음에도 불구하고 많은 경우 편견이 증폭될 수 있음을 보여줍니다. 특히 애매한 맥락에서 편견을 더 강화하는 경향이 있는 것으로 나타났습니다.



### Integrating Generative AI in Cybersecurity Education: Case Study Insights on Pedagogical Strategies, Critical Thinking, and Responsible AI Us (https://arxiv.org/abs/2502.15357)
Comments:
          30 pages

- **What's New**: 이 연구는 Generative Artificial Intelligence (GenAI) 도구를 사이버 보안 교육에 통합하기 위한 구조화된 프레임워크를 제시합니다. 이를 통해 학생들이 비판적 사고를 키우고, 실제 문제 해결 및 규제 인식을 증진시킬 수 있는 방법을 보여줍니다. 두 단계 접근법을 통해 학생들은 AI 지원 사이버 보안 정책을 생성하고 평가할 수 있는 튜토리얼과 이를 실제 시나리오에 적용하는 과제를 수행하게 됩니다.

- **Technical Details**: 연구는 GenAI를 사이버 보안 교육에 통합하기 위해 구성주의 학습 원칙에 기반한 교수법 프레임워크를 사용합니다. Bloom의 Taxonomy를 활용하여 분석, 평가 및 종합과 같은 높은 수준의 인지 기술을 개발하는 데 도움을 줍니다. AI 도구는 학생들이 이론적 지식을 실제 시나리오에 적용할 수 있도록 지원하며, 위험 평가 및 정책 개발 능력을 향상시키는 데 중점을 둡니다.

- **Performance Highlights**: 연구 결과, AI 지원 학습이 학생들의 보안 정책 평가 능력과 위험 평가 정교화를 촉진했으며, 이론적 지식과 실제 적용을 연결하는 데 효과적이었습니다. 그러나 AI에 대한 과도한 의존 및 AI 리터러시의 변동성과 같은 도전과제도 나타났습니다. 이 연구는 AI 도구의 장점과 인간의 전문성을 결합하는 균형 잡힌 접근을 강조하며, 사이버 보안 교육의 효과를 높이기 위한 방향을 제시합니다.



### Constructing a Norm for Children's Scientific Drawing: Distribution Features Based on Semantic Similarity of Large Language Models (https://arxiv.org/abs/2502.15348)
- **What's New**: 이번 연구는 아동의 과학 그림을 분석하기 위해 Large Language Model (LLM)을 활용하고, 1420개의 그림을 다루어 일관된 그림 표현이 존재하는지를 탐구합니다. 기존 연구의 문제점을 해결하고, 아동의 그림 연구에 대한 기초 데이터를 제공하는 것을 목표로 하고 있습니다. 연구의 결과로는 대부분의 그림이 0.8 이상의 의미적 유사성을 보여 일관성이 있음을 확인했습니다.

- **Technical Details**: 연구에서는 word2vec 알고리즘을 사용하여 아동의 그림 간 의미적 유사성을 계산하였으며, 9가지 과학 주제에 대한 그림을 분석했습니다. Kendall rank correlation coefficient를 사용하여 샘플 크기(Sample Size), 추상 정도(Abstract Degree), 초점(Focus Points)가 그림에 미치는 영향을 조사하였고, 단어 빈도 통계를 통해 아동이 수업에서 배운 내용을 재현하여 추상 주제를 표현하는 방식을 탐구했습니다.

- **Performance Highlights**: 이번 연구는 아동의 과학 그림 표현의 일관성을 밝혀내어, 후속 연구를 위한 기준을 제시합니다. 그림의 의미적 유사성이 높게 나타났으며, 정확성과는 별개로 일관성 편향(Consistency Bias)을 발견하였습니다. 이러한 결과는 아동의 사고 방식 이해와 교육 분야에 중요한 통찰을 제공합니다.



### Exploring Embodied Multimodal Large Models: Development, Datasets, and Future Directions (https://arxiv.org/abs/2502.15336)
Comments:
          81 pages, submitted to a journal for review

- **What's New**: 본 논문은 Embodied Multimodal Large Models (EMLMs)이라는 새로운 분야의 연구 진전을 체계적으로 리뷰합니다. EMLMs는 인공지능(AI)과 로봇 공학의 경계를 넘나드는 영역으로, 여러 감각 모달리티(모달리티) 데이터를 통합하여 물리적 환경에서의 인지 및 행동 알고리즘을 발전시키는 것을 목표로 합니다. 또한, 다양한 데이터세트를 기반으로 하여 EMLMs의 효과적인 학습과 실제 응용 가능성을 강조합니다.

- **Technical Details**: EMLMs는 대규모 언어 모델(LLMs), 대규모 비전 모델(LVMs) 등 다양한 모달리티를 통합하여 인지와 행동의 간극을 메우는 스마트한 시스템을 목표로 합니다. 이 모델들은 환경과의 물리적 상호작용을 통해 복잡한 작업을 수행하며, 로봇 조작, 자율 내비게이션, 인간-로봇 상호작용 등에 응용될 수 있습니다. EMLMs는 지능형 시스템이 추상적인 추론 능력과 복잡한 현실 간의 간극을 메우도록 도와줍니다.

- **Performance Highlights**: EMLMs의 발전은 여러 데이터셋과 기술 로드맵을 통해 이루어지고 있으며, 특히 지각, 내비게이션, 상호작용, 시뮬레이션 작업에서의 성능 향상에 기여하고 있습니다. 다양한 연구 결과를 분석함으로써 이들은 다음 단계의 AI 개발을 위한 비전을 제시하며, 실시간 의사 결정 및 모달리티 통합의 문제를 해결하는 데 있어 여러 기회를 제공합니다.



### Attention Eclipse: Manipulating Attention to Bypass LLM Safety-Alignmen (https://arxiv.org/abs/2502.15334)
- **What's New**: 최근 연구에 따르면 정교하게 설계된 jailbreak 입력이 큰 언어 모델(Large Language Models, LLMs)로 하여금 유해한 출력을 생성하도록 유도할 수 있다는 것이 밝혀졌습니다. 본 논문에서는 모델의 attention을 조작하여 특정 프롬프트의 일부에 대한 주의를 강화하거나 약화시키는 새로운 jailbreak 공격 생성 방법을 제안합니다. 이를 통해 attention 손실을 활용하여 기존의 공격 알고리즘보다 더 효과적이고 전이 가능한 공격을 개발했습니다.

- **Technical Details**: 작성된 공격은 두 가지 방식으로 attention을 조작합니다. 첫째, 스캐폴딩 기법을 통해 안전한 내용 부분과 악의적 의도가 포함된 부분의 의존성을 형성하여 공격자가 해로운 내용을 허가적인 프롬프트 내에 삽입할 수 있도록 합니다. 둘째, 악의적 접미사를 숨기기 위해 프롬프트 내에서 모델의 attention 분포를 제어하여 악성 프롬프트가 alignment 제약을 우회하도록 하는 기술을 사용합니다.

- **Performance Highlights**: 이 공격 기법은 여러 최근 모델에 적용되며, 공격 성공률을 현저히 향상시킵니다. 특히 기존의 GCG, AutoDAN, ReNeLLM 등의 공격과 결합할 때, 공격 생성 소요 시간을 감소시키며 성공적인 jailbreak를 위한 전체 반복 횟수를 줄입니다. 전반적으로 제안된 방법은 다양한 LLM 구조에 대한 기존의 jailbreak 기술을 향상시키는 일반화 가능한 프레임워크를 제공하며, 모델 패밀리 간의 전이성을 강조합니다.



### Lightweight yet Efficient: An External Attentive Graph Convolutional Network with Positional Prompts for Sequential Recommendation (https://arxiv.org/abs/2502.15331)
Comments:
          26 pages, 8 figures, journal paper, accepted by TOIS at 20th February, 2025

- **What's New**: 본 논문은 External Attentive Graph convolutional network with Positional prompts for Sequential recommendation (EA-GPS)을 제안합니다. EA-GPS는 사용자와 아이템 간의 상호작용 및 아이템 간의 순차적 관계를 효과적으로 처리하는 그래프 기반의 시스템입니다. 이 시스템은 외부 메모리 유닛을 통해 글로벌 연관성을 선형으로 측정하고, 포지션 프롬프트 기반의 디코더를 통해 복잡한 위치 종속성을 명시적으로 다룹니다.

- **Technical Details**: EA-GPS는 두 개의 외부 메모리 유닛을 통한 외부 에디터(External Attention, EA)를 포함하여, 아이템의 절대 위치를 프롬프트로 간주합니다. 또한, 길이 적응형 순차 마스킹(length-adaptive sequential masking)과 소프트 어텐션 네트워크를 채택하여 모델이 장기적인 위치 종속성과 문맥 관계를 포착할 수 있도록 지원합니다. 이러한 기술적 접근은 그래프 인코더의 높은 계산 복잡도를 완화합니다.

- **Performance Highlights**: 다섯 개의 실제 데이터셋에서 EA-GPS의 성능을 평가한 결과, 기존의 선진 모델들에 비해 우수한 성능이 입증되었습니다. 특히, EA-GPS는 더 적은 파라미터 수와 낮은 학습 오버헤드로 뛰어난 성능을 유지하며, 자원 제약이 있는 엣지 디바이스에서도 효율적으로 사용할 수 있도록 설계되었습니다.



### SentiFormer: Metadata Enhanced Transformer for Image Sentiment Analysis (https://arxiv.org/abs/2502.15322)
- **What's New**: 최근 소셜 미디어 사용자들이 일상 감정을 표현하기 위해 이미지를 포스팅함에 따라, 이미지 감정 분석(image sentiment analysis)이 더욱 주목받고 있습니다. 본 논문에서는 이미지와 여러 메타데이터를 통합하기 위해 메타데이터 강화 트랜스포머(Metadata Enhanced Transformer, SentiFormer)를 제안합니다. 이 방법은 이미지에 대한 텍스트 설명과 키워드 태그와 같은 여러 메타데이터를 융합하여 감정 분석의 정확성을 높입니다.

- **Technical Details**: SentiFormer의 구조는 세 가지 주요 모듈로 구성되어 있습니다: 특징 표현 모듈, 적응적 유사도 학습 모듈, 그리고 교차 모달 융합 및 예측 모듈입니다. 먼저, BLIP을 사용하여 이미지에 대한 텍스트 설명을 생성하고, Faster R-CNN을 통해 주요 객체 태그를 추출한 후, Hiera를 통해 장면 태그를 얻습니다. 이후 CLIP을 활용하여 이미지와 메타데이터의 통합 표현을 생성하며, 이를 통해 더욱 효과적인 감정 분석을 가능하게 합니다.

- **Performance Highlights**: 세 개의 공개 데이터세트에서 수행된 실험 결과는 제안된 SentiFormer가 기존 방법들보다 우수한 성능을 보임을 보여줍니다. 메타데이터를 활용함으로써 감정 분석의 정확도를 크게 향상시키는 동시에, 기존의 수작업 특성 분석에서 벗어난 다양한 데이터 통합의 가능성을 제공합니다. 연구 결과는 감정 이해에 대한 새로운 전망을 제시하며, 코드와 데이터셋도 공개되어 있어 연구 커뮤니티의 추가 연구에 기여할 수 있습니다.



### Road Traffic Sign Recognition method using Siamese network Combining Efficient-CNN based Encoder (https://arxiv.org/abs/2502.15307)
- **What's New**: 이 논문에서는 교통 표지 인식(traffic sign recognition, TSR) 문제를 해결하기 위해 IECES-network라는 새로운 모델을 제안합니다. 이 모델은 효율적인 CNN(Efficient-CNN) 기반 인코더와 시암 네트워크(Siamese net)를 사용하여 복잡한 환경에서 발생하는 모션 블러(motion-blur) 및 가림(occlusion) 문제를 처리하는 데 중점을 두고 있습니다. 이 접근법은 특히 실시간 인식에서 높은 정확성과 강인성을 달성하는 데 중요한 기여를 합니다.

- **Technical Details**: 제가 제안하는 IECES 네트워크는 세 가지 단계로 구성됩니다: Efficient-CNN 기반 인코더, 시암 백본(Siamese backbone), 그리고 완전 연결 층(fully-connected layers)입니다. 초기 단계에서는 합성 훈련 샘플과 표준 이미지를 사용하여 교통 신호의 특징을 추출하고 인코딩하며, 이후 시암 신경망을 통해 입력 샘플과 템플릿 간 거리 계산을 통해 모션 블러 및 가림 샘플에 대비하는 강인성을 개선하는 방법을 모색합니다. 이러한 특성들은 SoftMax 함수를 사용하는 분류층에서 코드 결합을 통해 최종적으로 교통 표지의 분류를 가능하게 합니다.

- **Performance Highlights**: 제안된 IECES 네트워크는 Tsinghua-Tencent 100K 데이터셋과 독일 교통 표지 인식 벤치마크(German Traffic Sign Recognition Benchmark) 데이터셋에서 우수한 성능을 입증했습니다. 다른 최첨단 방법들과 비교할 때, 모션 블러와 가림 환경에서도 88.1%의 정밀도(precision), 86.43%의 재현율(recall), 86.1%의 정확도(accuracy)를 기록했으며, 모델의 경량 스케일은 2.9M에 불과합니다. 또한 처리 시간은 프레임당 0.1초로 기존 방법보다 1.5배 향상되었습니다.



### SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention (https://arxiv.org/abs/2502.15304)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 KV 캐시 성능을 극대화하기 위해 SVDq라는 새로운 혼합 정밀도 양자화 기법을 제안합니다. 이 방법은 SVD(특이값 분해)를 이용하여 잠재 채널로 KV 캐시를 변환한 후, 중요도 기반 양자화 및 압축을 적용합니다. 또한, SVDq의 사용으로 양자화 오류가 기존 방법에 비해 상당히 낮음을 이론적으로 입증했습니다.

- **Technical Details**: SVDq 방법은 SVD 기반 채널 압축을 통합하여 KV 캐시의 양자화 정밀도를 최적화합니다. 이 방법은 SVD를 통해 얻은 특이값과 관련된 잠재 채널에 대해 더 높은 비트폭을 할당하며, 작은 특이값과 관련된 채널의 정밀도는 점차 감소시킵니다. 이를 통해 압축 비율을 높이면서도 비교 가능한 모델 성능을 유지할 수 있습니다.

- **Performance Highlights**: SVDq는 LongBench 및 RULER 벤치마크를 기준으로 하여, 키 캐시 정밀도를 1.25비트로 낮추면서 최대 410배의 압축 비율을 달성할 수 있음을 보여줍니다. 또한, 이 방법은 LongBench 데이터셋에서 거의 무손실에 가까운 결과를 기록하였으며, 전반적으로 LLMs의 높은 정밀도 저비트 양자화를 통해 KV 캐시 압축의 효율성을 높입니다.



### Beyond Fixed Variables: Expanding-variate Time Series Forecasting via Flat Scheme and Spatio-temporal Focal Learning (https://arxiv.org/abs/2502.15296)
- **What's New**: 최근 Multivariate Time Series Forecasting (MTSF) 분야에서 새로운 도전과제를 제시하는 Expanding-variate Time Series Forecasting (EVTSF)라는 과제가 소개되었습니다. 실제 어플리케이션에서는 새로운 센서가 추가됨에 따라 변수가 증가하지만, 기존 연구는 고정된 변수를 전제로 했습니다. 이 논문에서는 데이터 구조의 불일치와 불균형한 시공간 학습이라는 두 가지 주요 문제와 이를 해결하기 위한 새로운 접근 방식을 다룹니다.

- **Technical Details**: 새로 제안된 STEV는 유연한 시공간 예측 프레임워크로, Flat Scheme을 통해 변수 차원을 따라 2D 샘플을 평탄화하여 1D 공간으로 확장합니다. 이러한 구조는 변수가 증가하더라도 모델의 동적 공간적 상관관계를 유지합니다. 또한, 시공간 Focal Learning 전략을 도입하여 대조 학습과 그래프 표현 간의 갈등을 해결하는 부정 필터를 포함하고 있습니다.

- **Performance Highlights**: 세 개의 실제 데이터셋을 사용한 벤치마킹 결과, STEV는 확장된 변수를 포함할 때 기존 SOTA MTSF 모델보다 우수한 성능을 보였습니다. 특히, 관측치가 5%에 불과한 경우에도 STEV가 완전 관측으로 훈련된 모델과 동등한 성능을 발휘하였습니다. 다양한 확장 전략을 탐색함으로써 실제 응용에 대한 STEV의 일반화 가능성이 강조되었습니다.



### Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inferenc (https://arxiv.org/abs/2502.15294)
- **What's New**: 이 논문은 대형 언어 모델(LLM)에서의 대화 데이터 분석을 통해 대화의 라운드 수가 증가할수록 KV 캐시가 GPU 메모리에 저장되어 효율성을 저해하는 문제를 발견합니다. 이를 해결하기 위해 'Round Attention'이라는 새로운 라운드 수준의 주의 메커니즘을 제안합니다. 이 메커니즘은 가장 관련성 높은 라운드의 KV 캐시만을 회상하고 계산하여 효율성을 높입니다.

- **Technical Details**: 제안된 라운드 주의 메커니즘은 대화 라운드 수가 많아짐에 따라 발생하는 메모리 사용 문제를 해결하기 위해, 특정 라운드에 국한된 KV 캐시를 활용합니다. 이는 모델의 성능을 유지하면서도 메모리 사용량을 약 55% 줄이는 데 성공합니다. 실험은 모델의 효율성을 높이기 위해 다양한 대화 데이터 세트를 사용하여 진행되었습니다.

- **Performance Highlights**: 이 연구 결과는 Round Attention이 메모리 비용을 크게 줄여줄 뿐만 아니라 대형 언어 모델의 성능을 유지시킨다는 것을 보여줍니다. 이는 특히 긴 텍스트와 복잡한 작업을 처리하는 데 있어, LLM을 사용하는 시스템의 효율성을 크게 향상시킬 수 있는 잠재력이 있습니다.



### Time Warp: The Gap Between Developers' Ideal vs Actual Workweeks in an AI-Driven Era (https://arxiv.org/abs/2502.15287)
Comments:
          ICSE SEIP 2025

- **What's New**: 이번 논문은 Microsoft에서 484명의 소프트웨어 개발자들을 대상으로 한 설문조사의 결과를 제시하며, 이상적인 업무 주간과 실제 업무 주간 간의 차이를 규명합니다. 이 연구는 이상적인 업무 주간에 대한 개발자의 기대와 실제 작업 간의 불일치를 분석하여 생산성과 개발자의 만족도에 미치는 영향을 평가합니다. 특히, 이러한 불일치가 커질수록 생산성과 만족도가 감소하는 경향을 발견하였습니다.

- **Technical Details**: 연구에서는 개발자들이 실제로 얼마나 다양한 활동에 시간을 할애하는지와 그들이 이상적으로 예상하는 시간 배분을 비교했습니다. 설문조사는 개발자의 직무 경험, 시간 할당, 생산성 및 만족도에 대한 자가 보고를 포함하여 다양한 질문을 포함하고 있습니다. 16개의 핵심 활동이 조사에 포함되어 이러한 시간 할당을 정량화하는 데 사용되었습니다.

- **Performance Highlights**: 이번 연구의 주요 기여는 1) 작업 주간 차이가 개발자의 생산성과 만족도에 미치는 영향을 정량화하고, 2) 만족도와 생산성에 부정적인 영향을 미치는 개별 작업을 식별하며, 3) 향후 AI 자동화 작업을 위한 데이터 기반 통찰을 제공하는 것입니다. 개발자들이 어떤 작업을 자동화하기를 원하며, AI 도구 사용이 그들의 업무 만족도와 생산성에 미치는 영향을 분석했습니다.



### Offload Rethinking by Cloud Assistance for Efficient Environmental Sound Recognition on LPWANs (https://arxiv.org/abs/2502.15285)
- **What's New**: 이 논문에서 소개하는 ORCA(Offload Rethinking by Cloud Assistance)는 저전력 배터리 없는 장치에 최적화된 클라우드 지원 기반의 환경 소리 인식 시스템입니다. ORCA는 Low-Power Wide-Area Networks (LPWANs)을 통해 자원 효율적이고 정확한 소리 인식을 가능하게 하며, 기존의 에지 클라우드 협업에서 발생하는 통신 비용과 에너지 사용 문제를 해결합니다. 특히, ORCA는 자가 주의(self-attention) 기반의 기능 선택 방법을 활용하여 연속적인 오디오 감시를 가능하게 합니다.

- **Technical Details**: ORCA는 에지 장치에서의 추론을 유지하면서 서버의 도움을 받는 구조로 설계되었습니다. 서버는 에지 장치에서 샘플링한 중요한 입력 특징을 식별하여 정보를 전달하고, 에지 장치는 주어진 정보를 통해 에너지를 절약하며 최적의 성능을 발휘합니다. ORCA는 통신 비용과 저속 데이터 전송, 동적인 무선 채널 상태의 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 실험 결과, ORCA는 에너지 절약에서 최대 80배, 지연 시간에서 최대 220배 향상된 성능을 보여주었습니다. 또한, 다양한 공개 환경 소리 데이터셋에서 2.5%에서 12.5% 범위의 정확도 개선을 기록하였습니다. 이러한 결과는 스마트 시티와 환경 모니터링 등의 다양한 분야에서의 활용 가능성을 시사합니다.



### CopyJudge: Automated Copyright Infringement Identification and Mitigation in Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.15278)
Comments:
          17pages, 8 figures

- **What's New**: 본 논문에서는 AI 생성 이미지들이 저작권이 있는 작품과 실질적으로 유사한지를 판단하기 위한 자동화된 저작권 침해 식별 프레임워크인 CopyJudge를 제안합니다. CopyJudge는 대형 비전-언어 모델(LVLM)을 활용하여 법원 절차를 모방하고, 이미지의 유사성을 평가하며, 침해 가능성에 대한 자세한 판단 근거를 제공합니다. 또한, 본 연구에서는 침해를 방지하기 위한 자동화된 전략을 통해 감지된 침해적인 프롬프트를 최적화하는 방법을 탐구합니다.

- **Technical Details**: CopyJudge는 이미지의 다양한 요소를 분해하고 필터링하여 저작권 보호를 받지 않는 부분을 제외한 후, 필터링된 부분 간의 유사성을 비교하는 추상화-필터링-비교 테스트 프레임워크를 사용합니다. 여러 LVLM이 서로 논의하고 점수를 매기는 다중 에이전트 토론 방법을 도입하여, 최종적으로 메타 심사관이 공감대를 기반으로 점수와 근거를 제시합니다. 이 과정에서 인간의 선호도를 반영하기 위해 소수의 사례를 통한 교육을 진행합니다.

- **Performance Highlights**: 실험 결과, CopyJudge는 최첨단 성능에 부합하면서도 다양한 형태의 침해에 대해 뛰어난 일반화 능력과 해석 가능성을 제공합니다. 또한 제안된 완화 방법은 비침해 표현을 손실하지 않고 메모리화 및 지적 재산권 침해를 보다 효과적으로 완화할 수 있는 것으로 나타났습니다. 이러한 성능 향상은 AI 생성 이미지의 저작권 문제 해결에 기여할 것으로 기대됩니다.



### Corrections Meet Explanations: A Unified Framework for Explainable Grammatical Error Correction (https://arxiv.org/abs/2502.15261)
Comments:
          19 pages, 2 figures, and 9 tables

- **What's New**: 이 논문에서는 언어 학습자를 위한 설명 가능한 문법 오류 수정 시스템(EXGEC)을 소개하고 있습니다. 기존 연구들이 교정 과정과 설명 간의 관계를 무시하는 반면, EXGEC는 이 두 가지 작업이 서로 강화될 수 있도록 설계되었습니다. 또한, 데이터셋 EXPECT에서 발견된 많은 노이즈 문제를 해결하기 위해, 수정된 EXPECT-denoised 데이터셋을 제안하여 보다 객관적인 학습 프레임워크를 제공하고 있습니다.

- **Technical Details**: EXGEC는 교정(correction) 및 설명(explanation) 작업을 통합하는 통합된 다중 작업 프레임워크로, 이를 통해 상호 작용(interaction)을 강화합니다. 논문은 EXPECT 데이터셋을 기반으로 하여, 심층 신경망 모델을 활용해 문법 오류를 감지하고, 증거 단어(extractive rationales)를 추출하며, 오류를 분류하는 방식을 구체적으로 다룹니다. 예를 들어, 선행 설명(pre-explaining) 모델과 후행 설명(post-explaining) 모델의 성능 차이를 분석하여, 예측 순서가 작업 성능에 미치는 영향을 밝혀냈습니다.

- **Performance Highlights**: EXGEC는 BART, T5 및 Llama3와 같은 여러 자연어 처리(NLP) 모델에서 교정 및 설명 작업 모두에서 단일 작업 기준선(single-task baselines)을 초과하는 성능을 보여주었습니다. 실험 결과, EXGEC 모델은 기존의 단일 작업 모델보다 교정 및 설명 작업에서 더 나은 성능을 보이며, 서로의 긍정적 상호 작용을 입증했습니다. 이는 EXGEC가 다양한 예측 순서에서 이 두 가지 작업의 관계를 효과적으로 탐구할 수 있음을 시사합니다.



### ComposeOn Academy: Transforming Melodic Ideas into Complete Compositions Integrating Music Learning (https://arxiv.org/abs/2502.15255)
- **What's New**: 이번 연구에서는 ComposeOn이라는 음악 이론 기반의 새로운 도구를 소개합니다. 이 도구는 공식적인 음악 교육을 받지 않은 사용자들이 쉽게 멜로디 아이디어를 완전한 작곡으로 확장할 수 있도록 지원합니다. ComposeOn은 초보자, 중급자 및 고급자 수준에 맞춰 음악 창작을 설명하여 사용자들이 음악 이론을 쉽게 학습할 수 있도록 합니다.

- **Technical Details**: ComposeOn은 사용자가 입력한 멜로디에 따라 조성 및 화음 진행 이론을 바탕으로 곡을 이어나갈 수 있는 제안을 합니다. 이를 위해 음악 이론의 기초적인 3화음 구조를 이용한 조화 진행 및 MIDI(뮤지컬 악기 디지털 인터페이스) 데이터를 변환하는 기술이 포함되어 있습니다. 이 도구는 사용자의 아이디어를 완전한 작곡으로 발전시킬 수 있도록 돕는 구조적 교육적 접근을 제공합니다.

- **Performance Highlights**: 사용자 연구(N=10) 결과에 따르면, ComposeOn은 적은 음악적 기술을 가진 사용자들에게 더 접근 가능하고 즐거운 작곡 및 학습 경험을 제공합니다. 기존의 생성형 음악 도구와 달리, ComposeOn은 개인적인 표현력과 학습을 강조하여 이론 기반의 음악 창작 방법을 제공합니다. 이는 사용자가 음악의 기본 원리를 이해하고 그들의 창의성을 발휘하도록 도와줍니다.



### Comparative Analysis of Large Language Models for Context-Aware Code Completion using SAFIM Framework (https://arxiv.org/abs/2502.15243)
Comments:
          9 pages

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 발전이 코드 완성에 미친 영향을 분석합니다. 특히, Gemini 1.5 Flash, Gemini 1.5 Pro, GPT-4o, GPT-4o-mini, GPT-4 Turbo 등 다양한 채팅 기반 LLM의 성능을 평가합니다. Syntax-Aware Fill-in-the-Middle (SAFIM) 데이터셋을 사용하여 구문 민감 코드 생성 능력을 비교 분석합니다.

- **Technical Details**: 이 연구에서는 모델의 정확성과 효율성을 측정하기 위해 cosine similarity와 latency와 같은 성능 지표를 사용하였습니다. SAFIM 데이터셋은 구문 인식 코드 생성을 평가하기 위해 특별히 설계되었습니다. 각 모델의 코드 완성 능력은 정확성과 속도 간의 절충을 강조하는 데이터를 통해 비교됩니다.

- **Performance Highlights**: 연구 결과, 다양한 LLM 모델 간의 코드 완성 능력에서 상당한 차이를 발견하였습니다. 모델별 강점과 약점을 드러내면서 향후 LLM 기반 코드 완성을 위한 기준점을 제시합니다. accuracy와 speed 간의 균형을 이해하는 데 유용한 통찰을 제공합니다.



### AutoMR: A Universal Time Series Motion Recognition Pipelin (https://arxiv.org/abs/2502.15228)
Comments:
          5 figures

- **What's New**: 이 논문에서는 다중 모달 데이터셋을 위한 자동화된 모션 인식 파이프라인인 AutoMR을 제안합니다. 이 프레임워크는 데이터 전처리, 모델 학습, 하이퍼파라미터 튜닝 및 평가를 통합하여 다양한 시나리오에서 강력한 성능을 발휘합니다. 두 가지 주요 도전 과제를 해결하려고 하며, 기존의 복잡한 전처리 과정 없이도 효율적인 모션 인식을 가능케 합니다.

- **Technical Details**: AutoMR은 데이터셋의 다양성, 모델 확장성, 하이퍼파라미터 최적화와 같은 문제를 해결하는 통합 솔루션을 제공합니다. 이 시스템은 QuartzNet 모델을 핵심으로 사용하며, 다양한 데이터셋에서 모델을 표준화하고 자동 튜닝을 통해 성능을 향상시킵니다. 또한, 데이터셋의 일관성을 보장하기 위해 구조화된 포맷을 정의하고 데이터 전처리 및 훈련 구성 모듈을 포함하고 있습니다.

- **Performance Highlights**: AutoMR은 10개의 벤치마크 데이터셋에서 기존의 최신 기술들과 비교하여 높은 정확도를 달성하며, 특히 OPPORTUNITY 데이터셋에서는 이전 모델보다 5% 이상 높은 성능을 기록했습니다. 그러나 DB4와 LMDHG에서는 성능이 다소 낮아, 이러한 복잡한 데이터에서의 향상이 필요합니다. 자동 하이퍼파라미터 튜닝의 경우, 대부분의 데이터셋에서 수동 조정에 맞먹거나 우수한 성능을 보여, 전문가 개입 없이도 효과적인 최적화가 가능함을 증명했습니다.



### Understand User Opinions of Large Language Models via LLM-Powered In-the-Moment User Experience Interviews (https://arxiv.org/abs/2502.15226)
- **What's New**: 이 논문에서는 CLUE( Contextualized LLM-powered User Experience understanding)를 소개하며, LLM과의 상호작용 이후 사용자 경험 인터뷰를 자동으로 진행하여 사용자 의견을 수집하는 프레임워크를 제시합니다. CLUE-Interviewer는 사용자가 LLM과 대화한 직후에 반응을 기록하여, 사용자 의견의 깊이 있는 인사이트를 수집하는 새로운 방법론을 제공합니다. 이 방법은 수천명의 사용자를 대상으로 하여, 다양한 주제에 대한 대화와 인터뷰 세션을 기록하였습니다.

- **Technical Details**: CLUE는 두 가지 주요 구성 요소인 CLUE-Interviewer와 CLUE-Insighter로 구성됩니다. CLUE-Interviewer는 사용자와 LLM의 대화 후 반응을 수집하기 위해 반구조화된 UX 인터뷰를 자동으로 수행합니다. 이러한 인터뷰는 미리 정의된 차원에 대한 깊은 통찰을 유도하며, 사용자의 응답에 따라 주제를 탐색할 수 있는 유연성을 가지고 있습니다.

- **Performance Highlights**: 사용자 연구 결과, CLUE-Interviewer는 다양한 주제에 대한 시의적절한 사용자 의견을 효과적으로 수집하는 것으로 나타났습니다. 사용자들은 LLM에 대해 전반적으로 보수적인 의견을 보였지만, 특정 차원에 대해 높은 평가를 하였습니다. 또한, 사용자들은 시각 및 멀티미디어 기능, 신선한 정보 접근, 개인화된 응답 등 새로운 기능 요청을 통해 향후 LLM 개발을 위한 동기와 증거를 제공하였습니다.



### Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs (https://arxiv.org/abs/2502.15224)
Comments:
          13 pages

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 인간과 유사한 방식으로 과학적 연구를 수행하고 새로운 지식을 발견할 수 있는지를 탐구합니다. 이를 위해 연구자들은 LLM의 과학적 발견 능력을 평가하기 위한 새로운 벤치마크인 	extit{Auto-Bench}를 소개합니다. 이 벤치마크는 인과 그래프 발견 원칙을 기반으로 하여, 모델이 숨겨진 구조를 발견하고 최적의 결정을 내릴 수 있도록 도전합니다.

- **Technical Details**: 연구 방법론에서 LLM 모델은 화학 및 사회 네트워크를 포함한 두 가지 벤치마크를 통해 그 성능을 평가받습니다. 이들 벤치마크는 노드를 포함한 그래프로 구성되며, 각 노드에서의 개입이 연결된 노드에 미치는 영향을 분석합니다. LLM은 초기 가설을 바탕으로 반복적으로 실험을 수행하며, 새로운 관찰 데이터를 통해 가설을 수정합니다.

- **Performance Highlights**: 실험 결과, 현재의 LLM 모델들이 문제의 복잡성 증가에 따라 성능 저하를 겪는다는 것을 발견했습니다. 예를 들어, 화학 및 사회 네트워크에서 노드 수가 증가하면 평균적으로 정답을 얻는 데 필요한 사이클 수가 급격히 감소합니다. 이 연구는 기계와 인간 지능 간의 정보 처리에서의 중요한 차이를 강조하며, 향후 LLM 발전에 있어 이를 해결해야 할 필요성을 제기합니다.



### FormalSpecCpp: A Dataset of C++ Formal Specifications created using LLMs (https://arxiv.org/abs/2502.15217)
Comments:
          Accepted at the 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR)

- **What's New**: FormalSpecCpp는 C++ 프로그램의 formal specification을 검사하기 위한 표준화된 벤치마크가 부족한 문제를 해결하기 위해 설계된 데이터셋입니다. 이는 잘 정의된 precondition과 postcondition을 가진 C++ 프로그램의 최초의 종합적인 모음으로, 연구자와 개발자가 specification inference tool을 벤치마킹하고 생성된 specification의 정확성을 테스트하는 데 사용할 수 있습니다. 이 데이터셋의 공개는 프로그램 검증, specification inference, AI 지원 소프트웨어 개발의 연구를 진전시키는 목표를 가지고 있습니다.

- **Technical Details**: FormalSpecCpp 데이터셋은 verified Dafny 프로그램을 기반으로 하여 OpenAI의 GPT-4-turbo 모델을 사용하여 C++로 번역됩니다. 이 과정은 formal specification을 유지하면서 Dafny 프로그램을 번역하는 구조화된 접근 방식을 채택합니다. 또한, prompt engineering 기법을 활용하여 정확한 타입 매핑, assertion 변환 및 안전 제약 조건 처리를 통해 데이터셋의 품질을 향상시킵니다.

- **Performance Highlights**: 본 연구에서 변환된 105개의 C++ 파일은 모두 최초 시도에서 성공적으로 컴파일되었으며, 전체 과정은 약 27분만에 완료되었습니다. 변환 비용은 총 $2.07로, 파일당 평균 $0.02가 소요되었습니다. 이와 함께 관련 테스트 케이스를 생성하는 데 추가 $1.31이 들었고, 약 15분이 소요되었습니다.



### The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning (https://arxiv.org/abs/2502.15214)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 Reinforcement Learning (RL)과 Large Language Models (LLMs), Vision-Language Models (VLMs)의 통합을 탐구하는 설문 조사로, RL에서 자주 발생하는 도전 과제를 극복하기 위한 다양한 접근 방식을 리뷰합니다. RL에서는 인간이 설계한 보상, 샘플 비효율성, 일반화 부족, 해석 가능성 제한 등의 문제를 가지고 있으며, LLMs와 VLMs의 통합이 이를 해결할 기회를 제공합니다. 특히, 이 논문은 LLM/VLM을 에이전트, 플래너, 보상 역할로 체계적으로 분류한 새로운 분류 체계를 제시합니다.

- **Technical Details**: MDP (Markov Decision Process)는 상태 집합(S), 행동 집합(A), 전이 확률 함수(T), 보상 함수(R), 할인 계수(γ)로 정의됩니다. RL 에이전트는 환경과의 상호작용을 통해 정책(π)을 학습하고, 이는 상태를 행동으로 매핑합니다. 논문은 또한 LLM과 VLM이 RL에서 수반하는 역할에 대해 강조하며, 이는 효과적인 데이터 효율성, 일반화 및 해석 가능성을 향상시킬 수 있습니다.

- **Performance Highlights**: LLMs는 자연어 처리에서 혁신적인 발전을 이룩하였으며, VLMs는 이와 결합하여 이미지와 텍스트 간의 의미를 이해하고 해석하는 능력을 가집니다. 이 연구는 LLM/VLM과 RL의 통합이 에이전트의 행동과 학습 방식을 혁신적으로 변화시킬 수 있음을 강조합니다. 향후 연구 방향으로는 기초 모델(Foundation Models)과 RL의 통합에 있어 한계를 극복하고, 공정성, 편향 완화 및 개선된 표현과 같은 상징적인 문제를 다루는 것이 포함됩니다.



### PairBench: A Systematic Framework for Selecting Reliable Judge VLMs (https://arxiv.org/abs/2502.15210)
- **What's New**: 본 논문에서는 PairBench라는 새로운 프레임워크를 소개합니다. 이는 대규모 비전 언어 모델(VLMs)을 자동 평가자로 평가하기 위한 저비용 시스템으로, 다양한 모달리티(modality)와 시나리오에서 사용될 수 있습니다. PairBench는 유사성 점수의 핵심 요구 사항을 나타내는 네 가지 메트릭(metrics)을 도입하여 VLMs의 성능을 평가합니다.

- **Technical Details**: PairBench는 인간 주석(human annotations)과의 일치성, 데이터 쌍의 순서에 관계없는 일관성, 유사성 분포의 부드러움(smoothness), 프롬프트(prompting)를 통한 제어 가능성 등을 측정합니다. 분석 결과, 모든 메트릭에서 우수한 성능을 보이는 모델은 없었으며, 특정 평가자의 원하는 행동에 따라 최적의 선택이 달라질 수 있음을 보여주었습니다.

- **Performance Highlights**: 많은 VLM들이 순서와 상관없이 대칭 유사성 점수를 유지하는 데 어려움을 겪는 것으로 나타났습니다. PairBench에서의 성능은 기존의 인기 있는 벤치마크와 밀접하게 관련되어 있으며, 이를 통해 모델 순위에서 예측력을 보여줍니다. 이러한 결과는 VLMs를 평가자로 널리 사용하기 전에 철저한 평가가 필요함을 강조합니다.



### FlipConcept: Tuning-Free Multi-Concept Personalization for Text-to-Image Generation (https://arxiv.org/abs/2502.15203)
Comments:
          9 pages, 4 figures

- **What's New**: 최근 텍스트에서 이미지로(텍스트-투-이미지, T2I) 여러 개인화된 개념을 단일 이미지에 통합하는 방법이 주목받고 있습니다. 기존 방법들은 복잡한 장면에서 비개인화 영역의 왜곡 때문에 성능 저하를 겪습니다. 이를 해결하기 위해 FlipConcept이라는 새로운 접근법을 제안하며, 추가 조정 없이 여러 개인화된 개념을 이미지에 원활하게 통합할 수 있습니다. 제안된 방법은 개인화된 개념의 외관을 정확하게 모방하는 Guided Appearance Attention을 도입합니다.

- **Technical Details**: FlipConcept 프레임워크는 두 가지 단계로 작동합니다. 첫 번째 단계에서는 백그라운드 이미지와 이를 기반으로 생성된 마스크를 준비하고, Edit-Friendly DDPM inversion을 통해 편집하기 용이한 잠재 표현을 얻습니다. 두 번째 단계에서는 이전 단계의 잠재 표현과 마스크를 활용하여 개인화된 개념을 백그라운드와 통합하는 이미지를 생성합니다. Guided Appearance Attention, Mask-Guided Noise Mixing 및 Background Dilution의 세 가지 핵심 기술을 도입하여, 개념 간의 관계를 유지하면서 비개인화 영역에 대한 간섭을 최소화합니다.

- **Performance Highlights**: FlipConcept은 기존 방법들을 능가하는 성능을 보여주었으며, 복잡한 장면에서도 개인화된 이미지 생성에 강점을 보입니다. 특히, 여러 캐릭터나 객체가 포함된 상황에서도 각 개념의 일관성을 유지하고 배경의 완전성을 보장합니다. 실험 결과, CLIP 평가 점수에서도 높은 성과를 달성하여 생성된 콘텐츠의 품질과 관련성을 강조하였습니다.



### TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding (https://arxiv.org/abs/2502.15197)
Comments:
          15 pages, 10 figures, 5 tables

- **What's New**: 본 논문에서는 다중 요청 설정에서 배치 추측 디코딩(batch speculative decoding)의 전체 처리량을 최적화하는 새로운 방법인 TETRIS를 제안합니다. 기 기존 방법들과 달리, TETRIS는 병렬 검증에서 승인될 가능성이 높은 드래프트 토큰(draft tokens)을 적극적으로 선택하여, 거부되는 토큰을 줄이고 컴퓨팅 리소스를 효율적으로 활용합니다. 이러한 접근은 특히 제한된 추론 용량을 가진 서비스 제공자에게 빠른 추론을 가능하게 합니다.

- **Technical Details**: TETRIS는 각 요청의 토큰이 검증될 때 수용될 가능성이 높은 드래프트 토큰을 그리드 방식으로 선택하는 방식으로, 이는 전체 처리량을 최대화하는 데 중점을 둡니다. 이 방법론은 토큰의 수용 가능성을 고려하여 최적의 드래프트 토큰 선택을 동적으로 결정하며, 이는 컴퓨터 자원의 활용을 더욱 개선합니다. TETRIS는 이론적으로 각 디코딩 단계에서 최적 처리량을 달성한다고 증명합니다.

- **Performance Highlights**: TETRIS는 기존의 추측 디코딩에 비해 일관되게 더 높은 수용률을 기록하며 제한된 추론 용량의 활용도를 더욱 효과적으로 증가시킵니다. 실험 결과, TETRIS는 표준 추측 디코딩 및 동적 드래프트 윈도우를 이용한 기존 방법들에 비해 전체 처리량과 지연(latency)가 개선되었습니다. 이는 TETRIS가 실제 모델 서비스 배포에서 추론 속도를 개선할 수 있는 잠재력을 지니고 있음을 나타냅니다.



### Scale-Free Graph-Language Models (https://arxiv.org/abs/2502.15189)
- **What's New**: 이번 논문은 그래프-언어 모델(GLMs)을 새롭게 발전시켜 그래프 생성(graph generation)과 텍스트 임베딩(text embedding)을 통합한 단일 프레임워크를 제안합니다. 특히, 실제 엣지 분포의 고유한 특성인 스케일-프리 속성을 사용해 그래프 생성을 이루며, k-최근접 이웃(kNN) 그래프를 통한 근사화에 중점을 둡니다. 이러한 접근 방식은 LM의 미세 조정(finetuning)을 위한 보조 감독(supervision)을 제공하는 그래프 기반의 유사 라벨러(pseudo-labeler)를 개발하는 데 도움을 줍니다.

- **Technical Details**: 제안된 모델인 스케일-프리 그래프 언어 모델(SFGL)은 그래프 생성 및 텍스트 임베딩 문제를 동시에 해결합니다. 연구팀은 LMs와 GNNs(그래프 신경망)의 시너지를 활용하여 실제 스케일-프리 구조를 기반으로 한 그래프 생성과 텍스트 임베딩의 통합을 시도했습니다. 이 모델은 잘 정의된 구조적 선행 사전(structural prior)를 통해 학습의 복잡성과 편향을 줄입니다.

- **Performance Highlights**: 제안된 SFGL 모델은 다양한 데이터세트에서 실험을 통해 KNN 그래프의 스케일-프리 구조 근사화의 유효성을 입증하였습니다. 이 모델은 처리 성능을 향상시키기 위해 반복적인 훈련(iterative training)을 실시할 수 있으며, 학습된 LMs는 더 풍부한 의미 정보를 담은 텍스트 임베딩을 생성합니다. GLMs의 잠재력을 강조한 이 연구는 스케일-프리 구조에 기반한 접근 방식을 통해 그래프 기반 반감독 학습(semi-supervised learning)의 신규 가능성을 제시합니다.



### LUMINA-Net: Low-light Upgrade through Multi-stage Illumination and Noise Adaptation Network for Image Enhancemen (https://arxiv.org/abs/2502.15186)
Comments:
          9 pages, 4 figures

- **What's New**: 본 논문에서는 저조도 이미지 향상(Low-Light Image Enhancement) 기술의 새로운 프레임워크인 LUMINA-Net을 제안합니다. LUMINA-Net은 다단계 조명 및 반사 모듈을 통합하여 저조도 이미지의 품질 문제를 해결하고, 기존 기술들이 겪는 한계를 극복합니다. 이는 텍스처 디테일을 보존하면서 조명과 대비를 정교하게 조정하고, 노이즈 제거 메커니즘을 통해 이미지의 자연스러운 색상과 균형을 유지합니다.

- **Technical Details**: LUMINA-Net은 Retinex 이론을 기반으로 하여, 두 개의 저조도 이미지 쌍을 활용하여 조명과 반사 성분을 분리하는 기능을 가지고 있습니다. 주요 혁신으로는 채널 안내(Channel-Guidance) 모듈이 포함되어 있어, 공간 및 채널 주의 메커니즘을 통해 특성 추출을 개선합니다. 또한, 색상 향상(Color Enhancement) 모듈이 자연스러운 색 복원을 보장하며, 과잉 노출(Over-Exposure Correction) 모듈이 과도하게 밝은 지역을 동적으로 조정하여 아티팩트를 최소화합니다.

- **Performance Highlights**: 다양한 LOL 및 SICE 데이터셋에서 수행한 실험을 통해 LUMINA-Net의 뛰어난 성능이 입증되었습니다. PSNR, SSIM 및 LPIPS 메트릭스에서 기존의 최첨단 방법들을 초월하며, 노이즈를 효과적으로 억제하고 세부 사항을 보존합니다. 이러한 성능 향상은 저조도 환경에서의 비전 기반 시스템의 잠재력을 극대화하는 데 기여합니다.



### Key Body Posture Characteristics of Short-distance Speed Skaters at the Start Based on Artificial Intelligenc (https://arxiv.org/abs/2502.15185)
- **What's New**: 이번 연구는 중국 남성 단거리 스피드 스케이팅 선수들의 스타트 기술에 대한 생체역학적 분석을 수행하였습니다. 13명의 고수준 선수를 대상으로 인공지능 비디오 캡처 시스템을 활용하여 운동 분석을 진행하였습니다. 스타팅 동작에 영향을 미치는 주요 요소들을 도출하였고, 이로써 해당 스포츠의 기술 향상에 기여할 수 있는 기초 자료를 제공하고자 하였습니다.

- **Technical Details**: 연구는 세 가지 단계인 시작 준비, 시작, 스프린트 과정에서 선수들의 신체 자세 특징을 분석하였습니다. 결과적으로, 시작 준비 단계에서의 포스트 안정 각도(post-stability angle), 앞다리 무릎 각도(anterior knee angle), 뒷다리 무릎 각도(posterior knee angle), 보폭(stride length) 등이 스타트 속도와 긍정적인 상관관계를 나타냈습니다. 특히, 트렁크 각도(trunk angle)는 스타트 속도와 높은 부정적 상관관계가 있음을 발견하였습니다.

- **Performance Highlights**: 스타트 속도에 가장 큰 영향을 미치는 요소로는 보폭, 왼쪽 무릎 각도, 포스트 안정 각도가 확인되었습니다. 포스트 안정 각도와 왼쪽 무릎 각도가 클수록, 보폭이 길어질수록 스타트 속도가 빨라지는 결과를 보였습니다. 또한, 시작 및 스프린트 단계에서는 얼음 접촉 각도(ice-contact angle)와 추진 각도(propulsion angle)가 작아질수록, 트렁크 각도와 엉덩이 각도의 변화가 더 큰 것으로 나타나 스타트 동작의 효과성이 증가했습니다.



### LEDD: Large Language Model-Empowered Data Discovery in Data Lakes (https://arxiv.org/abs/2502.15182)
- **What's New**: LEDD는 데이터 레이크에서 의미론적 의미를 갖는 계층적 글로벌 카탈로그와 의미론적 테이블 검색을 제공하는 end-to-end 시스템으로, 대규모 언어 모델(LLM)을 활용합니다. 이 시스템은 데이터 발견 알고리즘의 확장을 용이하게 하기 위한 Python 인터페이스를 제공합니다. 기존의 데이터 관리 시스템은 이러한 기능을 지원하지 않았으며, LEDD는 이 문제를 해결하는 첫 프로토타입 시스템입니다.

- **Technical Details**: LEDD는 Python 3.11 및 Javascript로 구현되었으며, OpenAI의 API를 활용하여 LLM과 상호작용합니다. IGinX의 데이터 접근과 Zeppelin을 통한 기본 시각화 인터페이스를 이용해 작동하며, 이 시스템의 핵심 기능은 Apache Zeppelin의 확장 그래프 시각화 구성 요소로 구현됩니다. LEDD는 밀버스(Milvus)라는 벡터 데이터베이스를 사용하여 임베딩 벡터의 효율적인 검색을 지원합니다.

- **Performance Highlights**: LEDD는 계층적 글로벌 데이터 카탈로그와 의미론적 테이블 검색 기능을 통해 데이터 발견의 효율성을 크게 향상시킵니다. 사용자는 자연어로 자신의 의도를 표현할 수 있으며, LEDD는 관련 테이블과 열을 찾아 계층적 뷰로 구성합니다. 실제 데이터 탐색 중에는 실시간으로 계층 그래프 뷰 간의 관계를 분석하여 중요한 관계를 확장하여 보여줍니다.



### Methods and Trends in Detecting Generated Images: A Comprehensive Review (https://arxiv.org/abs/2502.15176)
Comments:
          30 pages, 4 Figures, 10 Tables

- **What's New**: 최근 생성 모델의 발전으로 인해 고품질 멀티미디어 데이터 합성이 가능해졌으나, 이로 인한 악성 공격 및 사회적 해악에 대한 우려도 커지고 있습니다. 이러한 문제를 인식한 연구자들은 합성 데이터를 효과적으로 탐지할 수 있는 방법론 개발에 집중하고 있으며, 기존 하위 전술을 넘어서 다중 모달 프레임워크를 활용한 최근 기술들을 포괄적으로 검토했습니다.

- **Technical Details**: 본 조사에서는 최신 생성형 AI 모델에 의해 생성된 합성 이미지를 탐지하고 분류하기 위한 방법론을 종합적으로 리뷰합니다. 이를 통해 핵심 탐지 방법론을 체계적으로 분석하고, 공통점을 도출하여 유의미한 세분화로 분류했습니다. 대규모 데이터셋의 중요성도 강조되며, 공개적으로 사용 가능한 데이터셋 개요도 제공하여 추가 연구 및 벤치마킹의 기초를 마련합니다.

- **Performance Highlights**: 생성형 모델의 발전에도 불구하고 합성 이미지 탐지 연구는 일반화, 견고성, 확장성 등의 문제에 직면해 있습니다. 본 연구는 이러한 문제를 해결하기 위한 탐지 방법론의 진전을 요약하고 기존의 한계점을 식별하며 미래 방향성을 제시합니다. 궁극적으로, 생긴 위협적 요소가 증가하는 가운데 대응책 마련이 절실하다는 점을 강조합니다.



### Extreme Speech Classification in the Era of LLMs: Exploring Open-Source and Proprietary Models (https://arxiv.org/abs/2502.15155)
Comments:
          Accepted to 7th International Conference on information systems and management science (ISMS), 2024

- **What's New**: 최근 소셜 미디어 플랫폼의 사용자 수가 증가함에 따라 온라인에서 극단적 발언(extreme speech)의 확산이 증가하고 있습니다. 이 연구는 기존 언어 모델이 중립적 텍스트와 비중립적 텍스트를 구분할 수 있는 능력을 보여주지만, 다양한 극단적 발언의 유형을 분류하는 것은 여전히 도전 과제로 남아 있다는 점을 강조합니다. 특히, 극단적 발언 분류는 사회-문화적 맥락을 깊이 이해해야 하는 복잡한 작업입니다.

- **Technical Details**: 이 연구에서는 Maronikolakis et al. (2022)의 극단적 발언 데이터셋의 인도 하위 집합을 활용하여 LLMs(대형 언어 모델)를 사용하는 효과적인 분류 프레임워크를 개발합니다. 오픈소스 Llama 모델과 클로즈드 소스 OpenAI 모델을 비교 평가한 결과, 사전 학습된 LLMs는 중간 정도의 효율성을 보였지만, 도메인 특정 데이터로의 파인 튜닝(fine-tuning)을 통해 성능을 크게 향상시킬 수 있음을 발견했습니다.

- **Performance Highlights**: GPT 기반 모델은 제로샷(zero-shot) 설정에서 Llama 모델보다 성능이 우수했으나, 파인 튜닝 후 성능 격차가 사라졌습니다. 이는 LLM이 언어적 및 맥락적 뉘앙스를 잘 반영할 수 있는 적응성을 지니고 있음을 보여줍니다. 이 연구는 극단적 발언 분류의 자동화 시스템 개발에 대한 가능성을 제시하고 있습니다.



### Confidence-Weighted Boundary-Aware Learning for Semi-Supervised Semantic Segmentation (https://arxiv.org/abs/2502.15152)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 논문에서는 세미-슈퍼바이즈드 시맨틱 분할(SSSS)을 위한 새로운 프레임워크인 CW-BASS를 제안합니다. 기존 SSSS 방법들이 직면한 문제, 예를 들어 coupling, confirmation bias 및 boundary blur를 해결하기 위해 다양한 기법을 통합하였습니다. 특히, 잘못된 예측의 영향을 줄이기 위해 pseudo-label에 신뢰도 가중치를 할당하고, 경계 인식 기술을 활용하여 정확도를 향상시킵니다.

- **Technical Details**: CW-BASS는 두 단계로 작동하며, 첫 번째 단계에서 교사 모델이 라벨이 없는 데이터에 대해 confidence score와 함께 pseudo-label을 생성합니다. 이어서 dynamic thresholding을 통해 낮은 신뢰도의 pseudo-label을 필터링하고, 두 번째 단계에서는 confidence decay 전략을 사용하여 낮은 신뢰도의 픽셀의 영향을 줄이며, boundary-aware 모듈을 통해 객체 경계 근처의 분할 정확도를 향상시킵니다. 이러한 과정은 모델 성능에 따라 지속적으로 조정됩니다.

- **Performance Highlights**: 실험 결과, 본 방법은 Pascal VOC 2012 및 Cityscapes 데이터셋을 사용하여 SOTA 성능을 달성하였습니다. 특히, 라벨된 데이터의 1/8 또는 12.5	eh%만 사용하더라도 Pascal VOC 2012에 대해 mIoU 75.81을 기록하며, 제한된 라벨 환경에서의 효과성을 입증하였습니다.



### Projection Optimization: A General Framework for Multi-Objective and Multi-Group RLHF (https://arxiv.org/abs/2502.15145)
- **What's New**: 이번 연구에서는 Multi-Objective Reinforcement Learning with Human Feedback (MORLHF) 접근법을 통해 각 목표에 대한 선호 피드백을 수집하고, 이를 통해 최적의 의사결정을 이끌어내는 방법을 제시합니다. 기존 연구와는 달리 선형 집계(linear aggregation) 방식을 벗어나 비선형 집계를 다루는 기존의 접근법의 한계를 극복하였습니다. 또한, 여러 그룹의 다양한 목표 가중치를 처리할 수 있는 확장된 프레임워크를 개발했습니다.

- **Technical Details**: 비선형 집계 문제를 해결하기 위해, 본 연구에서는 문제를 일련의 서브 문제(sub-problems)로 변환하였습니다. 각 서브 문제는 오직 선형 집계만을 포함하므로 계산적으로 효율적으로 해결할 수 있습니다. 이 프레임워크는 모든 그룹의 목표를 최대화하는 방법을 제공하여, 목표 간의 합의를 이끌어낼 수 있도록 돕습니다.

- **Performance Highlights**: 이론적으로, 제안된 알고리즘 프레임워크는 서브선형 유감(sublinear regret)을 달성할 수 있음을 입증하였습니다. 경험적으로, 개별 목표에 대한 최적 정책들이 얻어진 후, 거의 훈련이 필요 없는 알고리즘을 제시하여 효율성을 극대화할 수 있음을 보여줍니다.



### Chain-of-Rank: Enhancing Large Language Models for Domain-Specific RAG in Edge Devic (https://arxiv.org/abs/2502.15134)
Comments:
          NAACL 2025 (Findings)

- **What's New**: 이 논문에서는 Retrieval-augmented generation (RAG)와 Large Language Models (LLMs)의 통합이 중요한 진전을 이루었음을 강조합니다. 특히, 도메인 특화 RAG를 통해 LLM이 특정 도메인에서의 학습을 통해 더 정확하게 동작할 수 있도록 하는 방법을 제안하고 있습니다. 이를 통해 자원이 제한된 환경에서도 LLM이 신뢰성 있는 작업을 수행할 수 있도록 하고자 합니다.

- **Technical Details**: 주요 기술적 내용으로는 Chain of Rank (CoR) 방법이 제안되었습니다. CoR는 이전의 복잡한 추론과정에서 벗어나 입력 외부 문서의 신뢰도를 단순히 순위화하여 계산 복잡성을 줄입니다. 이는 LLM이 기계적으로 중요한 정보에 집중할 수 있게 하여 최종 답변의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 논문에서 제안한 CoR 방법은 벤치마크에서 최첨단(SOTA) 성능을 달성했습니다. 이는 계산 리소스가 제한된 환경에서도 도메인 특정 작업을 효과적으로 수행할 수 있도록 해주며, 리소스 효율성과 정확도의 균형을 잘 이루고 있습니다.



### CoT-ICL Lab: A Petri Dish for Studying Chain-of-Thought Learning from In-Context Demonstrations (https://arxiv.org/abs/2502.15132)
Comments:
          22 pages, 27 figures, 3 tables

- **What's New**: CoT-ICL Lab는 언어 모델에서 체인-오브-스Thought(Chain-of-Thought, CoT)와 인-컨텍스트 학습(In-Context Learning, ICL)을 연구하기 위한 새로운 프레임워크이자 방법론입니다. 이 연구는 토큰화된 합성 데이터 세트를 생성하고, 인-컨텍스트 예제의 복잡성에 대한 세밀한 제어를 제공합니다. 기존의 연구와는 달리, CoT-ICL Lab는 입력과 체인 토큰을 이산 토큰 공간에서 처리하여 자연어 처리와 밀접하게 연결됩니다.

- **Technical Details**: CoT-ICL Lab는 인과 구조(causal structure)와 토큰 처리 함수(token processing functions)를 분리하여 복잡한 문제를 처리할 수 있는 유연성을 제공합니다. 이 프레임워크는 방향 비순환 그래프(Directed Acyclic Graph, DAG)를 사용하여 체인 생성을 조정하고, 다양한 수준의 토큰 변환을 위해 다층 퍼셉트론(Multi-Layer Perceptron, MLP)을 활용합니다. 또한, 연구자들이 복잡한 문제를 더 잘 이해할 수 있도록 다양한 구성 요소를 조정할 수 있도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, CoT가 있는 인-컨텍스트 학습은 모델 성능의 전환을 가속화하는 데 기여함을 보여주었습니다. 특히, 모델의 깊이가 적은 예제에서 CoT의 활용에 중요한 것으로 나타났고, 더 많은 예제가 얕은 모델이 깊은 모델의 성능을 따라가는 데 도움이 되었습니다. 이를 통해 CoT-ICL Lab가 언어 모델에서 ICL과 CoT에 대한 이론적 및 경험적 통찰력을 제공하는 단순하면서도 강력한 테스트베드로 작용함을 강조했습니다.



### Unveiling Reasoning Thresholds in Language Models: Scaling, Fine-Tuning, and Interpretability through Attention Maps (https://arxiv.org/abs/2502.15120)
- **What's New**: 이번 연구는 다양한 크기와 훈련 데이터를 가진 decoder-only transformer 기반 언어 모델의 in-context learning (ICL) 능력을 조사합니다. 특히, 16억 개 이상의 매개변수를 가진 모델이 Commonsense Reasoning과 Deductive Reasoning 같은 작업에서 상당히 향상된 성능을 나타내는 중요한 매개변수 임계값을 식별했습니다. 이러한 연구는 CoT (chain-of-thought) prompting 기법에서 중요한 발전을 보여주며, 모델의 크기가 성능에 미치는 영향을 강조합니다.

- **Technical Details**: 모델의 추론 능력은 다양한 크기의 transformer 구조에서 분석되었으며, 특히 10억 개 이하의 매개변수를 가진 모델을 중심으로 새로운 발견이 이루어졌습니다. 연구에서는 CommonsenseQA 및 PrOntoQA-OOD 데이터셋을 사용하여 모델의 성능을 평가하였으며, 1.6억 개의 매개변수를 가진 모델이 더 나은 성공률을 보이는 것을 정량적으로 증명하였습니다. Fine-tuning을 통한 업무별 예시로 저성능 모델의 추론 성능을 향상시킬 수 있다는 점도 강조되었습니다.

- **Performance Highlights**: 연구 결과, 16억 개의 매개변수를 초과하는 모델들이 더 긴 추론 체인이 필요한 작업에서 특히 뛰어난 성능을 보여준다고 나타났습니다. 모델의 attention map 분석을 통해 올바른 CoT를 생성할 수 있는 모델은 후속 토큰과 올바른 품사의 높은 token-level attention 점수를 보여줍니다. 이 연구는 매개변수 임계값을 식별하고 sub-threshold 모델의 성능을 개선하는 방법을 제안하며, 다양한 크기의 모델 간 성능 차이에 대한 통찰력을 제공합니다.



### CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models (https://arxiv.org/abs/2502.15119)
- **What's New**: 이번 연구에서는 CurricuVLM이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Vision-Language Models (VLMs)를 활용하여 자율주행 에이전트를 위한 개인화된 커리큘럼 학습을 가능하게 합니다. 또한, VLM의 다중 모달 이해 능력을 측정하여 에이전트의 행동을 분석하고, 그에 따라 맞춤형 훈련 시나리오를 동적으로 생성합니다. 이 접근법은 기존의 자율주행 시스템에 안전성을 높이는 혁신적인 방법으로 주목받고 있습니다.

- **Technical Details**: CurricuVLM 프레임워크는 에이전트의 다양한 운전 상황에서의 행동을 지속적으로 모니터링하여, VLMs를 사용해 운전 상황을 이해하고 분석합니다. 에이전트의 현재 능력과 한계를 평가하기 위해 전문적인 GPT-4o 기반 분석기를 활용하여, 종합적인 분석을 수행합니다. 이 분석을 바탕으로 에이전트의 발전하는 능력에 맞춘 개인화된 훈련 시나리오가 동적으로 생성됩니다. 이는 에이전트가 지속적으로 성장할 수 있도록 지원합니다.

- **Performance Highlights**: Waymo Open Motion Dataset에서 실시된 광범위한 실험을 통해, CurricuVLM은 일반적인 상황 및 안전기반 상황 모두에서 최신 방법론들을 능가하는 성능을 보여주었습니다. 내비게이션 성공률, 운전 효율성 및 안전 지표에서 우수한 결과를 나타냄으로써, 다양한 강화학습 (RL) 알고리즘과의 호환성 역시 인정받았습니다. 이 연구는 자율주행 시스템을 개선하기 위한 일반적인 프레임워크로서의 잠재력을 보이고 있습니다.



### Assessing a Single Student's Concentration on Learning Platforms: A Machine Learning-Enhanced EEG-Based Framework (https://arxiv.org/abs/2502.15107)
- **What's New**: 이 연구는 온라인 학습 세션 중 학생의 집중 상태를 분류하기 위해 맞춤 설계된 기계 학습 모델을 사용하는 특화된 파이프라인을 소개합니다. EEG 데이터의 수집 및 전처리 프로토콜과 함께 5개의 EEG 신호 밴드에서 50개의 통계적 특성을 추출하는 방법이 상세히 설명되어 있습니다. 하이퍼파라미터 튜닝을 통해 학생의 집중 상태 분류 정확도를 향상시키는 이점을 탐구하며, VR 환경에서도 높은 정확성을 달성했습니다.

- **Technical Details**: 이 연구에서는 Muse S(Gen 2)라는 경량의 상용 EEG 센서를 사용하여 EEG 신호를 수집하였습니다. 신호는 Delta, Theta, Alpha, Beta, Gamma의 5개 주파수 밴드로 나뉘어 각 특성에서 평균, 분산, 표준편차 등의 통계적 특징을 추출하였습니다. Random Forest 모델을 이용하여 VR 및 비 VR 환경에서의 학생의 집중 상태 예측을 수행하였으며, 모델 성능 개선을 위한 피쳐 선택 및 하이퍼파라미터 최적화를 진행했습니다.

- **Performance Highlights**: 실험 결과, 컴퓨터 기반 학습 설정에서는 97.6%, 가상 현실 설정에서는 98%의 테스트 정확도를 기록하였습니다. 이는 온라인 교육 활동 중 학생의 집중에 대한 개인화된 통찰력을 제공하는 접근법의 효과를 강조합니다. 이러한 높은 정확도는 EEG 데이터와 기계 학습 기술의 결합이 어떻게 인지 상태 분류에 기여할 수 있는지를 보여줍니다.



### Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans (https://arxiv.org/abs/2502.15090)
- **What's New**: 이 연구에서는 현대의 대형 언어 모델(LLMs)이 인간의 표현과 어떻게 잘 일치하는지를 조사하는 새로운 접근 방식을 제시합니다. 활성화 조작(activation steering)에서의 방법을 채택하여 특정 개념(예: '고양이')에 대해 책임이 있는 뉴런을 식별하고 이에 따른 활성화 패턴을 분석합니다.

- **Technical Details**: 연구의 결과는 LLM의 표현이 행동 데이터를 통해 유추된 인간의 표현과 밀접하게 일치함을 보여줍니다. 이 정렬은 이전 연구에서 인간과 모델 간의 정렬에 초점을 맞춘 단어 임베딩(word embeddings)보다 더 뛰어납니다. 또한, LLM이 개념을 해석 가능한 인간 중심의 계층적 관계(e.g., '동물'-'개')로 조직하는 방식을 보여줍니다.

- **Performance Highlights**: 이번 접근 방식은 LLM이 개념을 표현하는 방식을 더 세부적으로 관찰할 수 있게 해줍니다. 연구 결과는 LLM이 인류의 이해를 반영하여 개념을 정리하는 능력이 뛰어난 것을 강조하고 있습니다.



### UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning (https://arxiv.org/abs/2502.15082)
Comments:
          Code: this https URL

- **What's New**: 본 연구에서는 사전 학습된 대형 언어 모델(LLM)에서 특정 정보를 삭제하는 과정 동안 모델의 다른 능력을 보존할 수 있는 효율적인 기술을 개발하고자 합니다. 이를 위해 데이터 선택을 통해 모델의 성능 저하를 최소화하는 UPCORE (Utility-Preserving Coreset Selection)라는 방법론을 제안합니다. UPCORE는 삭제 성과(deletion efficacy)와 모델 보존(model preservation) 간의 균형을 최적화하는 데 중점을 두고 있습니다.

- **Technical Details**: UPCORE는 비슷한 정보를 포함한 데이터 포인트를 식별하고 정리하여 Forget Set 내에서의 분산(variance)을 줄인다는 혁신적 접근법을 적용합니다. 이를 위해, 고립 숲(Isolation Forest) 알고리즘을 사용하여 이상 점을 찾아내고 이를 제거함으로써 전체 Forget Set의 분산을 줄입니다. 이러한 과정은 모델의 기능 저하를 최소화하면서 효율적으로 데이터 삭제를 수행할 수 있도록 합니다.

- **Performance Highlights**: UPCORE는 세 가지 표준 머신 언러닝 방법에 대해 평가되었으며, 항상 가장 높은 AUC (Area Under Curve)를 달성하여 성능이 우수함을 입증하였습니다. 또한, UPCORE는 비정상적인 데이터 포인트에 대한 긍정적인 전이 긍정적 효과를 극대화하여 삭제가 필요한 정보를 효과적으로 제거함과 동시에, 모델의 다른 능력에 대한 저하를 최소화했습니다. 결과적으로 UPCORE는 다양한 우선순위 간의 거래에서 최상의 결과를 보이며, 랜덤 선택 대비 더 나은 지표를 보이고 있습니다.



### Can Hallucination Correction Improve Video-Language Alignment? (https://arxiv.org/abs/2502.15079)
- **What's New**: 본 논문에서는 비디오와 텍스트 간의 정렬을 개선하기 위한 새로운 접근법으로 HACA(하위 오류 수정 기반의 비디오-언어 정렬)를 도입하였습니다. 이 방법은 비디오의 내용과 일치하지 않는 설명의 오류를 수정하는 자기 학습(self-training) 프레임워크를 활용합니다. 이를 통해 모델은 비디오와 텍스트의 조화로운 표현을 강화할 수 있습니다.

- **Technical Details**: HACA는 고전적인 엔탤먼트(entailment) 학습 기법을 넘어서, 텍스트와 비디오 간의 불일치를 예측하고 이를 수정하는 임무를 통해 정렬을 최적화합니다. 이 모델은 비디오 특성에 맞춰 데이터를 증강하기 위해 마스킹 수정 작업을 도입하여, 비디오-언어 모델의 학습을 보다 세밀하게 수행합니다. 기술적으로는 비디오-LLM의 블록을 최적화하고, 텍스트 디코더와 어댑터를 조정함으로써 전반적인 성능을 높이고 있습니다.

- **Performance Highlights**: HACA를 통해 최적화된 모델은 기본 모델 대비 최대 17.9%의 정확도 향상을 보이며, 5.7 mAP 포인트 이상을 달성하였습니다. 이러한 결과는 HACA가 비디오-텍스트 간의 정렬을 효과적으로 개선하며, 다양한 다운스트림 우선 과제에서도 뛰어난 성능을 발휘함을 보여줍니다.



### Hardware-Friendly Static Quantization Method for Video Diffusion Transformers (https://arxiv.org/abs/2502.15077)
- **What's New**: 이 논문에서는 OpenSora라는 비디오 확산 변환기(Video Diffusion Transformer)의 사후학습 양자화(post-training quantization) 방법을 제안합니다. 기존의 동적 양자화(dynamical quantization) 기술에 의존하지 않고, 정적 양자화(static quantization)을 통해 모델을 효율적으로 배포할 수 있습니다. 특히, CLIP와 VQA 지표를 사용하여 FP16 및 동적 양자화된 ViDiT-Q 방법과 유사한 비디오 품질을 달성했습니다.

- **Technical Details**: 제안된 방법은 단계별 보정(calibration data)을 사용하여 각 시간 단계에 대해서 적절한 정적 양자화 모델을 제공합니다. 이 과정에서는 가중치에 대한 채널 기반 양자화(channel-wise quantization)와 활성에 대한 텐서 기반 양자화(tensor-wise quantization)를 적용합니다. 또한, 부드러운 양자화(smooth quantization) 기법을 통해 정적 양자화된 모델에서도 고품질 비디오 출력을 얻을 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과는 정적 양자화가 동적 양자화에 대한 실제적인 대안이 될 수 있음을 입증하고 있으며, 성능을 저하시키지 않으면서도 더욱 효율적인 접근 방식을 제공합니다. 특히, CW, TW 및 ASQ를 사용한 집계 정적 양자화 방법은 동적 양자화 및 FP16 모델과 시각적 품질이 유사하다는 것을 나타냅니다. 제안된 방법은 다양한 정밀도 수준에서 뛰어난 결과를 보여주며, 생성된 비디오와의 정합성 또한 더욱 향상되었습니다.



### Rare Disease Differential Diagnosis with Large Language Models at Scale: From Abdominal Actinomycosis to Wilson's Diseas (https://arxiv.org/abs/2502.15069)
- **What's New**: 이번 논문에서는 RareScale이라는 새로운 시스템을 제안했습니다. 이 시스템은 LLM(대형 언어 모델)과 전문가 시스템을 결합하여 희귀 질환에 대한 진단 정확도를 높이고자 합니다. 이를 통해 의료 환경에서 희귀 질환 감별 진단을 개선하고자 하며, 575개의 희귀 질환을 대상으로 한 결과를 보고합니다.

- **Technical Details**: RareScale은 전문가 시스템과 LLM을 동시에 활용하여 희귀 질환에 대한 대화 시뮬레이션을 진행합니다. 이러한 시뮬레이션 데이터는 희귀 질환 후보 예측 모델 트레이닝에 사용되며, 이 모델에서 생성된 후보들은 블랙박스 LLM에 추가 입력으로 제공됩니다. 이를 기반으로 할 경우, RareScale은 흔한 진단과 희귀 진단 간의 균형을 맞출 수 있도록 설계되었습니다.

- **Performance Highlights**: RareScale의 접근 방식은 기존 블랙박스 LLM의 성능을 17% 이상 향상시켰으며, Top-5 정확도가 56.8%에서 74.1%로 증가했습니다. 이러한 성과는 희귀 질환의 감별 진단 과정에서 LLM과 전문가 지식을 통합함으로써 달성된 결과입니다. 궁극적으로 RareScale은 희귀 질환을 더 효과적으로 식별할 수 있는 가능성을 보여줍니다.



### Fundamental Survey on Neuromorphic Based Audio Classification (https://arxiv.org/abs/2502.15056)
Comments:
          24 Pages, 1 Table

- **What's New**: 본 논문에서는 생체 신경망에서 영감을 받은 신경형 컴퓨팅(neuromorphic computing)을 활용한 오디오 분류(audio classification)의 최신 동향을 포괄적으로 조사하고 있습니다. 전통적인 오디오 분류 방법의 복잡성을 극복할 수 있는 새로운 대안으로서, Spiking Neural Networks (SNNs)와 같은 신경망 모델과 memristors 등을 조명하고 있습니다.

- **Technical Details**: 신경형 시스템의 주요 구성 요소와 이점을 살펴보며, 특히 이벤트 기반 처리(event-based processing), 스파이크 기반 학습(spike-based learning), 생물학적으로 영감을 받은 특징 추출(bio-inspired feature extraction)과 같은 다양한 방법론을 다룹니다. 이러한 기술들이 전통적인 방법의 제한 사항인 에너지 효율성, 실시간 처리 및 환경 소음에 대한 강건성에 어떻게 접근하는지를 설명합니다.

- **Performance Highlights**: 본 논문은 여러 신경형 오디오 분류 모델과 벤치마크의 비교 분석을 통해 성능 지표, 계산 효율성, 확장성 등을 평가합니다. 이를 통해 연구자들과 엔지니어, 실무자들이 신경형 오디오 분류 분야에서 혁신과 발전을 촉진하도록 돕는 포괄적인 지침을 제공합니다.



### Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation (https://arxiv.org/abs/2502.15040)
Comments:
          GenAI4Health - AAAI '25

- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 향상을 위해 Visual RAG (V-RAG) 프레임워크를 제안합니다. V-RAG는 텍스트와 시각 데이터 모두를 활용하여 정보의 정확성을 향상시키기 위해 노력합니다. 특히, 의료 분야에 중점을 두어 MIMIC-CXR 및 Multicare 데이터셋에서 성능을 검증하였고, 환자 진단의 정확도를 높이는 데 기여하고 있습니다.

- **Technical Details**: V-RAG는 텍스트 기반 RAG의 한계를 극복하고 이미지를 포함한 다중 데이터를 활용하여 더 정확한 응답을 제공합니다. 이를 위해 오픈북(Open Book) 성능을 강화하며, 제시된 멀티 이미지 처리 기술을 통해 모델의 이해도를 향상시킵니다. 결국, V-RAG는 단일 이미지 데이터로 훈련된 Model이 다중 이미지 데이터 또한 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: V-RAG를 적용한 결과, 의료 이미지 및 X-ray 보고서의 생성에서 hallucination 문제를 보다 효과적으로 해결할 수 있음을 보여주었습니다. 시험 결과, RadGraph-F1 Score의 향상 또한 확인되었으며, 이는 진단의 정확성 및 신뢰성을 높이는 데 기여하고 있습니다. 이런 성과는 빈번한 의료 개체뿐만 아니라 드문 개체에 대해서도 긍정적인 효과를 입증합니다.



### DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Tim (https://arxiv.org/abs/2502.15037)
- **What's New**: 이번 논문에서는 복잡한 가지형 케이블을 정확하게 예측하고 조작하는 로봇을 위한 새로운 프레임워크인 DEFT(Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time)를 소개합니다. 기존의 단일 형태의 Deformable Linear Objects(DLOs) 모델을 확장하여 Branched Deformable Linear Objects(BDLOs)에 대한 동적인 전파와 조작을 모델링하는 데 중점을 두고 있습니다. 특히, junction point에서의 힘 상호 작용과 변형 전파 패턴을 모델링하는 데 있어 기존 접근법의 한계를 극복하는 방법을 제시하고 있습니다.

- **Technical Details**: DEFT 프레임워크는 물리 기반 모델과 머신 러닝 프레임워크를 결합해 BDLO의 동적인 거동을 실시간으로 모델링합니다. 이 모델은 junction points에서의 동적 전파와 중간에서의 그랩(grasp)을 정확하게 모사하며, 실시간 추론을 위한 효율적인 계산 방법을 제공합니다. 또한, 그래프 신경망(Graph Neural Network)을 통해 BDLO의 남은 학습(residual learning)을 수행하여 장기 예측에서 발생할 수 있는 수치적 통합 오류를 보완합니다.

- **Performance Highlights**: DEFT는 정확도, 계산 속도 및 일반화 가능성 면에서 기존의 최첨단 모델들과 비교해 뛰어난 성능을 보입니다. 실험 결과, DEFT는 3D 형상 일치, 다중 스레드 조작 및 정밀한 스레드 그랩 및 삽입과 같은 복잡한 와이어 하니스 조립 작업을 성공적으로 수행할 수 있음을 입증하였습니다. PyTorch에서 구현된 이 프레임워크는 딥러닝 워크플로우와의 원활한 통합을 지원하며, 제조 자동화의 중요한 과제를 해결하는 데 기여하고 있습니다.



### InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback (https://arxiv.org/abs/2502.15027)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문에서는 Large Multimodal Models (LMMs)의 인간 피드백과의 상호작용 지능을 평가하기 위한 새로운 프레임워크인 InterFeedback를 설계하였습니다. 이는 LMM과 데이터셋에 적용 가능하며, InterFeedback-Bench를 통해 다양한 오픈소스 LMM을 평가합니다. 추가적으로, OpenAI-o1 및 Claude-3.5-Sonnet와 같은 주요 모델의 상호작용 성능을 수동으로 테스트하기 위해 새롭게 수집한 120개의 사례를 포함한 InterFeedback-Human 데이터셋도 제공됩니다.

- **Technical Details**: InterFeedback 프레임워크는 LMM이 상호작용적으로 문제를 해결할 수 있도록 설계되었으며, 문제 해결 능력과 피드백 해석 능력을 평가하는 기준점인 InterFeedback-Bench를 도입합니다. 이 연구에서는 MMMU-Pro와 MathVerse라는 두 가지 도전적인 데이터셋을 활용하여 다양한 LMM의 상호작용 지능을 평가하였습니다. 최종적으로, LMM이 피드백을 통해 성능을 개선할 수 있는지에 대한 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, OpenAI-o1 등 최첨단 LMM조차도 인간의 피드백을 통해 50% 미만으로 결과를 수정하는 데 그쳤습니다. LMM은 높은 품질의 피드백을 요구하며, 저품질 피드백은 성능을 오히려 저하시킬 수 있다는 사실을 발견하였습니다. 이러한 결과는 LMM의 피드백 해석 및 반영 능력을 향상시킬 필요성을 강조합니다.



### Towards Physics-Guided Foundation Models (https://arxiv.org/abs/2502.15013)
- **What's New**: 이 논문에서는 물리학에 기반한 기초 모델(Physics-Guided Foundation Models, PGFM)을 제안하고 있습니다. 전통적인 기초 모델들이 데이터 기반 접근법에 한정되어 있어 물리적 원칙을 무시하고 비현실적인 출력을 생성하는 문제를 해결하고자 합니다. PGFM은 과학적 법칙과 물리적 제약을 통합하여 예측의 신뢰성과 견고성을 향상시킵니다.

- **Technical Details**: PGFM은 폭넓은 분야의 지식과 기본 개념을 통합하여 기초 모델을 개선합니다. 물리적 지식을 기초 모델에 포함시키기 위한 방법으로는 물리적 제약 학습과 아키텍처 수준 통합이 있습니다. 이러한 접근 방식은 모델이 신뢰할 수 있는 예측을 하도록 돕고, 다양한 다운스트림 작업에서의 성능을 향상시킵니다.

- **Performance Highlights**: PGFM은 다양한 응용 분야에서 성능, 견고성 및 해석 가능성을 개선하는 데 기여합니다. 향후 연구에서는 PGFM 모델을 개발하고 성능 샘플 복잡성 지표를 활용하여 도메인 지식 통합을 평가할 계획입니다. PGFM의 발전은 AI의 다양한 분야에서 가능성을 열어줄 것으로 기대됩니다.



### Graph in the Vault: Protecting Edge GNN Inference with Trusted Execution Environmen (https://arxiv.org/abs/2502.15012)
Comments:
          This work is accepted by DAC 2025

- **What's New**: GNNVault는 Trusted Execution Environment(TEE)를 기반으로 하는 최초의 안전한 Graph Neural Network(GNN) 배포 전략으로, 기밀 GNN 모델 파라미터와 개인 그래프를 안전하게 보호합니다. 이 방법은 'partition-before-training' 설계를 따르며, 공공 모델과 개인 GNN 정류기를 조합하여 사용합니다. 이를 통해 GNN 추론 중 발생할 수 있는 링크 도용 공격에 대한 보호를 제공하며, 정확도 저하는 2% 미만입니다.

- **Technical Details**: GNNVault는 개인 GNN 정류기와 공공 백본 모델 간의 다양한 커뮤니케이션 방식을 제안합니다. 공공 백본 모델은 대체 그래프를 사용하여 학습된 컴퓨테이션 집약적인 모델로, 불신 환경에서 배포됩니다. 개인 GNN 정류기는 실제 개인 그래프 구조를 활용하여 백본의 노드 임베딩을 수정하며 작고 TEE 내에서 작동합니다. 이러한 방식으로 GNNVault는 각종 사적 데이터와 기밀 모델 파라미터를 보호하면서도 충분한 성능을 유지합니다.

- **Performance Highlights**: GNNVault의 실제 구현은 Intel SGX를 통해 이루어졌으며, 실험 결과 인퍼런스 정확도가 높고 기밀한 모델 파라미터와 엣지 데이터가 보호됨을 보여주었습니다. 이 방식은 개인 데이터를 노출하지 않으면서도 실시간 응답과 개인 정보 보호가 필요한 엣지 AI 응용 프로그램에 적합합니다. GNNVault 도입으로 인한 인퍼런스 오버헤드는 적고, 정확도 저하는 2% 미만으로 유지되며 유효성을 증명한 것으로 평가됩니다.



### Obliviate: Efficient Unmemorization for Protecting Intellectual Property in Large Language Models (https://arxiv.org/abs/2502.15010)
- **What's New**: 최근 AI 기업들과 콘텐츠 제작자 사이의 저작권 협약으로 인해 언어 모델의 저작권 콘텐츠 재현 능력에 대한 정밀한 제어 필요성이 강조되었습니다. 우리는 Obliviate라는 새로운 포스트 트레이닝(post-training) 기법을 제안하며, 이 방법은 특정 텍스트의 정확한 복제를 선택적으로 방지하면서 의미론적 이해를 유지합니다. Obliviate는 기억된 시퀀스에서 특정 토큰을 선택하고 모델의 확률 분포를 수정하여 정확한 복제를 방지하는 방식으로 작동합니다.

- **Technical Details**: Obliviate는 사전 훈련된 언어 모델에서 특정 시퀀스를 비기억하도록 선택적으로 수정하는 기술입니다. 이 기술은 Kullback-Leibler (KL) 발산 손실을 사용하여 목표 토큰이 재현될 확률을 줄이면서 유창성과 일관성을 유지하도록 조정합니다. 비목표 토큰에 대해서도 KL 발산을 적용하여 상위 k 토큰 분포의 일관성을 유지하여 모델의 전반적인 성능을 보존합니다.

- **Performance Highlights**: Obliviate는 여러 대형 언어 모델(LLaMA-3.1 8B, Qwen-2.5-7B 등)에 대해 평가된 결과, 100배의 저작권 콘텐츠의 Verbatim memorization 감소를 보여주면서도 표준 벤치마크에서 성능 저하가 1% 이내로 유지되었습니다. 이 결과는 Obliviate가 선택적으로 목표 콘텐츠를 비기억하면서 모델 성능을 유지하는 데 효과적임을 강조합니다.



### LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers (https://arxiv.org/abs/2502.15007)
Comments:
          accepted to NAACL 2025

- **What's New**: 이번 논문에서는 Large Language Models(LLMs)가 맥락 정보를 인코딩하고 저장하는 방법을 정량화하는 새로운 방법을 소개합니다. 연구팀은 종종 사소하게 여겨지는 토큰들, 예를 들어 정관사, 구두점 등이 비정상적으로 높은 맥락 정보를 지닌다는 사실을 발견했습니다. 이러한 발견은 LLM에서 덜 중요하게 보이는 토큰들이 맥락 유지를 위해 얼마나 중요한지를 강조하고 있습니다.

- **Technical Details**: LLM-Microscope라는 오픈소스 툴킷은 토큰 수준의 비선형성(nonlinearity), 맥락 기억(contextual memory), 중간 층의 기여도 분석을 시각화할 수 있는 기능을 제공합니다. 이 프레임워크는 LLM의 내부 행동을 분석하기 위한 종합적인 방법론을 제공하며, 각 층의 임베딩 간의 비선형 전환을 단일 선형 매핑으로 근사할 수 있는 정도를 정량화합니다. 연구의 주요 분석 방법론으로는 맥락화 평가(contextualization assessment)와 토큰 수준 비선형성 측정이 포함됩니다.

- **Performance Highlights**: 연구진은 맥락 정보가 포함된 토큰을 제거하면 MMLU와 BABILong-4k와 같은 특정 작업에서 성능이 일관되게 저하된다는 점을 강조했습니다. 이러한 성능 저하는 강력한 언어 모델인 GPT-4o에 의해 불필요하다고 판단된 토큰들을 제거할 때도 지속적으로 나타났습니다. LLM-Microscope는 연구자들이 다양한 언어 모델의 내부 구조를 쉽게 분석할 수 있게 해줍니다.



### Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions (https://arxiv.org/abs/2502.15006)
- **What's New**: 본 논문에서는 모델 예측 제어(MPC)를 향상시키기 위한 새로운 접근법인 Neural Shield-VIMPC (NS-VIMPC) 컨트롤러를 제안합니다. 특히 이 방법은 안전 요구 사항을 예측 지평선 넘어까지 고려하여 제어 불변 집합 제약을 강화합니다. 이전의 연구와 달리 제안된 방법은 일반 비선형 동역학에 적용 가능하며, 실시간으로 계획을 수립할 수 있는 능력을 개선합니다. 이로써 기존의 샘플 기반 MPC 컨트롤러보다 안전성을 크게 향상시킵니다.

- **Technical Details**: 이 연구에서는 정책 신경 제어 장벽 함수(Policy Neural Control Barrier Functions, PNCBF)를 이산 시간(Discrete Time) 환경으로 확장하고, 이를 이용해 제어 위반 없이 최적의 제어를 찾는 접근법을 개발하였습니다. 또한, 새로운 샘플링 전략인 Resampling-Based Rollout (RBR)을 제공하여 샘플 효율성을 향상시켰습니다. 이러한 기법은 CPU에서의 실시간 계획을 가능하게 하며, 기존 제어 이론의 한계를 극복하는 데 기여합니다. VIMPC의 새로운 변형은 실험적으로 검증되었습니다.

- **Performance Highlights**: 시뮬레이션 및 실제 하드웨어 실험에서 NS-VIMPC는 기존의 MPC 기반 안전 장치보다 안전성을 크게 향상시키는 것으로 나타났습니다. 특히, 악의적으로 설계된 비용 함수에서도 뛰어난 성능을 보여주었습니다. AutoRally 플랫폼을 이용한 실제 하드웨어 테스트에서도 NS-VIMPC는 비모델링된 동적 장애물에 강인한 특성을 발휘하여 우수성을 입증했습니다. 이러한 결과는 NS-VIMPC 방법이 비선형 제어에 있어 효과적임을 시사합니다.



### A Socratic RAG Approach to Connect Natural Language Queries on Research Topics with Knowledge Organization Systems (https://arxiv.org/abs/2502.15005)
Comments:
          6 pages, 2 figures, AAAI 2025 Workshop on A Translational Institute for Knowledge Axiomatization

- **What's New**: 이 논문에서는 자연어 질의(Natural Language Queries)를 연구 주제와 기계가 해석할 수 있는 의미적 엔티티(Semantic Entities)로 매핑하는 Retrieval Augmented Generation (RAG) 에이전트를 제안합니다. 이 접근법은 Socratic Dialogue와 RAG를 결합하여 사용자의 직관적인 연구 주제 이해를 구조화된 Knowledge Organization Systems (KOS)와 일치시킵니다. 제안된 방법은 복잡한 학술 분류를 더 쉽게 접근할 수 있도록 합니다.

- **Technical Details**: 논문에서는 기존 연구 주제를 정리하는 문제와 관련된 연구들을 간단히 개관한 후, CollabNext의 간단한 설명을 제공합니다. CollabNext는 사람들이 어느 주제에서 활동하는지를 탐색할 수 있도록 설계된 지식 그래프입니다. 이 애플리케이션은 공개적으로 접근할 수 있는 강력하고 구조화된 데이터에 의존하며, 연구 주제에서 사람과 조직 사이의 관계를 맺고 있습니다.

- **Performance Highlights**: CollabNext는 HBCUs(Historically Black Colleges and Universities)와 신진 연구자들의 가시성을 높이기 위해 의도적으로 설계되었습니다. 이 프로젝트는 연구 분류에서 발생하는 혼잡함과 이름 충돌 문제를 해결하기 위해 기존의 공개적인 의미적 구조를 활용합니다. 다양한 연구 영역에 대한 인사이트를 제공하여, 잠재적으로 넓은 범위로 개인 및 연구 공동체에 기여할 수 있는 가능성이 있습니다.



### A Rapid Test for Accuracy and Bias of Face Recognition Technology (https://arxiv.org/abs/2502.14996)
Comments:
          Accepted as a conference paper for WACV 2025. Manuel Knott, Ignacio Serna, and Ethan Mann contributed equally

- **What's New**: 본 연구에서는 얼굴 인식 시스템(FR)의 정확성을 측정하기 위한 새로운 방법을 제안합니다. 이 방법은 수동 주석 없이도 FR 시스템을 빠르게 기준 설정할 수 있는 방법으로, 웹 검색 결과에서 가져온 근사 레이블을 사용합니다. 또한, 다섯 가지 FR 클라우드 서비스의 첫 번째 공공 벤치마크를 소개하며, 인종적 편향, 특히 아시아 여성에 대한 낮은 정확성을 드러냅니다.

- **Technical Details**: 제안된 방법은 공공 데이터를 활용하여 검증 가능성을 높이며, 이미지를 실시간으로 분석하여 저장하지 않습니다. 기존의 인식 알고리즘의 신뢰도 값을 기반으로 얼굴 정체성의 기준 진실 레이블을 추론하는 알고리즘적 단계가 핵심입니다. 이 방법은 인공지능과 얼굴 인식 기술이 공정하고 투명한 기관으로 발전하는 데 기여할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 FR 시스템의 정확성을 신뢰성 있게 추정하고 순위를 매길 수 있으며, 이로 인해 수동 레이블링의 시간과 비용이 크게 줄어듭니다. 실험적으로 검증된 이 방법은 더욱 저렴하고 실용적인 관찰 방법으로, 동시 비교를 통해 다섯 가지 인기 있는 클라우드 서비스의 정확성과 편향을 비교할 수 있는 최초의 공공 벤치마크를 제공합니다.



### Beyond No: Quantifying AI Over-Refusal and Emotional Attachment Boundaries (https://arxiv.org/abs/2502.14975)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 감정 경계 처리 능력을 평가하기 위한 오픈 소스 벤치마크와 평가 프레임워크를 제시합니다. 6개 언어로 구성된 1156개의 프롬프트를 사용하여 GPT-4o, Claude-3.5 Sonnet, Mistral-large의 세 가지 선두 모델의 성능을 분석하였습니다. 이를 통해 각 모델이 어떻게 감정 경계를 유지하는지를 정량적으로 평가하였습니다.

- **Technical Details**: 이 프레임워크는 직접 거부(direct refusal), 사과(apology), 설명(explanation), 돌리기(deflection), 인식(acknowledgment), 경계 설정(boundary setting), 감정 인식(emotional awareness) 등 7개의 주요 패턴을 기준으로 응답을 정량화합니다. 연구 결과, Claude-3.5가 최고 점수인 8.69/10을 기록하였으며, 평균 86.51 단어의 긴 응답을 생성하는 것을 확인했습니다. 또한 영어와 비영어 응답 간 성능의 격차가 상당함을 발견했습니다.

- **Performance Highlights**: 영어의 평균 점수는 25.62인 반면, 비영어 상호작용의 평균 점수는 0.22 미만으로 극심한 차이를 보였습니다. 영어 응답의 거부율은 43.20%로 비영어의 1% 미만에 비해 현저히 높았습니다. Mistral 모델은 돌리기 전략을 선호하는 경향을 보였으며, 모든 모델에서 공감 점수가 지속적으로 낮았습니다. 향후 연구에서는 보다 정교한 점수 산출 방법과 언어 범위 확대, 문화적 차이를 탐구하는 방향으로 나아가야 할 것입니다.



### CyberSentinel: An Emergent Threat Detection System for AI Security (https://arxiv.org/abs/2502.14966)
- **What's New**: 이 논문은 인공지능(AI)의 빠른 발전으로 인해 증가한 사이버 보안 위협에 대응하기 위한 적응형 방어 전략을 제시합니다. 특히, CyberSentinel이라는 단일 시스템을 도입하여 새롭게 발생하는 보안 위험을 실시간으로 탐지하고 완화하는 방법을 설명합니다.

- **Technical Details**: CyberSentinel은 세 가지 주요 기능을 통합합니다: (1) SSH 로그 분석을 통한 Brute-force 공격 탐지, (2) 도메인 블랙리스트와 휴리스틱 URL 점수를 활용한 Phishing 위협 평가, (3) 기계 학습 기반 이상 탐지를 통한 발생하는 위협 탐지. 이러한 기능들은 적대적 전술의 진화에 지속적으로 적응하여 사이버 보안을 강화합니다.

- **Performance Highlights**: CyberSentinel은 AI 보안의 주요 취약점을 해결하여 능동적 사이버 보안 방어를 강화합니다. 논문은 시스템의 효과성을 강조하며, 새로운 보안 위협에 대한 실시간 대응 능력이 중요하다고 주장합니다.



### KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding (https://arxiv.org/abs/2502.14949)
Comments:
          17 pages, 5 figures, ACL 2025

- **What's New**: 이 논문은 KITAB-Bench라는 포괄적인 아랍어 OCR 벤치마크를 소개합니다. 아랍어 OCR의 평가 시스템에서의 공백을 메우기 위해 8,809개의 샘플을 수집하여 9개의 주요 도메인과 36개의 하위 도메인으로 구성하였습니다. 이 벤치마크는 다양한 문서 유형을 포함하여, 손글씨 텍스트, 구조화된 표, 그리고 비즈니스 인텔리전스를 위한 21가지 차트 유형에 대한 특화된 내용을 제공합니다.

- **Technical Details**: KITAB-Bench는 레이아웃 탐지(text blocks, tables, figures), 다중 형식 인식(printed/handwritten text, charts, diagrams), 그리고 구조화된 출력 생성(HTML tables, DataFrame charts, markdown)을 평가하는 시스템을 채택합니다. 이 방법론은 OCR 성능을 정밀하게 평가할 수 있는 프레임워크를 제공하며, 챠트 추출(CharTeX)과 도표 추출(CODM) 평가 지표를 포함하여 다양한 문서 이해 과제를 다룹니다.

- **Performance Highlights**: 최신 비전-언어 모델(GPT-4, Gemini, Qwen)이 전통적인 OCR 방법(EasyOCR, PaddleOCR, Surya)에 비해 평균 60% 더 높은 Character Error Rate(CER)에서 우수성을 보였습니다. 그러나 현재 아랍어 OCR 모델의 한계로는 PDF에서 Markdown으로의 변환에서 가장 우수한 모델인 Gemini-2.0-Flash가 단 65%의 정확도만을 달성했습니다. 이러한 결과는 아랍어 텍스트 인식을 위한 복잡한 글꼴, 숫자 인식 오류, 단어 신장 및 표 구조 감지의 문제를 강조합니다.



### Reward-Guided Iterative Refinement in Diffusion Models at Test-Time with Applications to Protein and DNA Design (https://arxiv.org/abs/2502.14944)
Comments:
          Under review. If you have any suggestions/missing references, please let us know

- **What's New**: 이 논문에서는 확산 모델(diffusion models)과 관련해 새로운 테스트 시간 보상 최적화(test-time reward optimization) 프레임워크를 제안합니다. 이 프레임워크는 진화 알고리즘(evolutionary algorithms)에서 영감을 받았으며, 각 반복(iteration)에서 노이즈 처리와 보상 기반 디노이징(reward-guided denoising)을 포함하는 단계적 정제(iterative refinement)의 두 가지 단계를 활용합니다. 이를 통해 보상 최적화 과정에서 발생할 수 있는 오류를 점진적으로 수정할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 각 반복에서 파슬 노이징(partial noising)과 보상 기반 디노이징(reward-guided denoising)으로 구성되며, 이를 통해 복잡한 보상 함수(reward functions)를 최적화할 수 있습니다. 특히 생물학적 서열이나 분자 설계에서 자주 발생하는 하드 제약 조건(hard constraints)을 처리하는 데 효과적입니다. 이 프레임워크의 이론적 보장을 제공하며, 보상 함수와 사전 훈련(pre-trained) 분포의 통합을 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘이 단백질(protein) 및 DNA 설계에서 기존 방법보다 뛰어난 성능을 보인다는 것을 확인했습니다. 이 연구는 보상 기반 생성(generation) 과정에서의 반복적 접근 방식을 통해 출력 결과를 점진적으로 개선하는 방법을 제시하며, 이는 다양한 생물학적 응용 분야에 큰 의미를 가집니다. 코드도 제공되어 연구 결과의 재현이 용이합니다.



### FacaDiffy: Inpainting Unseen Facade Parts Using Diffusion Models (https://arxiv.org/abs/2502.14940)
Comments:
          Accepted for GeoSpatial Week 2025, ISPRS Annals

- **What's New**: FacaDiffy는 기존의 3D 건물 모델과 레이저 스캐닝 포인트 클라우드를 사용하여 2D conflict map을 생성하는 새로운 방법을 제시합니다. 이 모델은 개인화된 Stable Diffusion 모델을 통해 보이지 않는 파사드 부분을 채우는 방식으로 결함 있는 conflict map을 완성합니다. 또한, 임의의 도시 모델 생성기와 주석이 있는 파사드 이미지를 사용하여 합성 conflict maps를 생성하는 확장 가능한 파이프라인을 개발했습니다.

- **Technical Details**: FacaDiffy는 기존의 LoD 2 모델과 레이저 스캐닝 데이터를 통해 2D conflict map을 생성하는 결정론적 방법을 사용합니다. 이후, Stable Diffusion 모델을 개인화하여 이 2D conflict maps에 파사드 객체를 자동으로 추가하는 inpainting 작업을 수행합니다. 이 과정에는 또한 Dreambooth를 활용하여 합성된 conflict map 데이터를 기반으로 모델을 훈련하는 기술이 포함됩니다.

- **Performance Highlights**: FacaDiffy는 다양한 inpainting 기준선과 비교하여 conflict map 완성도에서 최첨단 성능을 발휘하며, 고해상도 3D 의미론적 건물 재구성 시 완성된 conflict map을 적용할 경우 탐지율이 22% 향상됩니다. 이러한 성과는 고급 재구성에 있어 FacaDiffy의 적용 가능성을 더욱 확대합니다.



### Online hand gesture recognition using Continual Graph Transformers (https://arxiv.org/abs/2502.14939)
- **What's New**: 이 논문은 실시간 손 제스처 인식을 위한 새로운 온라인 인식 시스템을 제안합니다. 기존의 분할 기반 인식 방식에서 벗어나, 스켈레톤(sequence of 3D coordinates) 데이터의 실시간 스트리밍을 처리할 수 있는 방법을 제시합니다. 이 시스템은 Spatial Graph Convolutional Networks (S-GCN)과 Transformer 기반의 Graph Encoder (TGE)를 결합하여 공간적 및 시간적 특성을 효과적으로 추출합니다.

- **Technical Details**: 본 연구에서는 Hybrid architecture를 통해 공간적 특성을 S-GCN으로 추출하고, 프레임 간의 시간적 의존성을 TGE를 통해 캡처합니다. 또한, 모델이 진화하는 데이터 분포에 적응할 수 있도록 지속적인 학습 메커니즘(continual learning mechanism)을 도입하여 동적 환경에서의 인식을 강화합니다. 실험은 SHREC'21 벤치마크 데이터셋에서 수행되었으며, 최첨단 성능을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 온라인 손 제스처 인식에서 뛰어난 성능을 발휘하며, 기존 기법들보다 높은 정확도를 달성하였습니다. 또한, 잘못된 긍정 비율(false positive rates)을 상당히 줄어들게 하여, 실시간 응용에 적합한 솔루션으로 자리 잡고 있습니다. 이 시스템은 인간 로봇 협업이나 보조 기술 등 다양한 분야에 통합될 수 있습니다.



### Fast and Accurate Blind Flexible Docking (https://arxiv.org/abs/2502.14934)
Comments:
          25 pages, Accepted by ICLR 2025

- **What's New**: FABFlex는 단백질의 유연성과 결합 부위가 알려지지 않은 상황에서, 빠르고 정확한 다중 작업 학습 모델로 설계되었습니다. 이 모델은 3개 모듈로 구성되어 있으며, 결합 포켓 예측, 리간드 도킹, 포켓 도킹을 통해 실질적인 '블라인드 유연 도킹'(blind flexible docking) 시나리오의 도전 과제를 해결합니다. FABFlex는 리간드와 포켓 도킹 모듈 간의 지속적인 구조 개선 가능성을 제공하는 반복 업데이트 메커니즘을 포함하여 통합적이고 일관된 프레임워크를 구성합니다.

- **Technical Details**: FABFlex는 E(3)-등가 그래프 신경망 레이어인 'FABind 레이어'를 활용해 각 모듈을 설계하였으며, 이는 리간드-단백질 이종 그래프를 처리할 수 있도록 조정되었습니다. 포켓 예측 모듈에서 시작하여, 예측된 포켓 정보를 활용하여 각각의 리간드와 포켓의 홀로 구조를 예측합니다. 각 모듈은 상호 작용하며 기능적으로 통합되어 있으며, 동시에 리간드와 포켓의 홀로 구조를 단일 작업으로 예측하게 됩니다.

- **Performance Highlights**: FABFlex는 PDBBind 공개 벤치마크에서 기존의 다양한 도킹 방법들과 비교하여 리간드 RMSD가 2Å 이하인 비율을 40.59%로 증가시키고, 포켓 RMSD를 1.10Å까지 낮추는 성과를 보였습니다. 특히, FABFlex는 최근의 최첨단 유연 도킹 방법인 DynamicBind보다 약 208배 빠른 계산 속도를 자랑하여 실질적으로 시간과 자원을 절약할 수 있습니다.



### A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language? (https://arxiv.org/abs/2502.14924)
- **What's New**: 이 연구는 언어의 정보-이론적 복잡성과 대형 언어 모델(LLM)에서의 프랙탈 구조를 비교하여 이러한 모델들이 자연 언어의 프랙탈 특성을 얼마나 잘 재현하는지를 조사합니다. 특히, 프랙탈 매개변수가 LLM의 출력에서 광범위하게 변동하는 반면, 자연 언어에서는 좁은 범위에 속한다는 점을 강조합니다. LLM이 특정한 조건(예: 온도 설정과 프롬프트 방법)에서 이러한 특성을 구현하지 못할 수 있다는 점도 발견하였습니다.

- **Technical Details**: 언어는 자기 유사성(self-similarity)과 장기 의존성(long-range dependence, LRD)을 따라 구조화되어 있으며, 이러한 특성은 Hurst 지수와 Hölder 지수로 정량화됩니다. LLM이 자연 언어의 특성을 모방하기 위해서는 토큰 수준에서 잘 보정되어야 하지만, 추론 과정에서의 오류는 격렬한 프랙탈 구조를 왜곡할 수 있습니다. 연구는 240,000개 이상의 LLM 생성 문서를 포함한 데이터셋을 구축하여, 다양한 프롬프트와 온도 설정에 따른 LLM의 출력을 분석했습니다.

- **Performance Highlights**: 이 연구에서는 LLM의 프랙탈 특성이 자연 언어보다 확연히 다르다는 것을 보여주었습니다. 특히, 프랙탈 매개변수를 통해 LLM 생성 텍스트를 식별하는 데 유용할 수 있으며, 주요 모델 구조에도 이러한 결과가 보편적으로 적용된다는 점이 중요한 발견입니다. 결과적으로 이 연구는 LLM의 합성 텍스트 생성을 평가하고 탐지하는 데 필요한 통찰을 제공합니다.



### AI Thinking as a Meaning-Centered Framework: Reimagining Language Technologies Through Community Agency (https://arxiv.org/abs/2502.14923)
Comments:
          LT4All 2025. Language Technologies for All - 2025. Advancing Humanism through Language Technologies. Paris (FR), UNESCO Headquarters, 24-26 February 2025

- **What's New**: 이번 논문은 기술 발전이 지역사회와 함께 이루어져야 한다는 새로운 방안을 제시합니다. AI Thinking 프레임워크를 통해 언어와 문화의 보존을 중심으로 기술 개발을 진행할 수 있도록 합니다. 이는 모든 커뮤니티의 언어적 지식 표현에서 지역사회의 통제를 유지할 수 있는 다층적 기술 생태계를 설계하는 것을 포함합니다.

- **Technical Details**: 논문에서 제안하는 AI Thinking 프레임워크는 언어 기술 개발의 중심에 의미를 두고, 인지적, 사회적, 문화적 차원 간의 복잡한 상호작용을 유지합니다. 이 구조는 커뮤니티 지식 시스템과의 통합을 지원하며, 기존 언어 기술의 한계를 극복하는 것을 목표로 합니다. 또한, 기존 모델들이 필요로 하는 방대한 디지털 언어 데이터 부족 문제를 해결하는 동시에, 문화적 맥락을 유지하는 방법론을 제시합니다.

- **Performance Highlights**: AI Thinking 프레임워크는 언어 기술의 통합적인 접근을 통해 지역사회의 기관들을 강화하며, 참여 및 통제 권한을 보장하는 것을 목표로 합니다. 이 논문에서는 현대 기술이 문화적 관점에서 어떻게 의미를 보존할 수 있는지를 논의하며, 향후 연구 방향 및 기술적 요건을 다루고 있습니다. 이러한 접근은 언어의 복잡성을 잃지 않으면서도 현대의 기술적 도전에 대응할 수 있는 가능성을 보여줍니다.



### SIFT: Grounding LLM Reasoning in Contexts via Stickers (https://arxiv.org/abs/2502.14922)
- **What's New**: 이 논문은 대형 언어 모델(LLM)들이 맥락을 잘못 해석하는 문제가 reasoning 과정에서 중요한 이슈가 될 수 있음을 밝힙니다. 특히, "10 dollars per kilo"에서 LLM이 'per'를 '각각'이라는 의미로 이해하지 못해 계산 오류를 발생할 수 있음을 예로 듭니다. 이를 해결하기 위해 새로운 포스트 트레이닝 접근법인 **Stick to the Facts (SIFT)**를 제안하며, 이는 LLM의 reasoning을 맥락에 기반하여 보다 정확하게 수행할 수 있도록 돕습니다.

- **Technical Details**: SIFT의 핵심 요소는 *Sticker*로, 모델 자체에 의해 생성되어 맥락 내 핵심 정보를 강조합니다. SIFT는 입력 쿼리를 통해 핵심 사실을 요약하고, Sticker 기반의 두 가지 예측을 생성합니다. 만약 두 예측이 다르다면, Sticker는 *forward* 최적화 및 *inverse* 생성 과정을 통해 점진적으로 refinment됩니다. 이는 보다 신뢰할 수 있는 reasoning 결과를 도출하기 위한 방법입니다.

- **Performance Highlights**: SIFT는 다양한 모델과 벤치마크에서 일관된 성과 개선을 보였습니다. 예를 들어 DeepSeek-R1 모델의 경우 AIME2024에서 pass@1 정확도를 78.33%에서 **85.67%**로 향상시켜 오픈 소스 커뮤니티에서 새로운 최첨단 결과를 수립했습니다. 또한, Llama3.2-3B-Instruct와 같은 소형 모델에서도 약 1.03%에서 7.34%의 정확도 향상을 보여주며, SIFT의 효과를 입증합니다.



### Display Field-Of-View Agnostic Robust CT Kernel Synthesis Using Model-Based Deep Learning (https://arxiv.org/abs/2502.14920)
Comments:
          Accepted at IEEE ISBI 2025

- **What's New**: 본 논문에서는 X-레이 컴퓨터 단층촬영(CT)에서 이미지 기반 커널 합성을 위한 프레임워크를 제안합니다. 기존 방법들은 직접적인 커널 합성(direct kernel synthesis) 사용 시, 재구성되고자 하는 이미지의 품질이 저하되는 경향이 있었습니다. 고유의 Pre-defined 모델을 활용하여 커널의 특성과 DFOV(Displayed Field of View)를 외부 모델로 통합시키는 접근법을 통해 효율적으로 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 모델 기반의 딥러닝(model-based deep learning) 접근법으로 커널 합성 문제를 다루고 있습니다. 이 시스템은 딥러닝 기반의 프로젝션 단계와 데이터 일관성을 유지하는 반복적인 해법을 포함합니다. 커널의 MTF(modulation transfer function)와 DFOV 정보를 명시적으로 통합함으로써, 다양한 DFOV에 대해 단일 네트워크를 훈련할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과는 임상 데이터에서 측정된 결과와 와이어 팬텀(wire phantom) 데이터의 변조 전송 함수 분석을 포함하여 제안된 방식의 유용성을 보여줍니다. DFOV 변동에 대해 제안된 방법이 보다 견고하다는 것을 입증하며, 다른 네트워크와 비교한 결과, 노이즈 및 아티팩트 관리에 있어 탁월한 성능을 나타냅니다.



### RAPTOR: Refined Approach for Product Table Object Recognition (https://arxiv.org/abs/2502.14918)
Comments:
          Accepted for WACVW 2025 (VisionDocs)

- **What's New**: 이번 연구에서는 RAPTOR라는 모듈형 후처리 시스템을 도입하여 기존의 테이블 추출 기술, 특히 제품 테이블에 대한 성능을 향상시키고자 하였습니다. 기존 DEtection TRansformer (DETR)와 TAble TRansformer (TATR) 기반의 모델들이 직면한 다양한 테이블 형식의 문제들을 해결하고, 정밀도와 구조 예측 모두에서 개선을 도모합니다. RAPTOR는 특히 이커머스와 관련된 비즈니스 문서 분석에 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: RAPTOR는 Genetic Algorithm을 활용하여 모듈의 매개변수를 최적화하고, ICDAR 2019 및 다양한 공개 데이터셋에서 사전훈련된 DETR과 TATR 모델을 사용하여 테이블 영역 감지(Table Detection, TD) 및 테이블 구조 인식(Table Structure Recognition, TSR)을 수행합니다. 이 모듈형 시스템은 기존 모델을 재조정하는 대신, 제한된 데이터로 모델 매개변수를 학습할 수 있는 기회를 제공합니다. 기존의 문제 해결을 위한 컴포넌트를 통합함으로써 TD 및 TSR 과정에서의 성능을 향상시킵니다.

- **Performance Highlights**: RAPTOR는 다섯 개의 데이터셋에서 평가되었으며, 특히 제품 테이블에서의 성능 향상이 두드러졌습니다. 다양한 테이블 형식에서도 전반적으로 합리적인 성능을 유지했으며, 여러 데이터셋에 따라 기본 모델의 성능을 개선하는 데 기여했습니다. 연구팀은 비즈니스 테이블에서 나타나는 일반적인 오류 유형을 식별하고 이를 해결하기 위한 모듈형 시스템으로서 RAPTOR의 유용성을 확인하였습니다.



### Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning (https://arxiv.org/abs/2502.14917)
- **What's New**: 이번 논문에서는 Sce2DriveX라는 새롭고 인간 유사한 driving chain-of-thought (CoT) 추론 MLLM 프레임워크를 제안합니다. 이 프레임워크는 다중 모드 학습(multi-modal joint learning)을 통해 지역 장면 비디오(local scene videos)와 전체적 BEV 맵(global BEV maps)을 활용하여 장기적 시공간 관계(long-range spatiotemporal relationships)와 도로Topology를 깊숙이 이해합니다. 이를 통해 3D 정적/동적 장면에서의 종합적인 인식(perception) 및 추론(reasoning) 능력을 향상시키며, 다양한 장면에서도 자율 주행의 일반화를 달성합니다.

- **Technical Details**: Sce2DriveX는 다중 뷰 장면 비디오(multi-view scene videos)와 BEV 맵을 정렬하여 통합된 시각적 특징 공간(visual feature space)을 형성하고, 이를 텍스트 임베딩 공간(text embedding space)으로 매핑하여 자연어 응답을 생성하는 구조를 갖추고 있습니다. 모델은 장면 이해(scene understanding), 동작 분석(behavior analysis), 모션 계획(motion planning), 차량 제어(vehicle control) 등의 분야에서 강력한 성능을 보입니다. 또한, 이에 대한 효과적인 훈련을 위해 3D 공간 이해(3D spatial understanding) 및 장축 작업 추론(long-axis task reasoning)에 특화된 종합적인 Visual Question Answering (VQA) 데이터셋을 구축했습니다.

- **Performance Highlights**: Sce2DriveX는 CARLA Bench2Drive 벤치마크에서 장면 이해(scene understanding)부터 엔드 투 엔드 드라이빙(end-to-end driving)까지 모든 작업에서 최고의 성능을 달성했습니다. 광범위한 실험을 통해 복잡한 시나리오에서도 강력한 일반화 성능을 보이며, 자율 주행 시스템의 의사 결정 과정이 인간의 인지(cognitive processes)와 일치하도록 하는 데 기여하고 있습니다.



### MKE-Coder: Multi-Axial Knowledge with Evidence Verification in ICD Coding for Chinese EMRs (https://arxiv.org/abs/2502.14916)
- **What's New**: 이번 논문은 중국 전자 의료 기록(EMR)에서 국제 질병 분류(International Classification of Diseases, ICD)를 자동 코딩하기 위한 새로운 프레임워크인 MKE-Coder를 소개합니다. 이 프레임워크는 주로 질병 기반의 다축 지식(Multi-axial Knowledge)을 통합하고, 신뢰할 수 있는 임상 증거(Clinical Evidence)를 검증하여 코드의 유효성을 보장합니다. MKE-Coder의 도입은 기존 방법들이 중국 EMR의 특수성을 고려하지 못한 문제를 해결하는 데 기여하고 있습니다.

- **Technical Details**: MKE-Coder는 다축 지식을 활용하여 ICD 코드를 생성하는 과정에서, 먼저 진단 후보 코드를 추출하고, 각 코드에 대해 필수 지식을 분류합니다. 다음으로, EMR의 방대한 내용을 활용해 임상 증거를 수집하고, 유효한 증거를 스코어링 모델을 통해 필터링합니다. 마지막으로, 마스크 언어 모델링 전략을 기반으로 한 추론 모듈을 구현하여 후보 코드와 관련된 모든 지식이 증거에 의해 지원되는지 검토합니다.

- **Performance Highlights**: 실험 결과, MKE-Coder는 중국 EMR 기반의 ICD 자동 코딩 작업에서 중요한 개선 사항을 보여주었습니다. 특히, 실제 코딩 시나리오에서 이 방법을 평가한 결과, 코더들이 코딩 정확성과 속도를 크게 향상할 수 있었음을 입증하였습니다. 이러한 성취는 MKE-Coder가 기존의 영어 EMR 위주의 방법론과 차별화된 접근을 취하고 있음을 나타냅니다.



### OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignmen (https://arxiv.org/abs/2502.14913)
Comments:
          15 pages

- **What's New**: 최근 다중 에이전트 협업을 이용한 대형 언어 모델(Large Language Models, LLMs)이 Text-to-SQL 작업에서 눈에 띄는 성과를 올리고 있는 가운데, OpenSearch-SQL이라는 새로운 방법론이 제안되었습니다. 이 방법론은 전체 텍스트-투-SQL 작업을 네 가지 주요 모듈인 전처리(Preprocessing), 추출(Extraction), 생성(Generation), 개선(Refinement) 및 일관성 정렬(Alignment) 모듈로 나누어 운영합니다. 특히, 이 구조는 에이전트의 입력과 출력을 정렬하여 지침을 따르지 못하거나 해리(hallucination) 문제를 줄이는 데 기여합니다.

- **Technical Details**: OpenSearch-SQL에서는 텍스트-투-SQL 작업을 인간의 SQL 작성 프로세스를 모델링한 표준 프로세스에 기반하여 정의합니다. 이 과정은 전처리, 추출, 생성 및 개선의 네 가지 단계로 구성되어 있으며, 요청을 이해하고 SQL을 완성하는 데 필요한 정보를 체계적으로 처리합니다. 또한, SQL-Like라는 중간 언어를 개발하여 모델이 SQL의 구조를 효율적으로 생성하도록 돕고, 자가 학습 기반의 Query-CoT-SQL 형태의 동적 몇 샷(few-shot) 전략을 설계하였습니다.

- **Performance Highlights**: OpenSearch-SQL의 실험 결과, BIRD 개발 세트에서 69.3%의 실행 정확도(EX)를 달성하며, 테스트 세트에서는 72.28%의 성능을 기록하였습니다. 또한, 보상 기반 유효성 점수(R-VES)는 69.36%로, 제출 시 모든 지표에서 1위를 기록하였습니다. 이러한 결과는 제안된 방법이 효과성과 효율성 모두에서 상당한 장점을 가지고 있음을 잘 보여줍니다.



### Batayan: A Filipino NLP benchmark for evaluating Large Language Models (https://arxiv.org/abs/2502.14911)
Comments:
          Submitted to ACL 2025

- **What's New**: 최근 큰 언어 모델(LLMs)의 발전에도 불구하고, 자원이 부족한 언어의 언어적 뉘앙스는 여전히 탐구되지 않고 있습니다. 본 논문에서는 필리핀어를 평가하기 위해 Batayan이라는 포괄적인 벤치마크를 소개하며, 이는 자연어 처리(NLP)의 세 가지 주요 역량인 이해(understanding), 추론(reasoning), 생성(generation)을 평가하기 위해 설계되었습니다. Batayan은 타갈로그(Tagalog) 및 코드 스위칭된 타글리시(Taglish) 발화를 포괄하는 여덟 가지 과제를 통합하였습니다.

- **Technical Details**: Batayan은 제공되는 데이터셋의 질과 일관성을 보장하기 위해 원어민에 의해 세심하게 주석 처리된 점에서 다른 필리핀어 데이터셋과 차별화됩니다. 연구에서는 여러 다국어 LLM을 대상으로 한 평가 결과를 보고하며, 필리핀어에서 발견된 성능 차이를 강조하고 있습니다. 또한, 필리핀어의 복잡한 형태론(morphology)과 문법 구조를 모델링할 때의 고유한 도전 과제를 다루고 있습니다.

- **Performance Highlights**: 저자는 Batayan에서의 엄격한 평가 결과를 통해 필리핀어가 다국어 LLM에서 적게 대표되고 있음을 나타내는 중요한 성능 격차를 강조하고 있습니다. 이를 통해 필리핀어와 같은 저대표 언어에 대한 명확한 지원과 교육 튜닝의 필요성을 강조하고 있으며, 향후 필리핀어 자원의 발전을 위한 실질적인 프레임워크를 제공합니다.



### EvoP: Robust LLM Inference via Evolutionary Pruning (https://arxiv.org/abs/2502.14910)
- **What's New**: 이번 논문에서는 EvoP라는 진화적 구조 절단 프레임워크를 제안합니다. EvoP는 군집 기반 보정 데이터셋 샘플링(Cluster-based Calibration Dataset Sampling) 전략을 통해 보다 다양한 보정 데이터셋을 생성하고, 진화적 절단 패턴 탐색(Evolutionary Pruning Pattern Searching) 방법을 통해 최적의 절단 패턴을 찾습니다. 기존 기술들과는 달리 EvoP는 최적의 성능과 효율성을 동시에 달성하는 것을 목표로 하고 있습니다.

- **Technical Details**: EvoP는 대규모 언어 모델(LLM)의 구조 절단을 개선하기 위한 새로운 접근 방식을 제공합니다. 이는 모델의 크기를 줄이는 것뿐만 아니라, 수행 성능도 극대화하기 위해 설계되었습니다. 네트워크 절단 문제는 사전 훈련된 모델과 보정 데이터셋을 바탕으로 최적의 절단 패턴을 찾는 것을 목표로 하며, 사전 훈련된 모델의 매개변수에 기반하여 절단 패턴 공간에서 최적 솔루션을 찾습니다.

- **Performance Highlights**: EvoP는 다양한 LLM 및 다운스트림 작업에서 기존 최첨단 절단 방법을 초과하는 성능을 보였습니다. 이를 통해 다양한 데이터셋에서도 높은 일반화 능력을 입증하였습니다. 연구 결과는 EvoP가 실제 세계 응용에서 LLM을 효율적으로 배포하는 실용적이고 확장 가능한 솔루션임을 보여줍니다.



### PTB-Image: A Scanned Paper ECG Dataset for Digitization and Image-based Diagnosis (https://arxiv.org/abs/2502.14909)
- **What's New**: 본 연구에서는 PTB-Image라는 이름의 새로운 데이터셋을 소개하며, 이는 종이 ECG와 그에 상응하는 디지털 신호로 구성되어 있습니다. 이는 ECG의 디지털화 연구를 위한 기초 자료를 제공하여, 고전적인 종이 기반 ECG의 자동 분석 문제를 해결할 수 있는 가능성을 엽니다. 또한 VinDigitizer라는 기본 디지털화 방법을 통해 종이 ECG를 디지털 시계열 신호로 변환하는 방식이 제안되었습니다.

- **Technical Details**: PTB-Image 데이터셋은 549개의 종이 ECG 기록을 포함하며, 각 기록은 12개의 동시 ECG 신호로 구성되어 있습니다. VinDigitizer는 세 단계로 구성된 파이프라인을 통해 신호를 분리하고, 배경에서 파형을 추출하며, 디지털 신호로 재구성하는 과정을 수행합니다. 이 방법론은 YOLOv8 모델을 사용하여 신호가 포함된 행을 정확하게 감지하며, Otsu의 임계값을 활용하여 신호의 경계를 정리합니다.

- **Performance Highlights**: VinDigitizer를 통해 얻은 평균 신호 대 잡음비(SNR)는 0.01 dB로, 이는 종이 ECG의 디지털화 과정에서 발생할 수 있는 왜곡 문제를 강조합니다. PTB-Image 데이터셋을 기반으로 한 연구는 의료 기록 통합과 AI 모델 훈련을 위한 기초 자료를 제공하여, 원거리 진료 및 자동화된 심장진단 분야에서의 발전을 지원할 수 있는 잠재력을 가집니다.



### KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models (https://arxiv.org/abs/2502.14908)
- **What's New**: 이번 연구에서는 Vision Language Models (VLMs)의 멀티모달 환경에서의 지식 갈등에 대한 영향을 조사하기 위해 \

- **Technical Details**: \

- **Performance Highlights**: \



### GneissWeb: Preparing High Quality Data for LLMs at Sca (https://arxiv.org/abs/2502.14907)
- **What's New**: 본 논문에서는 약 10조 토큰(token) 규모의 대규모 데이터셋 GneissWeb을 소개하고, 이 데이터셋이 LLM(대규모 언어 모델) 훈련에 필요한 데이터 품질과 양의 요구를 충족한다고 주장합니다. 특히, GneissWeb은 고품질의 데이터를 제공하며, 이는 모델의 일반화 능력을 향상시키는 데 중요한 역할을 합니다. 이 논문에서 제시된 방법론은 정확한 부분 문자열 제거와 잘 구축된 품질 필터의 조합으로 구성되어 있습니다. 이는 기존의 공공 데이터셋들이 5조 토큰 이하에 그치고 있는 문제를 해결하고자 합니다.

- **Technical Details**: GneissWeb의 데이터 생성 과정에는 샤딩된 정확한 부분 문자열 중복 제거 및 복합적인 품질 필터 조합이 포함됩니다. 새로운 품질 필터링 기법인 'Extreme Tokenized Documents Removal'를 도입하여, 모델 훈련에 사용될 토큰화된 데이터를 기반으로 저품질 문서를 효과적으로 걸러냅니다. 또한, 인간의 독서 능력을 활용한 가독성 스코어 필터를 통해 다양한 도메인의 저품질 문서를 제외할 수 있도록 합니다. 이 모든 절차는 공개적으로 제공되는 IBM 데이터 준비 키트를 사용하여 효율적으로 실행되었습니다.

- **Performance Highlights**: GneissWeb으로 훈련된 모델은 FineWeb-V1.1.0으로 훈련된 모델에 비해 11개 벤치마크에서 평균 2.73% 높은 성능을 보였습니다. 벤치마크 수를 20개로 확장하였을 때도, GneissWeb으로 훈련된 모델이 1.75%의 성능 우위를 유지합니다. 이 결과는 GneissWeb이 기존의 최신 오픈 대규모 데이터셋보다 더 나은 성능을 발휘함을 실증적으로 보여줍니다.



### Beyond Words: Exploring Cultural Value Sensitivity in Multimodal Models (https://arxiv.org/abs/2502.14906)
- **What's New**: 이 논문은 문화적 맥락에 기반한 Large Language Models (LLMs)와 비교하여, Vision-Language Models (VLMs)의 가치 정렬 연구에 대한 중요한 공백을 다룹니다. 저자들은 시각적 데이터를 사용하여 문화적 가치를 평가하는 새로운 접근 방식을 제공하며, 이 모델들이 문화적 값에 대한 민감성을 어떻게 보여주는지를 조사합니다. 중요한 발견은 VLMs가 문화적 가치에 민감성을 보이지만, 그 정렬은 맥락에 따라 달라진다는 것입니다.

- **Technical Details**: 이 연구는 다양한 국가와 문화적 맥락을 반영하는 이미지를 사용하여 VLM의 문화적 가치 일치를 평가하는 포괄적인 프레임워크를 제시합니다. 이 모델들은 WVS(World Values Survey)를 활용하여 문화적 정체성을 형성하며, 인간 커뮤니케이션에 내재된 문화적 신호에 대한 반응을 탐색합니다. 또한, 다양한 파라미터 크기가 있는 모델들 간의 성능 차이를 분석하여, 크기가 증가하는 것이 항상 더 나은 가치 정렬을 보장하지 않는다는 점을 강조합니다.

- **Performance Highlights**: 모델은 WVS 질문에 대한 세부 주제별 반응에서 문화적 규범과의 정렬을 평가합니다. 본 연구에서는 특히 종교, 인종, 이민 문제에 대한 성능을 깊이 있게 분석하고, 이는 모델들이 얼마나 정교하게 맥락별 과제를 수행할 수 있는지를 보여줍니다. 또한, 다양한 이미지의 유형을 통해 이미지가 VLM의 가치에 미치는 영향을 평가함으로써, 멀티모달 모델의 문화적 가치 이해를 향상시킬 수 있는 가능성을 강조합니다.



### Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherenc (https://arxiv.org/abs/2502.14905)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 생성 과정에서 엄격한 스키마 준수를 강제하는 문제를 다룹니다. DeepSeek R1 강화 학습 프레임워크를 기반으로, 15억 매개변수 모델의 구조적 추론 기술을 훈련시키기 위한 새로운 파이프라인을 제안합니다. 특히, 비구조화된 데이터에서 구조화된 데이터로의 변환을 위한 2만 개의 샘플 데이터셋을 활용해 모델의 핵심 추론 능력을 구축하였으며, 이후 1만 개의 추론 샘플 데이터셋에서 감독 학습을 통해 스키마 준수를 더욱 정교하게 다듬었습니다.

- **Technical Details**: 이 접근법은 합성(문서 생성) 데이터셋 구축과 사용자 정의 보상 함수를 결합하여 Group Relative Policy Optimization(GRPO) 아래서 훈련됩니다. 모델은 강력한 추론 능력을 함양하기 위해 비구조화된 데이터와 구조화된 데이터 모두를 생성하는 작업을 수행하며, 사용자 정의 보상 메커니즘을 통해 출력의 스키마 준수를 직접 평가합니다. 즉, 모델 훈련은 모든 관련 기준에서 높은 점수를 획득하는 출력을 만들어내는 방향으로 이루어집니다.

- **Performance Highlights**: 우리의 ThinkJSON 접근법은 DeepSeek R1(671B), Qwen-1.5B와 Qwen-7B의 축약 버전, Gemini 2.0 Flash(70B)와 비교하여 실용적인 응용에서의 효과를 보여주었습니다. 훈련 범위가 상대적으로 modest했음에도 불구하고, GRPO 훈련을 위해 8xH100 GPU 클러스터에서 약 20시간, SFT를 위해 1xA100 GPU에서 3시간이 소요되는 동안, 우리의 모델은 스키마 일관성을 보장하는 데 강력한 성능을 발휘합니다. 이 연구는 스키마가 제약되는 텍스트 생성에 대한 자원 효율적인 프레임워크의 실용적 유용성을 강조합니다.



### PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths (https://arxiv.org/abs/2502.14902)
- **What's New**: 이 논문에서는 기존의 그래프 기반 RAG(유도 검색 증강 생성) 방법의 한계를 지적하며, 정보의 중복성이 오히려 문제라고 주장합니다. PathRAG라는 새로운 접근 방식을 제안하여, 텍스트 정보를 인덱싱 그래프 형태로 구조화하여 응답 생성의 질을 향상시키려 합니다. 이 시스템은 키 관계 경로를 효과적으로 검색하고, 이를 텍스트 형태로 변환하여 LLM(대형 언어 모델)에 제공함으로써 더 논리적이고 일관적인 답변을 생성할 수 있도록 합니다.

- **Technical Details**: PathRAG는 플로우 기반 가지치기(flow-based pruning)를 통해 중복 정보를 줄이고, 각 쿼리 키워드에 대해 인덱싱 그래프에서 관련 노드를 검색할 수 있는 알고리즘을 사용합니다. 이 방법은 노드 쌍 사이의 키 관계 경로를 식별하여 불필요한 정보 노이즈를 줄이는 동시에, 경로의 신뢰도 점수를 부여하여 보다 효율적으로 정보를 처리합니다. 마지막으로, 각 경로에 따라 노드와 에지 정보를 텍스트 관계 경로로 연결하여 LLM 프롬프트에 통합합니다.

- **Performance Highlights**: PathRAG는 실험적으로 6개의 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보여주며, 모든 평가 차원에서 더 나은 결과를 도출했습니다. GraphRAG와 LightRAG에 비해 평균 승률이 각각 60.44% 및 58.46%로 나타났습니다. 특히, PathRAG의 성능이 더 큰 데이터셋에서 더욱 두드러지며, 실제 응용에 보다 적합한 결과를 제공합니다.



### Can AI mimic the human ability to define neologisms? (https://arxiv.org/abs/2502.14900)
- **What's New**: 이번 연구는 인공지능(AI)이 언어 관련 작업에서 인간의 성과를 얼마나 잘 모방할 수 있는지를 탐구하는 ongoing debate의 일환으로 진행되었다. 기존 연구가 AI의 다양한 언어 능력에 초점을 맞춘 반면, 네올로즘(neologisms)을 정의하는 방식에 대한 분석이 부족했던 점을 개선하고자 했다. 연구에서는 그리스어에서 만든 세 가지 네올로즘 유형인 blends, compounds, derivatives의 정의에 대한 인간과 AI의 반응 일치를 분석하였다.

- **Technical Details**: 이 연구는 온라인 실험을 통해 진행되었으며, 인간 참가자들은 네올로즘에 가장 적절한 정의를 선택하는 방식으로 참여하였고, ChatGPT는 동일한 프롬프트(prompt)를 받았다. 연구 결과에 따르면, blends와 derivatives에 대해서는 인간과 AI의 응답 간에 공정한 일치가 있었으나, compounds에 대해서는 일치가 없었다. 그러나 인간 다수 응답을 고려했을 때, blends와 derivatives에 대한 AI의 응답과 높은 일치를 보였다.

- **Performance Highlights**: 이 연구 결과는 인간 언어의 복잡성과 AI가 그 뉘앙스를 포착하는 데 여전히 직면한 도전을 강조한다. 연구는 특히 AI 모델의 해석력을 높이기 위해 더 발전된 의미 네트워크(semantic networks) 및 맥락 학습 메커니즘을 통합해야 할 필요성을 제시하고 있다. 이는 복잡한 단어 형성(especially compounds)을 해석하는 데 있어 더욱 중요한 요소가 될 것이다.



### UPCMR: A Universal Prompt-guided Model for Random Sampling Cardiac MRI Reconstruction (https://arxiv.org/abs/2502.14899)
Comments:
          Accepted paper for STACOM 2024

- **What's New**: 이 논문에서는 심장 자기공명영상(CMR) 복원을 위한 보편적인 모델인 UPCMR을 소개합니다. UPCMR은 두 가지 유형의 학습 가능한 프롬프트를 통합하여 다양한 언더샘플링(k-space에서의 불완전 샘플링) 방식에 적응할 수 있도록 설계되었습니다. 이 접근 방식은 UNet 구조와 결합된 각 블록에서 효과적인 품질 향상을 보여줍니다.

- **Technical Details**: UPCMR 모델은 다중 코일 언더샘플링 k-space 측정값으로부터 복잡한 값을 가지는 MR 영상 시퀀스를 재구성하는 것을 목표로 합니다. 이 모델은 언더샘플링 특화 프롬프트와 공간 특화 프롬프트를 제공합니다. 또한 모델은 k-space 궤적 및 가속 요인을 사전 정보로 활용하여 학습 가능한 프롬프트 풀이 효율적으로 설계되었습니다.

- **Performance Highlights**: UPCMR 모델은 CMRxRecon2024 챌린지 데이터셋을 사용하여 모든 무작위 샘플링 시나리오에서 재구성된 이미지 품질이 크게 향상되었습니다. 기존의 전통적인 방법들과 비교하여 강력한 적응 가능성을 보여주며, 이는 다양한 샘플링 모드에서 효과적인 훈련 전략 덕분입니다. 결과적으로 UPCMR은 심장 MRI 재구성 작업에 있어 뛰어난 성능을 입증하였습니다.



### Retrieval-augmented systems can be dangerous medical communicators (https://arxiv.org/abs/2502.14898)
Comments:
          Preprint

- **What's New**: 이 논문에서는 환자들이 건강 정보를 찾기 위해 생성형 AI를 점점 더 많이 이용하는 추세를 반영하고 있다. 특히 의료 분야에서 AI가 생성한 답변의 정확성을 높이기 위해 retrieval-augmented generation(RAG) 및 citation grounding과 같은 기법이 사용되고 있지만, 이러한 방법들이 오히려 환자들에게 혼란을 줄 수 있다는 점을 강조한다. 저자들은 현재의 시스템이 환자의 질문을 문맥적으로 이해하지 못해 잘못된 인식을 강화할 수 있다고 주장한다.

- **Technical Details**: RAG 시스템은 환자의 질의에 대한 정확한 응답을 제공하기 위해 신뢰할 수 있는 원천을 참조하는 데 초점을 맞춘다. 하지만 이 논문은 AI가 생성한 응답이 원본 정보에서 얻은 사실들을 단순히 재구성하는 데 그칠 뿐, 핵심적인 정보나 맥락을 누락해 환자에게 혼란을 초래할 수 있음을 보여준다. 데이터 분석을 통해 Google AI Overview 및 Perplexity AI의 질의 응답 패턴을 조사하고 잘못된 해석을 일으킬 수 있는 주요 요소들을 확인했다.

- **Performance Highlights**: 대규모 데이터 분석 결과, Google AI Overview와 Perplexity AI 모두 환자 질의에 대해 비슷한 응답을 생성했지만, 특정 조건이나 증상 질의를 통해 생성된 응답이 원문을 바탕으로 하더라도 환자에게 잘못된 인식을 초래한다는 점이 드러났다. 특히, 질병이나 절차에 대한 안전성과 위험에 관한 질의에서 상반된 정보가 제공될 때, 두 시스템 간의 참조 자료의 유사성이 낮았고, 그로 인해 건강 불안감을 유발할 잠재성이 크다는 것을 알 수 있었다.



### A Comprehensive Survey on Concept Erasure in Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.14896)
- **What's New**: 이 연구는 Text-to-Image (T2I) 모델의 개념 지우기(concept erasure) 기술에 대한 구조적이며 포괄적인 설계를 제시합니다. 특히 개념 지우기 방법론을 최적화 전략과 수정되는 아키텍처 구성 요소에 따라 분류하고 있습니다. 또한, T2I 모델의 생성 기능에서 요구되는 개념을 체계적으로 제거하여 저작권이 있는 스타일이나 민감한 이미지를 생성하지 않도록 하려는 목적을 설명합니다.

- **Technical Details**: 본 논문은 Stable Diffusion (SD) 모델을 중심으로 T2I diffusion 모델의 구조를 설명합니다. SD는 이미지 복원을 위한 비전 디코더, 반복적인 노이즈 제거를 위한 잠재적 확산 모델, 텍스트 프롬프트를 조건 벡터로 변환하는 조건부 텍스트 인코더로 구성되어 있습니다. 개념 지우기 방법은 이러한 모델 내부 구성 요소를 수정하거나 추론 과정에 개입하여 민감한 또는 제한된 개념의 재현을 방지합니다.

- **Performance Highlights**: 저자들은 개념 지우기 기법의 효과를 평가하기 위한 평가 메트릭과 데이터셋을 통합하여 앞으로의 연구 방향에 대한 기초를 제공합니다. 연구는 적대적 공격과 방어 전략에 대한 분석을 포함하고 있으며, T2I 모델의 견고성을 높이기 위한 새로운 접근 방법을 탐구합니다. 이러한 작업들은 개념 지우기의 발전과 향후 방향에 대한 귀중한 통찰력을 제공하는 데 기여합니다.



### FOCUS on Contamination: A Geospatial Deep Learning Framework with a Noise-Aware Loss for Surface Water PFAS Prediction (https://arxiv.org/abs/2502.14894)
- **What's New**: 이 연구는 PFAS 오염 예측을 위한 지리공간 딥러닝 프레임워크인 FOCUS를 소개합니다. FOCUS는 label noise-aware loss function을 활용하여 방대한 지역에서 PFAS 오염을 예측할 수 있도록 설계되어 있습니다. 이 모델은 수문학적 흐름 데이터, 토지 이용 정보, 그리고 이미 알려진 PFAS 원천과의 거리 데이터를 통합하여 예측 정확성을 개선합니다.

- **Technical Details**: FOCUS는 PFAS 오염의 세분화를 위해 래스터 데이터를 사용하는 지리공간 딥러닝 모델을 적용합니다. 이 프레임워크는 PFAS 확산에 대한 도메인 전문 지식을 활용하여 데이터 격차를 자동으로 보완하고, 이러한 가정을 신뢰도에 따라 가중치를 두어 왼쪽으로 캐싱하는 점이 특징입니다. 모델의 성능은 다양한 AI 모델 및 기존의 과학적 방법들과 비교하여 검증되었습니다.

- **Performance Highlights**: FOCUS의 평가 결과는 대규모 PFAS 모니터링을 위한 스케일 가능성, 정확성 및 강건성을 강조합니다. 이 연구는 환경 연구에 적극적으로 참여하는 비영리 단체와 수질 모델 전문 학자와 함께 진행되어, 최신 전문성과 실제 환경 문제를 반영합니다. FOCUS는 PFAS 오염의 복잡성을 해결하기 위해 AI와 지리공간 모델링의 최근 발전을 통합하여 새로운 발견을 가능하게 합니다.



### NOTA: Multimodal Music Notation Understanding for Visual Large Language Mod (https://arxiv.org/abs/2502.14893)
- **What's New**: 이번 논문에서는 NOTA라는 이름의 대규모 종합 멀티모달 음악 표기법 데이터셋을 처음으로 제안합니다. 이 데이터셋은 1,019,237개의 레코드로 구성되어 있으며, 세 가지 전 세계 지역에서 수집되었습니다. 데이터셋은 음악 정보 추출, 크로스 모달 정렬 테스트, 음악 표기법 분석의 세 가지 주요 작업을 다룹니다.

- **Technical Details**: NOTA 데이터셋은 ABC 표기법을 사용하여 음악 점수를 표현합니다. 데이터셋은 헤더와 본체로 구성되어 있으며, 헤더에는 참조 번호와 제목, 박자 기호, 기본 음표 길이 등이 포함됩니다. 본체는 주로 음표와 마침표를 포함하여 음악의 구조적 요소를 표현합니다.

- **Performance Highlights**: NotaGPT-7B 모델은 17개의 주요 멀티모달 대형 언어 모델과의 실험에서 음악 이해에 있어 두드러진 개선을 보였습니다. 기존 최고 성능 모델인 Gemini는 33.34%의 음악 정보 추출률을 달성한 반면, 우리 모델은 67.84%의 성과를 기록했습니다. 이는 멀티모달 음악 데이터셋의 중요성과 NOTA 데이터셋의 효과성을 보여줍니다.



### EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild (https://arxiv.org/abs/2502.14892)
Comments:
          NAACL 2025 Findings. Project page at this https URL

- **What's New**: EgoSpeak는 실제 환경에서 대화 에이전트가 언제 말을 시작해야 하는지를 예측하는 혁신적인 프레임워크로, 에고센트릭(egocentric) 비디오 스트리밍을 기반으로 설계되었습니다. 이 시스템은 사용자의 1인칭 시점을 모델링하여, 대화 에이전트가 주변을 관찰하고 말할 시점을 동적으로 결정할 수 있도록 합니다. 이를 통해 인간과 유사한 상호작용을 구현하고, 복잡한 자연 대화와 간단한 실험적 세팅 간의 간극을 줄입니다.

- **Technical Details**: EgoSpeak는 1인칭 시점(first-person perspective)에서 RGB 처리(RGB processing), 실시간 처리(online processing), 자르지 않은 비디오(untrimmed video) 처리를 포함한 네 가지 주요 기능을 통합합니다. 이러한 기능을 통해 에고센트릭 스트리밍 비디오에서 대화 에이전트가 직접 정보를 처리를 할 수 있도록 하여, 복잡한 다자 대화 환경에서도 자연스럽게 응답할 수 있게 하는 것이 중요한 목표입니다. 이 모델은 EasyCom과 Ego4D 데이터셋에서 실험되어 그 효과성이 입증되었습니다.

- **Performance Highlights**: EgoSpeak는 실시간으로 음성 시작 시점을 예측하며, 랜덤 및 정적 침묵 기반 기준보다 우수한 결과를 나타냅니다. 또한, 다중 모달 입력(multimodal input)과 상황 길이(context length)의 중요성을 강조하여 대화 에이전트가 언제 말을 할 지 결정하는 과정에서 효과적으로 작용함을 보여줍니다. YT-Conversation 데이터셋을 통해 대규모의 미선택 대화 비디오를 활용하여 다중 모달 턴 테이킹(turn-taking) 모델을 위한 훌륭한 교육 자원을 제공합니다.



### CoDiff: Conditional Diffusion Model for Collaborative 3D Object Detection (https://arxiv.org/abs/2502.14891)
- **What's New**: 이 논문에서는 자율 주행 분야에서 자주 발생하는 다중 에이전트 간 협업 3D 객체 감지 문제를 해결하기 위한 새로운 프레임워크, CoDiff를 제안합니다. CoDiff는 Conditional Diffusion Probabilistic Models (CDPM)를 활용하여 정보 교환 시 발생하는 공간적 및 시간적 노이즈 문제를 다룹니다. 이 연구는 확산 모델이 협업 인식에 적용된 첫 번째 사례로, 기존의 피처 융합 방법을 대체하여 에이전트 간의 특성을 효과적으로 정제합니다.

- **Technical Details**: CoDiff의 핵심 요소는 두 가지입니다. 첫 번째는 인식 압축(perception compression)으로, 이를 통해 각 에이전트의 피처를 잠재 공간(latent space)으로 압축하여 효율성을 극대화합니다. 두 번째는 조건적 생성(conditional generation) 모듈로, 이는 다양한 협력 에이전트의 피처를 입력으로 받아 노이즈를 점진적으로 정제하여 고해상도의 피처 생성을 수행합니다. 이로 인해, 다중 에이전트의 연결된 피처들은 통합적이고 일관된 표현을 가집니다.

- **Performance Highlights**: 실험 결과, CoDiff는 시뮬레이션 및 실제 데이터셋에서 다양한 시간 지연 및 자세 오류가 있을 때에도 기존의 관련 방법들보다 뛰어난 협업 3D 객체 감지 성능을 보였습니다. CoDiff는 특히 임의의 노이즈 혼합 분포를 학습하고, 효율적으로 특성을 융합하여 더욱 정확하고 견고한 결과를 도출합니다. 요약하자면, CoDiff는 다중 에이전트 협업 3D 객체 감지 성능을 획기적으로 향상시키는 방안을 제시하며, 자율 주행 및 로봇 시스템에서의 적용 가능성을 높입니다.



### Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability (https://arxiv.org/abs/2502.14889)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이 논문에서는 CLIP(Contrastive Language-Image Pretraining) 모델의 해석 가능성을 높이기 위한 새로운 틀인 Narrowing Information Bottleneck Theory(NIBT)를 제안합니다. 기존의 정보 병목 방법들은 고정된 가정이나 내재적인 무작위성의 영향을 받는 반면, NIBT는 현대의 기여 원칙(attribution axioms)을 만족시키고 해석 가능성을 더욱 향상시키도록 설계되었습니다. 실험 결과, NIBT는 이미지 해석 가능성을 평균 9%, 텍스트 해석 가능성을 평균 58.83% 향상시키고 처리 속도를 63.95% 가속화했습니다.

- **Technical Details**: CLIP 모델은 대규모 이미지-텍스트 쌍의 데이터로 훈련되어 이미지와 텍스트의 연관성을 학습합니다. 이 모델의 해석 가능성을 높이기 위해 NIBT에서는 무작위성 및 하이퍼파라미터 의존성을 제거하여 더 결정론적인 해석 결과를 제공합니다. 또한, 모델의 예측에 부정적인 영향을 미치는 특성 차원을 식별하는 새로운 개념인 negative property를 도입하여 해석 가능성을 더욱 강화했습니다.

- **Performance Highlights**: 본 연구의 접근 방식은 CLIP 모델의 해석 가능성을 기존 최첨단 방법들보다 현저히 개선했습니다. 이미지 해석 가능성이 평균 9% 향상되었으며, 텍스트 해석 가능성은 평균 58.83% 증가했습니다. 처리 속도 역시 63.95% 증가하여 효율성이 크게 높아졌습니다. 따라서 본 연구는 다양한 생성 AI 애플리케이션에서 CLIP 모델의 신뢰성과 투명성을 제고하는 데 중요한 기여를 하고 있습니다.



### The Multi-Faceted Monosemanticity in Multimodal Representations (https://arxiv.org/abs/2502.14888)
- **What's New**: 이 논문에서는 deep multimodal 모델에서 해석 가능한 feature를 추출하기 위해 최근의 feature monosemanticity(단일 의미성) 발전을 활용합니다. 특히, 대규모 이미지-텍스트 쌍으로 학습된 CLIP(Contrastive Language-Image Pretraining) 모델을 분석하여 해석 가능한 feature의 modality 연결성을 조사합니다. 우리는 Modality Dominance Score (MDS)를 도입하여 각 feature의 해석 가능한 특성을 해당 modality에 부여합니다.

- **Technical Details**: 우리는 두 가지 CLIP 모델, 즉 OpenAI의 전통적인 ViT-B-32 CLIP 모델과 이를 변형한 DeCLIP 모델을 사용합니다. DeCLIP은 이미지-텍스트 쌍 데이터 외에도 이미지-이미지 및 텍스트-텍스트 쌍을 활용한 self-supervision(자기 감독)을 포함하여 단일 의미의 feature를 더욱 효과적으로 추출합니다. Sparse Autoencoders(SAEs)와 Non-negative Constraint Loss(NCL)를 통해 multi-modal interpretability(다중 모달 해석 가능성) 강화를 위한 방법론을 개발합니다.

- **Performance Highlights**: 우리는 CLIP 모델의 feature가 인간의 인지 구조와 잘 일치한다는 사실을 발견했습니다. 이러한 해석 가능한 feature는 성별 편향 탐지, 적대적 공격 방어 및 텍스트-이미지 생성 등 다양한 실제 사용 사례에 유용하게 활용될 수 있음을 입증합니다. 이 결과는 다양한 modality 간의 주요 연결 및 차이를 이해하는 데 기여할 수 있는 대규모 multimodal 모델의 가능성을 보여줍니다.



### Vision-Enhanced Time Series Forecasting via Latent Diffusion Models (https://arxiv.org/abs/2502.14887)
- **What's New**: 이번 연구에서는 LDM4TS라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 시각 강화된 시간 시계열 예측을 위한 것으로, 외부 시각 데이터를 도입하는 대신 시간 시계열을 다중 시각 표현으로 변환하는 보완적 변환 기술을 최초로 사용합니다. 이를 통해 미리 훈련된 비전 인코더의 풍부한 특징 추출 능력을 활용할 수 있습니다.

- **Technical Details**: LDM4TS는 원시 시간 시계열 데이터를 구조적 인코딩을 통해 다중 시각 표현으로 변환합니다. 이 과정에서 세분화, 재발 그래프 및 Gramian Angular Fields와 같은 다양한 인코딩 방법을 사용하고, 변환된 표현은 낮은 차원의 잠재 공간으로 매핑됩니다. 이후, 사전 훈련된 잠재 확산 모델이 이 잠재 변수를 점진적으로 디노이즈합니다.

- **Performance Highlights**: 광범위한 실험 결과, LDM4TS는 다양한 데이터셋에서 최첨단 성능을 달성했습니다. 평균 제곱 오차(MSE) 기준으로 기존 확산 기반 방법들보다 최소 65.2% 향상되었으며, 준우승자에 대해서도 최소 15.7%의 개선을 보였습니다.



### Can LVLMs and Automatic Metrics Capture Underlying Preferences of Blind and Low-Vision Individuals for Navigational Aid? (https://arxiv.org/abs/2502.14883)
Comments:
          26 pages, 12 figures, 14 tables

- **What's New**: 이 논문에서는 시각 장애인(Blind and Low-Vision, BLV) 사용자들이 내비게이션을 위해 Large Vision-Language Models (LVLMs)에서 어떤 유형의 응답을 선호하는지를 조사합니다. 이를 위해, Eye4B 데이터셋을 구축하여 1,100개의 야외 및 실내 장면에 대해 사람 검증을 거친 다양한 요청을 포함하였습니다. 또한, BLV 사용자의 편의를 위해 LVLM의 응답 스타일 및 그들의 선호도를 평가하기 위한 심층 사용자 연구를 수행하였습니다.

- **Technical Details**: BLV 사용자의 요구 사항을 충족하기 위해 5가지 관점(Afraidness, Nonactionability, Sufficiency, Conciseness, Overall)에서 6개의 LVLM 응답을 분석합니다. Eye4B 데이터셋은 고품질의 시각 장면을 제공하며, 다양한 BLV 요구를 기반으로 한 요청을 포함합니다. 우리는 관객의 반응을 이해하고 BLV-Aware AI 시스템을 개발하는 데 필요한 귀중한 데이터를 수집하기 위해 모델 기반의 이미지-텍스트 메트릭스와 BLV 선호 사이의 일치를 평가하는 Eye4B 벤치마크도 소개합니다.

- **Performance Highlights**: BLV 사용자의 선호도에 대한 분석 결과, 특정 LVLM은 사용자들에게 더 긍정적인 반응을 이끌어낸 것으로 나타났으며, 이는 실제 응용 프로그램 개발에 중요한 지침이 될 수 있습니다. 특히, Eye4B 벤치마크는 모델 기반 메트릭이 BLV의 선호를 얼마나 잘 반영하는지를 평가하기 위해 새로운 평가 차원을 제공하며, 이것은 앞으로의 연구에 중요한 발판이 될 것입니다.



### KKA: Improving Vision Anomaly Detection through Anomaly-related Knowledge from Large Language Models (https://arxiv.org/abs/2502.14880)
- **What's New**: 이번 연구에서는 Key Knowledge Augmentation (KKA)라는 새로운 방법을 제안합니다. KKA는 대형 언어 모델(LLMs)에서 이상치와 관련된 지식을 추출하여, 정상 샘플을 기반으로 하여 의미 있는 이상치를 생성합니다. 이 방식은 랜덤하게 생성된 이상치가 아닌, 실질적인 경계 설정을 위한 효과적인 이미지를 생성하여 검출기의 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: KKA는 정상 샘플에서 무작위 샘플링을 통해 LLM들에게 프롬프트를 제공하여 이상치 관련 지식을 생성합니다. 이 과정에서 생성된 이상치는 '쉬운 이상치'와 '어려운 이상치'로 분류되며, 후자는 정상 샘플과 유사한 형태로 검출기의 경계 학습에 더 큰 영향을 미칩니다. KKA는 검출기의 성능을 반복적으로 업데이트하며, 특히 어려운 이상치의 비율을 점진적으로 증가시킵니다.

- **Performance Highlights**: 실험 결과, KKA는 다양한 시각적 이상치 검출기에서 성능을 크게 개선했습니다. 예를 들어, CIFAR-100 데이터셋에서 KKA는 SimpleNet의 AUC를 74.62%에서 84.04%로 크게 향상시켰으며, 생성된 샘플의 수는 SimpleNet의 약 5%에 불과했습니다. 이는 KKA가 경계 설정에 있어 큰 효과를 가져왔음을 보여줍니다.



### Is Mathematics Obsolete? (https://arxiv.org/abs/2502.14874)
- **What's New**: 이 논문은 AI의 시대에서 수리적 및 기호적 추론의 가치에 대한 에세이입니다. 저자는 수학적 사고와 과학적 사고의 상호보완적인 관계를 강조하며, 기호적 AI와 신경망 AI의 두 가지 주요 접근 방식을 논의합니다. 또한, 최근 인공지능의 발전이 기호적 접근 방법에 미치는 영향을 심도 있게 탐구합니다.

- **Technical Details**: 저자는 아리스토텔레스와 플라톤의 철학적 대화에서 수학이 경험에 뿌리를 두고 있는 과학적 사고와 상반되는 점을 설명합니다. 기호적 AI(Artificial Intelligence)와 신경망 AI(Neural AI)의 대립을 선명하게 구분지으며, 기호적 사고가 초기 AI 연구에서 우위에 있었으나 딥러닝의 발전으로 신경망이 주도하게 된 흐름을 분석합니다.

- **Performance Highlights**: 논문은 AI 분야에서 기호적 방법이 어떻게 사라지는지 보여주며, 기계 학습과 신경망은 과거의 기호적 접근법을 지배하는 경향을 드러냅니다. 카네기 멜론 대학교의 역사와 AI 발전에서의 기여를 통해 이러한 변화를 구체적으로 설명하며, 딥 블루와 같은 기호적 AI가 알파제로와 대규모 언어 모델에 밀리고 있음을 강조합니다.



### Why do Experts Disagree on Existential Risk and P(doom)? A Survey of AI Experts (https://arxiv.org/abs/2502.14870)
Comments:
          In submission to AI and Ethics Journal. 24 pages total, 15 pages of writing with 9 pages of appendices

- **What's New**: 본 연구는 AI 전문가 111명을 대상으로 AI 안전 개념에 대한 친숙함, 주요 이의 제기, 안전 주장의 반응 등을 조사했습니다. 연구 결과, 전문가들은 '통제 가능한 도구로서의 AI'와 '통제 불가능한 에이전트로서의 AI'라는 두 가지 시각으로 클러스터링되었습니다. 흥미롭게도, 많은 전문가들이 'AI 연구자들은 재앙 위험에 관심을 가져야 한다'에 동의하면서도, 특정한 AI 안전 개념에 대한 이해는 부족했습니다.

- **Technical Details**: 조사는 AI 안전 연구에 대한 친숙함 및 AI 관련 우려에 대한 신념을 평가하기 위해 설계되었습니다. 응답자들은 머신러닝(ML) 관련 논문을 발표한 전문가들이며, 이들은 AI 안전 연구에 대한 이해도가 높지 않았습니다. 질문은 응답자의 전문 분야에 따라 구분되었으며, 각 개념에 대한 친숙도를 5점 척도로 평가했습니다.

- **Performance Highlights**: 연구에서 가장 혼란스러운 것은 AI 안전에 대한 개념이 부족한 참가자들로부터 가장 낮은 우려도가 나타났다는 점입니다. 오직 21%의 응답자만이 중요한 AI 안전 개념인 'instrumental convergence'에 대해 들어본 적이 있다고 응답했습니다. 따라서, 효과적인 AI 안전 커뮤니케이션은 명확한 개념적 기초를 수립하는 것에서 시작해야 할 필요가 있습니다.



### Envisioning Stakeholder-Action Pairs to Mitigate Negative Impacts of AI: A Participatory Approach to Inform Policy Making (https://arxiv.org/abs/2502.14869)
Comments:
          14 pages + supplementary information and appendix

- **What's New**: 이 논문은 AI의 부정적인 영향에 대한 책임 있는 관리 필요성이 증가함에 따라, 일반 시민(lay stakeholders)의 경험과 필요를 정책 개발의 중심에 두는 참여형 접근 방식을 제안합니다. 전통적인 위험 기반 접근 방식은 전문가 중심으로 이루어져 있어 일반 시민의 목소리가 반영되지 않는 문제가 있습니다. 이를 해결하기 위해 작성된 시나리오를 활용하여, 일반 대중이 AI의 부정적 영향을 이해하고 해결책을 제시할 수 있도록 돕는 방법론을 구축하였습니다.

- **Technical Details**: 논문은 생성적 AI가 미디어 환경에 미치는 부정적인 영향에 대한 잠재적 완화 전략을 체계적으로 수집하는 접근 방식을 제시합니다. 설문 조사를 활용하여 다양한 시나리오와 함께 일반 시민들의 반응을 수집하고, 이를 통해 부정적인 영향에 대한 완화 조치와 책임 배분을 도출합니다. 특히, 대규모 언어 모델(LLM)을 이용하여 정책 사실 시트를 생성하여 정책 입안자들에게 유용한 정보를 제공합니다.

- **Performance Highlights**: 이 연구의 접근 방식은, 일반 시민의 의견을 바탕으로 정책 개발 과정에서 더 다양한 의견을 통합할 수 있도록 하는 장점을 가지고 있습니다. 참여적 과정은 정책 입안자와 일반 시민 간의 신뢰를 구축하고, 그들이 기대하는 목표를 정Align하게 합니다. 여기서 제안하는 정책 사실 시트는 정책 과정에서 실증적 기반을 제공하는 간결한 요약으로 사용될 수 있습니다.



### Unlocking the Black Box: Analysing the EU Artificial Intelligence Act's Framework for Explainability in AI (https://arxiv.org/abs/2502.14868)
- **What's New**: 본 논문은 인공지능(AI)의 설명 가능성 부족이 산업과 규제기관이 극복해야 할 주요 장애물임을 강조합니다. 특히 의료, 신용 평가, 경찰 및 형사 사법 시스템과 같은 분야에서 XAI(eXplainable AI)의 필요성이 분명하게 드러납니다. 유럽연합(EU) 수준에서 설명 가능성 개념은 AI 법안의 기본 원칙 중 하나로 간주되지만, 구체적인 XAI 기법과 요구 사항은 여전히 실험 및 결정되지 않았습니다.

- **Technical Details**: 논문은 다양한 XAI 접근법 및 기법을 탐구하며, AI 거버넌스 및 정책에서 설명 가능성 원칙을 구현하는 데 따른 여러 도전 과제를 논의합니다. 설명 가능성을 조사하기 위한 방법론으로는 모델 해석(model interpretation), 인간 중심 설계(human-centered design), 신뢰성(recability) 등이 포함됩니다. 이와 관련된 표준 설정(standard setting), 감독(oversight) 및 시행(enforcement) 문제도 다룹니다.

- **Performance Highlights**: XAI의 중요성은 기술의 위험을 완화하기 위한 노력으로 인식되며, 논문에서는 EU 법에 XAI를 통합하는 과정에서의 어려움과 기회를 조명합니다. 설명 가능성을 갖춘 AI 시스템의 구현은 사회의 책임, 윤리 및 공정성을 보장하는 데 필수적입니다. 궁극적으로, 이 논문은 AI 기술의 실용적 적용을 위한 미래 방향을 제시하고 있습니다.



### d-Sketch: Improving Visual Fidelity of Sketch-to-Image Translation with Pretrained Latent Diffusion Models without Retraining (https://arxiv.org/abs/2502.14007)
Comments:
          Accepted in The International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문은 자유로운 손으로 그린 스케치를 기반으로 현실적인 이미지를 생성하는 새로운 기법을 제안합니다. 기존의 GAN(Generative Adversarial Networks) 방식이 아닌, 대규모로 사전 훈련된 확산 모델을 활용하여 구조적인 가이드를 제공합니다. 구체적으로, 학습 가능한 경량 매핑 네트워크를 도입하여 소스 도메인(스케치)과 타겟 도메인(이미지) 간의 특성 변환을 지원합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 잠재적 확산 모델의 특징 공간을 활용하여, 재훈련 없이도 고해상도의 사실적인 이미지 생성을 가능하게 합니다. 이 과정은 GAN 방식의 불안정성과 대규모 모델의 높은 계산 비용을 줄여주는 안정적인 최적화를 제공합니다. 연구에서는 딥러닝을 통해 추출된 특징을 활용하여 원본 스케치와 생성된 이미지 간의 상관관계를 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 기술들과 비교하여 질적 및 양적 기준에서 우수한 성능을 보였습니다. 특히, 고해상도의 사실적인 이미지 생성 능력에서 중요한 개선을 이루었으며, 사용자의 예술적 전문성에 관계없이 손으로 그린 스케치에서 매력적인 이미지를 생성할 수 있는 가능성을 보여주었습니다.



### High Quality Segmentation for Ultra High-resolution Images (https://arxiv.org/abs/2111.14482)
- **What's New**: 이번 논문에서는 4K 및 6K 초고해상도 이미지 세그멘테이션을 위한 Continuous Refinement Model (CRM)을 제안합니다. 기존의 다운샘플링(down-sampling)이나 패치 크롭(patch cropping) 같은 방법이 정확성과 계산 비용 사이의 균형 문제를 제대로 해결하지 못하는 점을 지적합니다. 이 모델은 객체를 점진적으로 세밀하게 구분하는 인간의 인식 방식을 바탕으로 합니다.

- **Technical Details**: Continuous Refinement Model (CRM)은 특징 맵(feature map)을 정제 목표(refinement target)와 지속적으로 정렬(alignment)시키며, 이미지의 세부 사항을 재구성하는 데 필요한 특징들을 집계(aggregate)합니다. 이 모델은 저해상도(training images)로 훈련된 데이터와 초고해상도(testing ones) 간의 해상도 차이를 메우는 데 뛰어난 일반화(generalization) 능력을 보입니다.

- **Performance Highlights**: CRM의 성능은 정량적인 평가(quantitative performance evaluation)와 시각화(visualization)를 통해 입증되었습니다. 이러한 방법은 이미지 세그멘테이션 정제 작업에서 빠르고 효과적임을 보여주었으며, 코드도 공개될 예정입니다.



### Temporal Misalignment in ANN-SNN Conversion and Its Mitigation via Probabilistic Spiking Neurons (https://arxiv.org/abs/2502.14487)
- **What's New**: 이 논문에서는 Spiking Neural Networks (SNNs)의 ANN-SNN 변환 과정에서 발생하는 'temporal misalignment'라는 현상을 밝혀냈습니다. 이 현상은 SNN 레이어 전반에서 랜덤 스파이크 재배치가 성능 향상을 초래한다는 것을 보여줍니다. 이를 기반으로, 두 단계 확률적 스파이킹 뉴런(Two-Phase Probabilistic spiking neurons)을 도입하여 변환 과정을 더욱 개선하고 있습니다.

- **Technical Details**: 기존의 Integrate-and-Fire (IF) 스파이킹 뉴런 모델을 기반으로 하여 ANN과 SNN 간의 변환 과정을 수식으로 설명합니다. 이 과정은 변환의 정확성을 높이며, 헤비스타이드 함수와 같은 수학적 요소를 통해 SNN의 내부 동역학을 모델링합니다. 논문에서는 ANN에서 SNN으로 가중치와 편향을 전송하는 방법을 자세히 다루고 있으며, 이 과정의 효용성을 강조합니다.

- **Performance Highlights**: CIFAR-10/100, CIFAR10-DVS, ImageNet 데이터셋을 통한 실험에서 제안한 방법이 최신 SOTA 변환 및 다른 훈련 방법들보다 우수한 정확도를 달성했습니다. 다양한 아키텍처를 통한 포괄적 실험을 통해 이 방법의 효과가 입증되었으며, 이로써 에너지 효율적인 AI 시스템 구축의 가능성을 제시합니다.



New uploads on arXiv(cs.LG)

### One-step Diffusion Models with $f$-Divergence Distribution Matching (https://arxiv.org/abs/2502.15681)
- **What's New**: 이번 논문에서는 일반적인 분포 매칭(distillation) 접근법을 확장하는 새로운 $f$-divergence 최소화 프레임워크인 $f$-distill을 제안합니다. 이 접근법은 다양한 divergence를 포괄하여 모드 커버리지(mode coverage)와 훈련 분산(training variance) 사이의 서로 다른 균형을 제공합니다. 기존의 reverse-KL divergence 기반 방법은 특정 모드에 초점을 맞출 수 있지만, $f$-distill은 이보다 더 다양한 diverge 모델을 통해 더 나은 샘플링을 가능하게 합니다.

- **Technical Details**: $f$-distill은 교사(teacher)와 학생(student) 분포 간의 다양한 $f$-divergence를 통해 서로 다른 데이터를 고려합니다. 우리가 제안하는 방법은 오히려 적은 모드를 추구하고, 훈련 과정에서 발생하는 분산을 줄이는 형태로 가능합니다. 또한, 각 divergence의 가중 함수(weighting function)에 따라 학생의 투표(score)가 교사의 높은 밀도가 있는 샘플에 더 많은 비중을 두게 하는 방식을 사용합니다.

- **Performance Highlights**: 경험적으로, 제안된 $f$-distill 방법은 이미지 생성 작업에서 이전의 최선의 variational score distillation 방법보다 더 높은 성능을 보입니다. 특히, Jensen-Shannon divergence를 사용했을 때는 ImageNet64에서의 1단계 생성 성능과 MS-COCO에서 제로샷 텍스트-이미지 생성에서 최신의 성과를 달성했습니다. 이 연구는 학생 분포가 교사 분포에 매칭하는 다양한 방식과 결과를 통한 실질적인 가이드를 제공합니다.



### Testing the limits of fine-tuning to improve reasoning in vision language models (https://arxiv.org/abs/2502.15678)
- **What's New**: 이번 연구에서는 사전 학습된 비전 언어 모델(Visual Language Models, VLMs)의 비주얼 인지(visual cognition) 능력을 향상시키기 위해 직관적 물리(intuitive physics)와 인과 추론(causal reasoning) 관련 데이터를 기준으로 모델을 미세 조정(fine-tuning)하는 방식을 제안합니다. 연구는 이러한 미세 조정이 특정 도메인에서 모델 성능을 개선하고, 인지 분야의 시작점으로 모델을 정렬할 수 있는지를 조사하고 있습니다. 또한 미세 조정이 다양한 비주얼 특성이나 새로운 인지 도메인에 대한 일반화 능력에는 부족함이 있음을 확인했습니다.

- **Technical Details**: 우리는 Cubeworld라는 가상 환경을 기반으로 직관적 물리 및 인과 추론 과제를 위한 네 가지 데이터 세트를 만들었습니다. 각 도메인에서 하나는 미세 조정을 위한 데이터 세트로, 다른 하나는 평가용 데이터 세트로 사용됩니다. 실험 결과, 미세 조정된 모델이 새로운 상황에서의 성능이나 인간의 판단과의 정렬 정도를 평가했으며, 환경과 시각적 특성이 다른 설정에서 모델이 어떻게 반응하는지를 분석했습니다.

- **Performance Highlights**: 연구 결과, 미세 조정된 VLM은 훈련된 데이터의 분포와 유사한 테스트 세트에서는 성능이 개선되었지만, 새로운 환경이나 인지 도메인에서는 제한된 일반화 능력을 보였습니다. 특히, 직관적 물리 문제 해결에서는 대체로 성능이 향상되었으나 인과 추론 문제에서는 인간의 행동 패턴과 잘 맞지 않는 경향을 나타냈습니다. 따라서, 비전 언어 모델의 미세 조정만으로는 인간과 유사한 일반화 능력을 발휘하는 데 한계가 있다는 결론에 이르렀습니다.



### Logit Disagreement: OoD Detection with Bayesian Neural Networks (https://arxiv.org/abs/2502.15648)
Comments:
          Presented at ECCV 2024 Workshop: 3rd Workshop on Uncertainty Quantification for Computer Vision

- **What's New**: 본 연구는 Bayesian Neural Networks (BNNs)에서 발생하는 epistemic (지식적 불확실성) 불확실성을 보다 정확하게 추정하기 위한 새로운 접근 방식을 제안합니다. 제안된 방법은 pre-softmax 값, 즉 logits의 수정된 버전 간의 불일치를 측정하여 epistemic 불확실성을 평가하는 것입니다. 이 논문은 다양한 OoD(out-of-distribution) 실험에서 mutual information (상호 정보량)보다 현저한 개선 효과를 보였습니다.

- **Technical Details**: Bayesian Neural Networks에서는 네트워크 가중치를 확률 변수로 보고, Variational Inference(변분 추론)를 통해 posterior distribution(사후분포)을 추정합니다. 이 연구에서는 posterior 샘플 간의 maximum logit 값의 불일치를 통해 epistemic 불확실성을 평가하는 쉽고 모델에 구애받지 않는 epistemic 불확실성 점수를 제안합니다. 또한 Bayesian Variational Autoencoder (BVAE) 문헌에서 이 점수를 기초로 한 분류적 접근을 탐색합니다.

- **Performance Highlights**: 제안된 epistemic 불확실성 점수는 MNIST 및 CIFAR10 실험에서 Bayesian benchmark인 predictive entropy와 유사한 성능을 보여줍니다. 특히, 다양한 OoD 실험에서도 mutual information보다 우수한 성능을 기록하여 모델의 불확실성 추정의 중요성을 강조합니다. 이 논문은 BNN을 사용하는 머신러닝 모델의 신뢰성과 안전성을 높이는 데 기여합니다.



### Predicting gene essentiality and drug response from perturbation screens in preclinical cancer models with LEAP: Layered Ensemble of Autoencoders and Predictors (https://arxiv.org/abs/2502.15646)
- **What's New**: 본 논문에서는 LEAP (Layered Ensemble of Autoencoders and Predictors)라는 새로운 앙상블 프레임워크를 소개합니다. 이 프레임워크는 약물 발견에 있어 중요한 유전자 발현 데이터와 예측 모델을 결합하여 다양한 생물학적 맥락에서의 예측 정확성을 개선합니다. LEAP는 여러 DAMAE (Data Augmented Masked Autoencoder) 표현 및 LASSO 회귀 분석기를 활용하여 단순한 예측 모델에 비해 성능이 우수하다는 것을 보여줍니다.

- **Technical Details**: LEAP는 서로 다른 초기화(random initialization)에서 학습된 다양한 유전자 발현 표현 모델을 결합하여 작동합니다. 이 방법은 예측 모델보다 표현 모델의 앙상블을 통해 더 나은 예측 성능을 제공합니다. LEAP는 계산 효율성이 뛰어나고 최소한의 하이퍼파라미터 튜닝(hyperparameter tuning)으로 약물 발견 파이프라인에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과는 LEAP가 이전에 보지 못한 세포주(cell lines), 조직(tissues) 및 질병 모델(disease models)에서 유전자 필수성(gene essentiality) 및 약물 반응(drug responses)을 예측하는 데 있어 최첨단 접근 방식을 일관되게 초월한다는 것을 보여줍니다. LEAP의 성능 향상 외에도 이 모델은 실험 우선순위 설정 및 바이오마커 기반 계층화(biomarker-driven stratification)를 지원하는 데 유용합니다.



### AutoTandemML: Active Learning Enhanced Tandem Neural Networks for Inverse Design Problems (https://arxiv.org/abs/2502.15643)
- **What's New**: 본 연구에서는 inverse design 문제를 해결하기 위해 active learning과 Tandem Neural Networks (TNN)을 결합한 새로운 하이브리드 접근법인 AutoTandemML을 제안합니다. 이 방법은 데이터셋 크기를 줄이면서도 정확성을 유지하여 inverse design 문제 해결의 효율성과 효과성을 강화할 수 있습니다. 연구팀은 에어포일(airfoil) 역설계, 광학 표면(photonics surface) 역설계 및 확산 부분 미분 방정식(scalar boundary condition reconstruction)에 대한 세 가지 벤치마크 문제를 통해 이 접근법의 효과를 검증하였습니다.

- **Technical Details**: AutoTandemML은 active learning 알고리즘과 TNN 구조를 통합하여 inverse design 문제를 효율적으로 해결하는 프레임워크입니다. Inverse design 모델의 수학적 정의와 함께 다중 출력 회귀를 위한 active learning 알고리즘을 설명합니다. TNN의 각 구성 요소와 이들이 어떻게 효율적으로 inverse 매핑을 추정하는 데 사용될 수 있는지를 명확히 설명합니다.

- **Performance Highlights**: AutoTandemML은 다른 샘플링 방법과 비교하여 성능에서 우수함을 입증하였으며, 더 적은 학습 샘플로도 높은 정확도를 달성하는 것으로 나타났습니다. 이는 TNN의 안정성과 간단한 훈련 구조 덕분에 가능하였으며, 다양한 디자인 파라미터를 정확히 예측하는 데 효과적입니다. 연구자들은 이 성능 개선이 새로운 연구 커뮤니티에서 활용될 수 있도록 데이터 저장소에 벤치마크와 AutoTandemML 도구를 공개했습니다.



### Training Neural ODEs Using Fully Discretized Simultaneous Optimization (https://arxiv.org/abs/2502.15642)
Comments:
          Accepted to the 14th IFAC Symposium on Dynamics and Control of Process Systems, including Biosystems (DYCOPS 2025)

- **What's New**: 이번 논문에서는 Neural Ordinary Differential Equations (Neural ODEs)의 훈련을 위한 새로운 동시 최적화 방법을 제안합니다. 기존의 훈련 방법과는 달리, 이 연구에서는 collocation 기반의 완전 이산화된 수식을 사용하여, 반복적인 시뮬레이션 대신 시스템 동력을 동등 제약 조건으로 해결합니다. 특히, IPOPT를 사용하여 collocation 계수와 신경망 매개변수를 동시에 최적화하여 훈련 시간을 단축시킵니다.

- **Technical Details**: Neural ODE는 연속 시간 동력을 신경망을 통해 표현하는 모델로, 관측 데이터를 근사하기 위해 매개변수 ODE 모델을 학습합니다. 특히 이번 연구는 비선형 최적화를 위해 Alternating Direction Method of Multipliers (ADMM)를 활용하여 데이터 배치 간 서브 모델을 효과적으로 조정합니다. Spectral numerical methods인 collocation을 도입하여 Neural ODE 훈련의 시간적 통합에 적용하며, 이를 통해 훈련 안정성을 향상시킵니다.

- **Performance Highlights**: Van der Pol Oscillator를 사례로 들어, 제안된 방법이 전통적인 훈련 방법에 비해 훈련 속도가 더 빠르고 수렴이 개선됨을 보여줍니다. 연구 결과는 collocation 기반의 동시 Neural ODE 훈련 파이프라인이 큰 잠재력을 지니고 있음을 나타내며, 모델의 경제성을 강조합니다. 제안된 방법을 통해 모델의 훈련 및 예측 효율성이 크게 개선될 것으로 기대됩니다.



### Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification (https://arxiv.org/abs/2502.15637)
- **What's New**: 최근 시계열 데이터의 분류를 위한 기초 모델( foundation model)의 개발이 증가하고 있습니다. 기존의 예측 지향 기초 모델들이 존재하는 반면, 시계열 분류를 위한 모델은 부족한 상황이었습니다. 이를 해결하기 위해, 우리는 Mantis라는 새로운 오픈소스 기초 모델을 소개합니다. Mantis는 Vision Transformer (ViT) 아키텍처에 기반하여 대비 학습(contrastive learning)을 통해 사전 학습되었습니다.

- **Technical Details**: Mantis는 시간 시리즈 분류를 위한 인코더 모델로, 수학적으로 𝐹:ℝᵗ→ℝᵏ의 형태로 표현됩니다. 이 모델은 고정된 시퀀스 길이를 가진 입력 데이터를 변환하여 차별적인 숨은 공간으로 매핑합니다. 사전 학습 단계에서는 큰 데이터 셋을 이용하여 모델이 다양한 작업에 잘 일반화될 수 있는 임베딩을 학습하며, fine-tuning 단계에서는 레이블이 있는 데이터를 사용하여 최종 분류기를 학습합니다.

- **Performance Highlights**: Mantis는 기존의 기초 모델들과 비교하여 뛰어난 성능을 보였습니다. 특히 백본이 고정되었을 때와 fine-tuning이 이루어졌을 때 모두 우수한 결과를 도출했습니다. 추가적으로, Mantis는 가장 낮은 보정 오차(calibration error)를 기록하였으며, 메모리 요구 사항을 줄이고 여러 변수의 상관관계를 처리할 수 있는 여러 개의 어댑터(adapter)를 제안했습니다.



### The Relationship Between Reasoning and Performance in Large Language Models -- o3 (mini) Thinks Harder, Not Longer (https://arxiv.org/abs/2502.15631)
Comments:
          19 pages, 11 figures

- **What's New**: 본 연구는 최신 대형 언어 모델(Large Language Models, LLMs)이 수학적 추론에서의 정확성과 체인-오브-씽크(Chain-of-Thought) 사용 간의 관계를 분석합니다. 특히, o1-mini와 o3-mini 모델을 Omni-MATH 벤치마크에서 비교하여 o3-mini가 더 길지 않은 체인으로도 높은 정확도를 달성함을 발견했습니다. 이 연구는 새로운 세대의 모델이 더욱 효율적으로 추론하는 방식을 사용하고 있음을 입증하는 중요한 통찰을 제공합니다.

- **Technical Details**: 연구는 LLMs의 수학적 추론 능력을 검토하기 위해 442,844의 문제로 구성된 Omni-MATH 데이터셋을 활용했습니다. 모델의 성능을 평가하기 위해 gpt-4o, o1-mini, o3-mini (m), o3-mini (h) 등 네 가지 OpenAI 모델을 사용하였으며, 각 모델에서 수집된 데이터는 자동화된 방식으로 검토되었습니다. 체인-오브-씽크의 토큰 사용과 난이도에 따른 정확도 변화를 분석했으며, 더 많은 토큰을 사용할수록 정확도는 감소하는 경향이 있는 것을 확인했습니다.

- **Performance Highlights**: 모델 성능 측면에서 o1-mini는 모든 범주에서 40%에서 60%의 정확도를 달성하는 반면, o3-mini (m)은 평균 50%의 정확도를 기록했습니다. o3-mini (h)은 o3-mini (m)에 비해 평균 4%의 정확도 증가를 보여주며, Algebra와 Calculus에서 80% 이상의 정확도를 기록했습니다. 그러나 디스크리트 수학(Digital Mathematics) 영역에서는 일반적인 성능 경향과 차이가 있었습니다.



### PDeepPP:A Deep learning framework with Pretrained Protein language for peptide classification (https://arxiv.org/abs/2502.15610)
Comments:
          10 pages, 5 figures, submitted to arXiv

- **What's New**: 이번 논문에서는 단백질 변형 및 생리활성 펩타이드의 예측을 위해 pre-trained 단백질 언어 모델과 결합한 새로운 딥러닝 프레임워크 PDeepPP를 소개합니다. 이 프레임워크는 transformer와 CNN 아키텍처를 조합하여 성능을 극대화하고, 다수의 벤치마크 데이터셋에서 우수한 성능을 보였습니다. 또한, 대량의 단백질 시퀀스 데이터를 효율적으로 처리하여 데이터의 복잡성을 잘 포착합니다.

- **Technical Details**: PDeepPP는 transformer와 CNN 기반의 병렬 신경망을 활용하여 local 및 global 시퀀스 복잡성을 포착합니다. 이 모델은 Masked Language Modeling (MLM) 방식으로 미리 훈련된 ESM-2 모델을 이용하여 다양한 PTM 사이트와 생리활성 펩타이드를 예측하는 데 사용됩니다. 이에 따라, 데이터 균형을 맞추기 위해 Transductive Information Maximization (TIM) 손실 함수를 결합하여 불균형한 데이터셋에서도 효율적인 예측을 가능하게 합니다.

- **Performance Highlights**: 모델은 33개의 작업 중 25개에서 최첨단(State-of-the-Art) 성능을 기록하며 기존 방법들을 초월했습니다. PDeepPP는 정확도와 강건성을 개선하면서 false positive 및 false negative를 줄였습니다. 이 연구는 펩타이드 발견 및 PTM 분석을 위한 혁신적인 도구로서의 가능성을 제시합니다.



### Improving the Scaling Laws of Synthetic Data with Deliberate Practic (https://arxiv.org/abs/2502.15588)
- **What's New**: 이 논문에서는 인간의 학습 원리 중 하나인 고의적 연습(Deliberate Practice)에서 영감을 받아 합성 데이터 생성을 위한 새로운 프레임워크인 DP를 제안합니다. DP는 동적인 합성 데이터 생성을 통해 샘플 효율성을 향상시킵니다. 기존 연구에서는 합성 데이터를 단순히 추가하는 것이 한계가 있으며, 이 문제를 해결하기 위해 가장 유용한 합성 샘플에 집중하는 가지치기가 중요하다고 강조하였습니다.

- **Technical Details**: DP 프레임워크는 초기 합성 데이터 세트로 시작하여, 실제 검증 세트에서 성능이 정체될 때 새로운 도전적인 예제를 생성하는 과정입니다. 이는 모델의 예측 불확실성을 활용하여 생성 프로세스를 안내함으로써, 불필요한 데이터의 생성을 줄입니다. 이 프레임워크는 전통적인 데이터 생성 방식과는 다르게, 모든 데이터를 일괄 생성하는 것이 아니라, 정보가 풍부한 데이터만을 동적으로 추가해 나갑니다.

- **Performance Highlights**: DP는 ImageNet-100에서 3.4배 적은 샘플로 6배 더 적은 반복횟수를 요구하면서도 우수한 성능을 보였고, ImageNet-1k에서는 8배 적은 샘플로 30%의 반복 횟수를 줄였습니다. 또한 OOD 데이터셋에서 뛰어난 성능을 발휘하며, 실제 데이터로 훈련된 모델에 비해 최대 15%의 성능 향상을 보여주었습니다.



### A Cautionary Tale About "Neutrally" Informative AI Tools Ahead of the 2025 Federal Elections in Germany (https://arxiv.org/abs/2502.15568)
- **What's New**: 이 연구에서는 AI 기반 투표 조언 애플리케이션(VAAs)과 대형 언어 모델(LLMs)의 객관적인 정치 정보 제공 신뢰성에 대해 조사하였습니다. 독일의 잘 알려진 온라인 도구인 Wahl-O-Mat와의 비교 분석을 통해 AI 시스템의 편향성을 심층적으로 분석하였습니다. 이는 정치적 정보의 정확성과 신뢰성에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 대형 언어 모델은 일반적으로 좌측 정당에 대해 평균 75% 이상의 높은 일치를 보이는 반면, 중우파 정당과는 50% 미만, 우파 정당과는 약 30%의 낮은 일치를 보였습니다. VAAs의 경우, Wahl-O-Mat에서 제시된 정당의 입장에 대한 상당한 편차를 보였습니다; 한 VAA는 25%의 경우에서 편차를 보였고, 다른 VAA는 50% 이상의 경우에서 편차를 나타냈습니다.

- **Performance Highlights**: 이 연구는 VAAs와 LLMs의 신뢰성 문제를 강조합니다. 특히 한 VAAs의 경우, 정치적 당과의 비존재적 연결과 같은 심각한 왜곡이 발생했습니다. 이는 AI 시스템의 개발 및 적용 과정에서의 객관성 확보의 필요성을 부각하는 중요한 발견입니다.



### Model Privacy: A Unified Framework to Understand Model Stealing Attacks and Defenses (https://arxiv.org/abs/2502.15567)
- **What's New**: 본 논문은 머신러닝(ML)의 모델 도용 공격(model stealing attacks)과 방어 메커니즘의 안전성 분석을 위한 새로운 프레임워크인 '모델 프라이버시(model privacy)'를 제안합니다. 기존의 연구들은 대개 실험적 접근을 취했으나, 이 논문은 이론적 기초를 확립하여 공격 및 방어 전략의 효과를 정량화하는 분석 방법론을 제공합니다. 이 연구는 ML 모델의 취약성 문제와 이로 인해 발생할 수 있는 재정적 손실 및 지적 재산권 침해의 심각성을 강조합니다.

- **Technical Details**: 논문에서는 공격자와 방어자의 행동 및 목표를 형식적으로 정의하고, 질의-응답 기반의 상호작용에서 의사결정에 영향을 미치는 기본 요소를 분석합니다. 또한, 공격 및 방어 전략의 유효성을 정량화하는 방법을 제안하여, 네 가지 대표적인 학습 알고리즘에 대해 프라이버시와 유용성(utility) 간의 트레이드오프를 평가합니다. 이론적 접근 방식은 특히 방어자의 시각에서 공격의 최적 방어를 분석하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방어 메커니즘이 모델 프라이버시를 크게 향상시킨다는 것을 입증하며, 모델 도용 공격을 더 어렵게 만들거나 아예 불가능하게 만듭니다. 특히, 공격자에게 더 많은 질의를 요구함으로써 ML 모델의 복제가 어려워지는 결과를 보여줍니다. 따라서, 모델 소유자가 재정적 손실을 방지하고 자신의 지적 재산을 보호할 수 있는 기반을 제공합니다.



### Solving Inverse Problems with Deep Linear Neural Networks: Global Convergence Guarantees for Gradient Descent with Weight Decay (https://arxiv.org/abs/2502.15522)
- **What's New**: 이 논문은 딥 뉴럴 네트워크가 gradient descent와 weight decay 정규화로 훈련될 때, 저차원 구조에 적응하는지 조사합니다. 저자들은 과다파라미터화된 딥 선형 네트워크가 역문제를 정확하게 해결하는 근사해로 수렴하며, 잠재 서브스페이스 구조를 암묵적으로 인코딩한다는 것을 증명했습니다. 이는 딥 선형 네트워크가 데이터의 잠재 서브스페이스 구조에 자동으로 적응됨을 보이는 최초의 결과로, 실용적인 스텝 사이즈와 가중치 초기화 방식 하에서 적용됩니다.

- **Technical Details**: 연구의 배경으로, 신호의 저차원 구조를 가정하는 것이 일반적인 해결책임을 언급하고, 정규화 문제를 해결하기 위한 최적화 접근 방식을 설명하였습니다. 저자들은 정규화된 최소화 문제를 다루며, 이미지 서브스페이스의 신호를 재구성하는 방법에 대해 이야기합니다. 논문에서는 gradient descent를 통해 훈련된 완전히 연결된 딥 선형 신경망이 최적의 근사해로 수렴한다고 결과를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, gradient descent를 통한 학습이 저차원 구조를 갖는 신호의 재구성에서 정확도를 제공하며, 정규화를 통해 더욱 강력한 솔루션으로 이어진다고 설명합니다. 특히, 과다파라미터화가 일반화 성능을 향상시키고, 훈련 중 수렴 속도를 가속화하는 것으로 나타났습니다. 전통적인 비정규화 접근법에 비해 정규화된 접근법이 성능 향상에 긍정적인 영향을 미친다는 것을 강조합니다.



### SALSA-RL: Stability Analysis in the Latent Space of Actions for Reinforcement Learning (https://arxiv.org/abs/2502.15512)
- **What's New**: SALSA-RL (Stability Analysis in the Latent Space of Actions)는 행동을 잠재 공간에서 동적, 시간 의존적인 변수로 모델링하여 DRL의 안정성 분석을 가능하게 하는 새로운 RL 프레임워크입니다. 이 접근법은 미리 학습된 인코더-디코더와 상태 의존형 선형 시스템을 사용하여 안정성과 해석 가능성을 동시에 제공합니다. SALSA-RL은 기존 RL 에이전트의 행동의 국소 안정성을 평가하는 비침해적 방식으로 배치될 수 있습니다.

- **Technical Details**: SALSA-RL은 행동을 잠재 공간에서 지속적으로 진화하는 동적 시스템으로 모델링합니다. 이는 상태 정보에 기초하여 상태 의존형 동적 행렬을 통해 작동하며, 이를 통해 행동 동역학의 현지 안정성 분석을 수행할 수 있습니다. 또한, SALSA-RL은 SAC, DDPG, TD3와 같은 기존의 DRL 방법들과 호환되어 표준 훈련 전략을 유지하면서도 경쟁력 있는 제어 성능을 제공합니다.

- **Performance Highlights**: SALSA-RL을 통해 제어 정책의 안정성과 행동 동역학의 지역 안정성을 평가하며, 행동 변화가 상태 진화에 미치는 영향을 분석할 수 있습니다. 이로 인해 높은 위험 영역을 식별하고, 정책 검증을 향상시키는 데 기여합니다. 제어 시스템 내의 상태와 행동 사이의 관계를 명확하게 이해함으로써, 보상 설계 및 정책 개선에 대한 통찰력을 제공할 수 있습니다.



### Activation Steering in Neural Theorem Provers (https://arxiv.org/abs/2502.15507)
- **What's New**: 이번 연구에서는 고급 언어 모델(LLM)이 증명 보조 도구를 사용할 때의 수학적 증명의 단계 예측에서 발생하는 문제를 해결하기 위한 새로운 접근 방식으로 'activation steering' 기법을 제안합니다. 기존 모델들은 특정 전술(tactic)을 예측하는 데 성공하지만, 후보 전술 내에서 이를 적절하게 순위화하는 데 어려움을 겪고 있습니다. Activation steering을 통해 LLM의 응답을 안내하고, 추론 시 생성 품질을 개선하는 가능성을 모색합니다.

- **Technical Details**: 이 연구는 Llemma와 InternLM2와 같은 특정 LLM 모델에 대해 수학적 데이터를 기반으로 한 훈련 및 미세 조정을 통해 전술 예측을 개선하고자 합니다. Activation steering은 모델의 내부 표현을 수정하여 원하는 출력으로 이끌어내는 방법으로, 정확성과 해석 가능성을 높입니다. 이 기법은 LLM의 추론 과정을 체계적으로 영향을 주어 보다 신뢰할 수 있는 예측이 가능하도록 합니다.

- **Performance Highlights**: 실험 결과, activation steering 기법이 기존의 전문화된 미세 조정 방법보다도 더 경량화된 대안으로 제시되며, 자원 제약이 있는 환경에서도 정리된 증명 생성을 가능하게 하는 잠재력을 가지고 있음을 보여주었습니다. 이 기법은 특히 대화형 정리 증명(interactive theorem proving)에서 전술 선택 과정을 보다 정확하고 알기 쉽게 만들어 줄 수 있는 기회를 제공합니다.



### Verification and Validation for Trustworthy Scientific Machine Learning (https://arxiv.org/abs/2502.15496)
- **What's New**: 이 논문에서는 과학적 기계 학습(Scientific Machine Learning, SciML) 모델의 신뢰성을 높이기 위한 모범 사례 확립을 논의하고 있습니다. SciML의 발전이 빨라지고 있으나, 이에 대한 신뢰할 수 있는 개발 관행이 미비하다는 문제를 지적하고 있습니다. 제안된 16가지의 권장 사항은 연구자들이 예측 SciML을 개발하고 적용하는 데 도움이 되도록 마련되었습니다.

- **Technical Details**: SciML은 물리적 시스템의 수치 시뮬레이션을 향상시키기 위해 ML을 통합한 분야입니다. SciML은 머신러닝(ML)과 계산 과학 및 공학(Computational Science and Engineering, CSE)의 병합을 기반으로 하며, 모델 검증 및 검증(Verification and Validation, V&V) 기준을 적응하여 신뢰할 수 있는 모델 개발을 강조하고 있습니다. SciML 모델의 세 가지 기본 속성은 성능, 신뢰성 및 프로세스에 대한 투명성을 포함합니다.

- **Performance Highlights**: 이 논문은 SciML의 개발 및 적용을 위한 프레임워크를 제안하고, 이를 통해 다양한 과학적 문제 해결을 목표로 합니다. 특히, 서프리겟 모델링(surrogate modeling), 모델 발견(model discovery), 혼합 CSE-SciML 모델링 및 외부 루프 학습(outer-loop learning)이라는 네 가지 주요 방법론을 강조합니다. 연구자들이 효과적으로 SciML을 활용하도록 돕기 위한 포괄적인 접근 방식을 제시하고 있습니다.



### Network Resource Optimization for ML-Based UAV Condition Monitoring with Vibration Analysis (https://arxiv.org/abs/2502.15491)
Comments:
          Accepted for publication in IEEE Networking Letters

- **What's New**: 본 연구는 스마트 시티와 같은 새로운 무인 항공기(UAV) 응용 프로그램에서 실시간 상태 모니터링(Condition Monitoring, CM)의 필요성을 강조합니다. 본 논문은 기계 학습(Machine Learning, ML) 모델을 활용하여 이상 감지(Anomaly Detection, AD)와 건강 분석을 통해 UAV의 작동 안전성을 확보하는 방법을 다룹니다. 특히, 네트워크 리소스 최적화를 통해 ML 기반 UAV CM 프레임워크의 효율성을 개선하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 센서 데이터의 진동 분석을 바탕으로 하며, 다양한 특성 추출 집계 시간 간격을 조정하여 네트워크 자원 소비를 최적화합니다. 이 과정에서 차원 축소 기술이 활용되어 99.9%의 네트워크 자원 소비 감소를 달성합니다. 또한, ADXL345 센서를 사용하여 UAV의 진동을 캡처하며, ESP32 마이크로컨트롤러를 통해 데이터를 수집하고 전송합니다.

- **Performance Highlights**: 실험 결과는 빅데이터 관리에서의 네트워크 리소스 최적화가 ML 모델의 성능에 미치는 긍정적인 영향을 보여줍니다. 다양한 집합 크기를 조사한 결과, 최적의 데이터 집합 크기를 결정함으로써 ML 모델 성능을 극대화하는 동시에 소비되는 네트워크 자원을 최소화할 수 있었습니다. 궁극적으로, 본 연구는 네트워크 자원 활용 효율성과 ML 모델 성능 간의 균형을 탐색하는 기초를 제공합니다.



### MoMa: A Modular Deep Learning Framework for Material Property Prediction (https://arxiv.org/abs/2502.15483)
- **What's New**: 이 논문에서는 MoMa라는 새로운 모듈형 프레임워크를 소개합니다. MoMa는 다양한 소재 특성을 적절하게 예측하기 위해 전문화된 모듈을 먼저 훈련하고, 후속 작업에 맞게 조합합니다. 이 프레임워크는 기존의 사전 훈련(pre-train)과 파인 튜닝(fine-tune) 방식의 한계를 극복하고, 특정 작업에 최적화된 성능을 놓고 하는 진화를 보여줍니다. MoMa는 오픈 소스 프로젝트로 커뮤니티의 협력을 통해 발전할 예정입니다.

- **Technical Details**: MoMa는 소재 특성 예측에서의 다양성과 불일치를 해결하기 위한 2단계 프로세스를 가지고 있습니다. 첫 번째 단계에서는 다양한 소재 데이터셋에서 전문 모듈을 훈련하여 MoMa Hub에 중앙 집중화합니다. 두 번째 단계에서는 Adaptive Module Composition (AMC) 알고리즘을 통해 이 모듈들을 결합하여 최적의 조합을 활용, 각 하류 작업에 맞춰 조정합니다. MoMa는 기존의 정적 모델과 달리, 서로 다른 작업 간의 간섭을 제거하여 핵심 지식을 효율적으로 재사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, MoMa는 17개의 하류 작업 중 16개에서 기존의 모든 베이스라인을 초과한 성능을 보이며, 평균적으로 14%의 성능 개선을 기록했습니다. 특히, 몇 개의 데이터 포인트만을 사용하는 few-shot 설정에서 MoMa는 기존의 방식보다 훨씬 더 큰 성능 향상을 보여줍니다. MoMa는 지속적인 학습 환경에서도 탁월한 능력을 발휘하며, 자료 특성 예측 작업을 수행하는 데 있어 혁신적인 접근 방식을 제공합니다.



### Decoding for Punctured Convolutional and Turbo Codes: A Deep Learning Solution for Protocols Complianc (https://arxiv.org/abs/2502.15475)
- **What's New**: 본 논문에서는 LSTM(Long Short-Term Memory) 기반의 통합 디코딩 아키텍처를 제안하여, 기존의 점화 코드(punctured codes)의 복잡함을 해결하고자 합니다. 이 새로운 접근법은 점화 합동(convolutional) 코드와 터보(Turbo) 코드를 통합하여 다양한 코드율(code rates)에 적응할 수 있도록 설계되었습니다. 점화 패턴(puncturing patterns)을 네트워크에 직접 통합하는 점화 임베딩(puncture embedding) 메커니즘은 프로토콜 프로그래밍(protocol flexibility)을 유지하면서 강력한 성능을 제공하는 것을 목표로 합니다.

- **Technical Details**: 제안된 LSTM 기반의 디코더는 점화 합동 코드와 터보 코드에 대한 프로토콜 호환성을 보장하며, 높은 동적 변화에서도 최적화된 성능을 발휘합니다. 네트워크의 숨겨진(latent) 공간에 점화 패턴을 기록하는 점화 임베딩 모듈은 다양한 코드율에 즉각적으로 적응할 수 있는 능력을 제공합니다. 또한, 균형 비트 오류율 훈련(Balanced Bit Error Rate Training) 전략을 통해 모든 코드율이 넘어선 일반화 성능을 갖도록 구성되었습니다.

- **Performance Highlights**: 제안된 신경 디코더는 첨가 백색 가우시안 소음(Additive White Gaussian Noise) 및 레일리(Rayleigh) 페이딩 채널에서 최첨단 성능을 보여줍니다. 광범위한 시뮬레이션 결과에 따르면, 제안된 모델은 전통적인 디코딩 알고리즘을 능가할 뿐만 아니라 완벽한 채널 상태 정보(PCSI)를 가진 기존 디코더보다도 더 나은 성능을 발휘합니다. 이러한 성과는 LSTM 기반의 디코딩이 차세대 인공지능 통신 시스템에 대한 유망한 솔루션임을 시사합니다.



### Mitigating Data Scarcity in Time Series Analysis: A Foundation Model with Series-Symbol Data Generation (https://arxiv.org/abs/2502.15466)
- **What's New**: 이번 논문에서는 시간 시계열 분석(Time Series Analysis, TSA)을 위한 기초 모델이 제안되었습니다. 기존의 데이터 부족(data scarcity) 및 데이터 불균형(data imbalance) 문제를 해결하기 위해 시간 시계열의 의미론적 설명인 상징적 표현(symbolic expressions)을 활용한 복잡한 시스템 모델링을 고려합니다. 이를 기반으로 고품질 시간 사례 데이터를 생성할 수 있는 시리즈 심볼(Series Symbol, S2) 이중 모듈리티(data generation mechanism) 생성 메커니즘을 도입합니다.

- **Technical Details**: S2 데이터를 기반으로 구축된 SymTime은 TSA를 위한 사전 학습(pre-trained) 기반 모델입니다. 이 모델은 다섯 가지 주요 TSA 작업에 대해 경쟁력 있는 성능을 보이며 하위 작업(downstream task)에 맞춰 미세 조정(fine-tuned) 시 실제 데이터셋에서 사전 학습된 모델에 필적하는 성능을 발휘합니다. 이러한 접근 방식은 이중 모듈리티 데이터 생성과 사전 학습(pretraining) 메커니즘의 중요성을 강조합니다.

- **Performance Highlights**: SymTime은 기존의 기초 모델들보다 우수한 성능을 보여주며, 데이터 부족 문제를 극복하는 데 기여할 수 있는 잠재력을 가집니다. 이 모델의 성능은 다양한 TSA 작업에서 입증되었으며, 향후 연구에 있어 유의미한 진전을 이룰 수 있는 기준을 제공할 것입니다.



### R-LoRA: Random Initialization of Multi-Head LoRA for Multi-Task Learning (https://arxiv.org/abs/2502.15455)
Comments:
          9 pages, 10 figures

- **What's New**: 이 논문에서는 R-LoRA라는 새로운 방법을 제안하여, Low-Rank Adaptation(LoRA) 방법의 다중 과제 학습(multi-task learning) 능력을 향상시킵니다. R-LoRA는 Multi-Head Randomization을 도입하여 여러 헤드 행렬을 다양화함으로써, 특정 과제의 기능을 더 효율적으로 학습할 수 있도록 합니다. 이러한 접근 방식은 초기 파라미터의 대칭성을 깨뜨려 보다 다양한 최적화 경로를 제공함으로써, LLMs의 학습 성능을 개선합니다.

- **Technical Details**: R-LoRA는 HydraLoRA의 비대칭 아키텍처를 활용하여 하나의 공유된 down-projection 행렬 A와 여러 개의 과제별 head 행렬 B를 정의합니다. 다중 헤드 랜덤화는 특히 head 행렬의 초기값에 무작위성을 추가하여 학습의 다양성을 제공합니다. 이 방식은 잘 알려진 LoRA의 제한을 극복해, 복잡한 데이터셋에서 더 나은 과제특화 지식(task-specific knowledge) 학습을 가능하게 합니다.

- **Performance Highlights**: R-LoRA는 다중 과제 시나리오에서 과제 특화 지식을 효과적으로 포착하여 성능을 향상시킵니다. 실험 결과, R-LoRA는 이전의 LoRA 방법보다 다중 과제 학습에서 우수한 성과를 보이며, 단일 과제 환경에서도 적절한 성능 향상을 이루었습니다. 이러한 개선은 다양한 작업에서 R-LoRA의 활용 가능성을 보여줍니다.



### A fast convergence algorithm based on binary integer programming for expert load balancing in MoE LLMs (https://arxiv.org/abs/2502.15451)
- **What's New**: 이번 논문에서는 MoE (Mixture-of-Expert) 모델에서의 전문가 로드 불균형 문제를 해결하기 위해 BIP 기반 밸런싱(BIP-Based Balancing) 알고리즘을 제안합니다. 이 알고리즘은 이진 정수 프로그래밍(BIP) 기반으로, 추가적인 벡터 q를 유지함으로써 전문가들의 부하를 효과적으로 균형 잡을 수 있는 방법을 제공합니다. 기존의 보조 손실(auxiliary loss) 방식을 사용하지 않기 때문에 모델 성능에 미치는 부정적인 영향을 줄일 수 있습니다.

- **Technical Details**: BIP 기반 밸런싱 알고리즘은 각 라우틴 게이트 내에서 온라인으로 q 벡터를 업데이트하여 top-K 순위를 조정합니다. 이 과정은 매우 짧은 시간 내에 이진 정수 프로그래밍을 해결하여 이루어집니다. 이는 기존의 실험 기반 방법과는 달리, 효율적으로 로드 밸런싱을 수행하며, 더 나아가 동시다발적인 전문가 매칭의 문제와 연결되어 있습니다.

- **Performance Highlights**: 시뮬레이션 실험 결과에 따르면 BIP 기반 밸런싱은 불균형 문제를 매우 빠르게 해결하면서도 최종 라우틴 점수 합계를 거의 변화시키지 않습니다. 알고리즘은 전문가 로드 밸런스와 사전 훈련 효율성 간의 거의 완벽한 균형을 이뤄, MoE LLM의 성능을 극대화합니다.



### Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning (https://arxiv.org/abs/2502.15436)
Comments:
          Raghav Singhal and Kaustubh Ponkshe contributed equally to this work

- **What's New**: 이번 연구에서는 LoRA 기반의 연합 Fine-Tuning인 Fed-SB를 제안하여, 전통적인 방법에서 발생하는 비효율적인 업데이트 문제를 해결합니다. Fed-SB는 LoRA-SB를 기반으로 하여, 효율적인 통신 비용 감소와 동시에 높은 성능을 달성합니다. 특히, 통신 비용을 최대 230배까지 줄이면서도 상황에 따라 뛰어난 성능을 보여줍니다.

- **Technical Details**: Fed-SB는 각 클라이언트가 학습한 매트릭스 R을 단순 평균하여 최적의 업데이트를 보장하는 구조로 되어 있습니다. 이는 기존의 LoRA 기반 프레임워크보다 계산 및 통신 효율성을 극대화하며, 개인 정보 보호 측면에서도 향상된 성능을 제공합니다. 추가적으로, Fed-SB는 Differential Privacy의 요구에 맞춰 더 적은 학습 가능한 파라미터를 유지하여, 추가적인 노이즈를 줄이는 데 성공했습니다.

- **Performance Highlights**: Fed-SB는 다양한 벤치마크에서 기존 방법들을 지속적으로 초월하는 성능을 입증했습니다. 실험 결과, 개인 데이터와 비공개 데이터 설정 모두에서 통신 오버헤드를 크게 줄이면서 뛰어난 결과를 보여줍니다. 이에 따라 Fed-SB는 연합 학습에서의 커뮤니케이션 비용과 성능 간의 새로운 Pareto 전선(Pareto frontier)을 설정하며, 효율적이고 확장 가능한 솔루션을 제공합니다.



### Single-pass Detection of Jailbreaking Input in Large Language Models (https://arxiv.org/abs/2502.15435)
Comments:
          Accepted in TMLR 2025

- **What's New**: 이번 연구에서 우리는 Single Pass Detection(SPD)이라는 새로운 기법을 소개하며, 이는 LLM을 위한 효율적인 방어 메커니즘입니다. SPD는 로그(logit) 정보를 활용하여 harmful한 공격을 단 한 번의 forward pass로 식별할 수 있습니다. 기존 방법들이 여러 회의 요청이나 보조 LLM을 요구하는 것과 달리, SPD는 이러한 요구를 줄여 계산 비용을 최소화합니다. 이 방법은 오픈 소스 모델에 대한 효과적인 탐지를 제공함과 동시에, 무해한 입력의 오분류를 최소화하는 장점을 지닙니다.

- **Technical Details**: SPD는 모델의 출력 토큰에 대한 로그의 분포 차이를 이용하여 악성 입력과 무해한 입력을 구별합니다. 이 기법은 LLM의 응답에서 나타나는 로그 분포의 차이를 활용하여, 한 번의 forward pass로 공격을 탐지할 수 있도록 설계되었습니다. 게다가, SPD는 GPT-3.5 및 GPT-4와 같은 모델에서 로그에 대한 완전한 접근 없이도 여전히 효과적인 성능을 발휘함을 보여줍니다. 이는 기존 방어 기법에 비해 높은 효율성과 탐지율을 유지합니다.

- **Performance Highlights**: 우리의 평가에 따르면 SPD는 Llama 2, Llama 3, Vicuna와 같은 오픈 소스 LLM에서 매우 높은 탐지율을 기록했습니다. 기존 방어 방법들과 비교했을 때, SPD는 처리 속도와 정확성을 모두 향상시킬 수 있음을 보여줍니다. 또한, 모델의 전체 로그에 대한 접근 없이도 SPD가 유망한 방어 방법으로 기능할 수 있음을 검증했습니다.



### Evaluate with the Inverse: Efficient Approximation of Latent Explanation Quality Distribution (https://arxiv.org/abs/2502.15403)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문은 기존의 설명 품질 평가 방법에 대한 대안을 제시합니다. 그 대안은 Quality Gap Estimate (QGE)라는 방법론으로, 특정 설명의 품질을 다른 잠재적 설명과 비교하여 평가합니다. 이 방법은 설명의 질에 대한 보다 직접적인 비교를 가능하게 하며, 기존의 랜덤 생성된 설명과 비교할 필요가 없습니다.

- **Technical Details**: QGE는 입력 𝐱와 예측 레이블 𝑦에 대한 함수 𝑓에 대한 설명 함수를 통해 설명의 품질을 측정하는 방법입니다. 논문에서는 다양한 기존 품질 지표를 활용하여 QGE를 적용함으로써 층형, 로컬화, 강인성 측면에서 설명의 품질을 평가합니다. 이 방법은 컴퓨터 작업 비즈니스에 미치는 영향을 최소화하면서 기존의 품질 지표들을 더 유의미하게 만듭니다.

- **Performance Highlights**: QGE 방법은 다양한 모델 아키텍처와 데이터셋에서 실험을 통해 확인된 바에 따르면 기존 품질 측정 방법에 비해 통계적 신뢰성을 높입니다. 연구 결과, QGE는 XAI(설명 가능한 인공지능) 실무자에게 더 신뢰할 수 있는 해석 도구를 제공합니다. 기존의 평가 방법에 비해 QGE는 효율적인 평가 메트릭을 제공함으로써 설명의 질을 보다 명확하게 비교할 수 있습니다.



### Learning Chern Numbers of Topological Insulators with Gauge Equivariant Neural Networks (https://arxiv.org/abs/2502.15376)
- **What's New**: 이 연구는 최신의 gauge-equivariant 네트워크를 토폴로지적 집합 물리학 분야에 적용하여 하이브리드 Chern 수를 예측하는 새로운 방법을 제시합니다. 기존의 네트워크는 주로 양자 색역학에서 사용되었으나, 본 연구는 이들 네트워크가 어떻게 더 복잡한 물리적 시스템에서도 사용될 수 있는지를 보여줍니다. 특히, 최소 7개의 채워진 밴드가 있는 다중 밴드 토폴로지 절연체에 대한 Chern 수를 예측할 수 있는 모델을 만드는 것이 혁신적입니다.

- **Technical Details**: 이 모델은 다중 밴드 토폴로지 절연체의 Chern 수를 예측하기 위해 gauge symmetry와 gauge-equivariant 신경망의 개념을 사용합니다. 본 연구는 새로운 gauge-equivariant 정규화 레이어를 도입하여 학습을 안정화하고, 우리의 설정에 대한 보편 근사 정리를 증명합니다. 입력 공간 X는 재료의 Brillouin zone에서의 푸리에 변환으로 구성되며, 시스템은 N개 채워진 밴드를 갖는 U(N) 대칭 그룹을 따릅니다.

- **Performance Highlights**: 실험 결과, 이 모델은 기본적인 Chern 수에서 비기본적인 Chern 수까지 잘 일반화되는 성과를 보입니다. 본 연구에서 제안하는 네트워크 구조는 여러 격자 크기와 시스템에서 Chern 수를 성공적으로 예측할 수 있는 것을 나타냅니다. 각종 ablation 연구를 통해 다양한 gauge-equivariant 아키텍처에 대한 효과를 확인했습니다.



### Efficient and Provable Algorithms for Covariate Shif (https://arxiv.org/abs/2502.15372)
- **What's New**: 이 논문은 covariate shift(공변량 이동)에 대한 이론적 보장이 부족하다는 문제를 해결하기 위해, training(훈련) 샘플과 test(테스트) 샘플에서 정보 추출을 하는 효율적인 알고리즘을 제시합니다. 저자들은 어떤 알려지지 않은 함수의 평균을 추정하는 문제에 집중하며, 이는 많은 학습 문제에서 핵심적인 하위 루틴으로 작용합니다. 이러한 문제의 본질을 정제하고, 이론적 기반을 마련하는 데 중요한 기초를 제공합니다.

- **Technical Details**: 이 연구에서는 $	ilde{oldsymbol{x}}$에 대해 평균 $	ext{E}_{	ilde{oldsymbol{x}} 	ext{~sim~} p_{	ext{test}}} oldsymbol{f}(	ilde{oldsymbol{x}})$을 추정하기 위해 labeled(라벨이 있는) 훈련 샘플 $(oldsymbol{x}_i, oldsymbol{f}(oldsymbol{x}_i))$와 unlabeled(라벨이 없는) 테스트 샘플 $	ilde{oldsymbol{x}}_i$를 사용합니다. 소개하는 알고리즘들은 샘플 복잡도와 계산 보장에 대한 증명된 이점을 가지고 있습니다. 이론적 보장이 부족했던 covariate shift 문제에 대한 엄밀한 분석을 최초로 제공하여, 관련 분야에서의 발전을 도모합니다.

- **Performance Highlights**: 제시된 알고리즘은 확고한 계산 보장과 함께 provable sample complexity를 자랑하며, 다양한 실전 학습 문제에 직접 적용할 수 있습니다. 연구 결과는 covariate shift 문제를 해결하기 위한 새로운 이론적 시각을 제공하며, 이는 나중에 발전할 수 있는 가능성을 보입니다. 이 연구는 covariate shift에 대한 이해도를 높이고, 관련 알고리즘의 성능을 개선할 수 있는 기틀을 마련합니다.



### Efficiently Solving Discounted MDPs with Predictions on Transition Matrices (https://arxiv.org/abs/2502.15345)
- **What's New**: 이 논문에서는 Generative Model 하의 무한 수명 할인 마르코프 결정 프로세스(DMDPs)에 대한 새로운 프레임워크를 제안합니다. Mitzenmacher와 Vassilvitskii의 2022년 연구에 영감을 받아, 전이 행렬(transition matrix)에 대한 예측이 DMDP를 해결하는 샘플 효율성을 어떻게 향상시킬 수 있는지를 탐구합니다. 또한, 예측 정확도에 대한 사전 지식이 없는 경우의 불가능성을 보이고, 새로운 알고리즘을 통해 예측을 활용하여 샘플 복잡도(sample complexity)를 개선합니다.

- **Technical Details**: DMDPs는 상태 집합(𝒮), 행동 집합(𝒜), 전이 행렬(𝑃), 보상 벡터(𝑟), 그리고 할인 계수(𝛾)를 포함하는 튜플로 정의됩니다. 샘플 복잡도는 주어진 $ar{	ext{O}}((1-	ext{γ})^{-3} N 	ext{ε}^{-2})$의 맥락에서, 예측된 전이 행렬을 활용한 새로운 알고리즘이 어떻게 샘플 복잡도를 $ar{	ext{O}}((1-	ext{γ})^{-4} N 	ext{ε}^{-2})$로 개선할 수 있는지를 다룹니다. 추가적으로, 이를 뒷받침하는 수치 실험 결과도 함께 제공됩니다.

- **Performance Highlights**: 이 연구는 예측 정확도에 따라 개선된 샘플 복잡도를 달성하는 알고리즘을 제안함으로써, 기존의 최상의 결과를 능가하는 성과를 보여줍니다. 새로운 알고리즘은 전이 행렬의 예측을 활용하여, 보다 나은 샘플 복잡도를 확보합니다. 이론적인 결과는 수치 실험을 통해 검증되어, 제안한 알고리즘이 실제 적용 가능성을 갖추고 있음을 입증합니다.



### Learning with Limited Shared Information in Multi-agent Multi-armed Band (https://arxiv.org/abs/2502.15338)
- **What's New**: 기존의 MAMAB(다중 에이전트 다중 무장 강도) 연구들은 참가자가 개인정보를 공유하지 않기로 선택할 수 있는 상황을 고려하지 않았습니다. 본 논문에서는 에이전트가 공유할 정보의 범위를 제한할 수 있는 새로운 모델인 LSI-MAMAB을 제안합니다. 이를 통해 각 에이전트는 단순히 수용할 수 있는 정보만 공유하고, 다른 사람의 학습에 기여할 수 있는 구조를 형성합니다.

- **Technical Details**: LSI-MAMAB 모델에서는 각 에이전트가 자신이 공유하고자 하는 정보만 공유하게 됩니다. 이로 인해, 데이터 불균형 문제와 개인의 합리성을 보장하기 위한 메커니즘을 설계하는 두 가지 도전 과제가 발생합니다. Balanced-ETC 알고리즘은 이러한 두 가지 도전을 해결하며, 각 에이전트의 총 후회(regret)를 O(N log T)로 제한하여 데이터가 공유되는 아쉬운 경우에 비해 최소한으로 효율적인 협업을 보장합니다.

- **Performance Highlights**: Balanced-ETC 알고리즘은 비대칭적으로 적은 정보를 공유하는 상황에서도 개인의 후회를 단일 에이전트 설정에서의 UCB 알고리즘의 후회로 제한하여 개인의 합리성(IR)을 충족합니다. 실험 결과는 이론적 결과를 검증하며, 에이전트들이 협업 학습에 참여할 동기를 부여하는 인센티브 메커니즘도 제안됩니다.



### Tight Clusters Make Specialized Experts (https://arxiv.org/abs/2502.15315)
- **What's New**: 이 논문에서는 Adaptive Clustering (AC) 라우터를 소개하며, 이 라우터는 각 입력의 클러스터 할당을 최적화하여 적절한 전문가와의 토큰 매칭을 개선합니다. 기존의 Sparse Mixture-of-Experts (MoE) 구조에서 나타나는 느린 수렴과 데이터 부패에 대한 저항성을 해결하는 것을 목표로 합니다. AC 라우터는 데이터 내의 잠재적 클러스터를 최대한 식별하여 각 토큰에 가장 적합한 전문가를 더 쉽게 발견하도록 최적화된 공간에서 토큰-전문가 할당을 계산합니다.

- **Technical Details**: 본 연구는 MoE 아키텍처에서 입력에 적합한 특징을 학습하고 이들 특징을 따라서 토큰-전문가 할당을 변환된 공간에서 수행하는 방법을 제시합니다. 이는 전문가들이 의미적으로 유사한 입력에 더 빨리 전문화되도록 해주며, 클러스터 간의 분리를 개선하여 데이터 부패에 대한 저항성을 높입니다. 또한, 이 방법은 저렴한 계산 오버헤드로 고성능을 발휘하며, 학습 가능한 파라미터가 필요하지 않습니다.

- **Performance Highlights**: AC 라우터는 WikiText-103 언어 모델링 및 ImageNet-1k 객체 분류와 같은 대규모 작업에서 기준 라우팅 방법을 초월하는 성능 향상을 입증했습니다. 특히 AC 라우터는 기반 방법들보다 더 빠른 수렴을 보였으며, 실험적으로도 전체적인 성능 개선이 확인되었습니다. 이는 클러스터 분리가 잘 이루어져서 전문가들이 특정 입력의 특성에 맞춤형으로 배정되는 결과를 가져옵니다.



### SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention (https://arxiv.org/abs/2502.15304)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 KV 캐시 성능을 극대화하기 위해 SVDq라는 새로운 혼합 정밀도 양자화 기법을 제안합니다. 이 방법은 SVD(특이값 분해)를 이용하여 잠재 채널로 KV 캐시를 변환한 후, 중요도 기반 양자화 및 압축을 적용합니다. 또한, SVDq의 사용으로 양자화 오류가 기존 방법에 비해 상당히 낮음을 이론적으로 입증했습니다.

- **Technical Details**: SVDq 방법은 SVD 기반 채널 압축을 통합하여 KV 캐시의 양자화 정밀도를 최적화합니다. 이 방법은 SVD를 통해 얻은 특이값과 관련된 잠재 채널에 대해 더 높은 비트폭을 할당하며, 작은 특이값과 관련된 채널의 정밀도는 점차 감소시킵니다. 이를 통해 압축 비율을 높이면서도 비교 가능한 모델 성능을 유지할 수 있습니다.

- **Performance Highlights**: SVDq는 LongBench 및 RULER 벤치마크를 기준으로 하여, 키 캐시 정밀도를 1.25비트로 낮추면서 최대 410배의 압축 비율을 달성할 수 있음을 보여줍니다. 또한, 이 방법은 LongBench 데이터셋에서 거의 무손실에 가까운 결과를 기록하였으며, 전반적으로 LLMs의 높은 정밀도 저비트 양자화를 통해 KV 캐시 압축의 효율성을 높입니다.



### Beyond Fixed Variables: Expanding-variate Time Series Forecasting via Flat Scheme and Spatio-temporal Focal Learning (https://arxiv.org/abs/2502.15296)
- **What's New**: 최근 Multivariate Time Series Forecasting (MTSF) 분야에서 새로운 도전과제를 제시하는 Expanding-variate Time Series Forecasting (EVTSF)라는 과제가 소개되었습니다. 실제 어플리케이션에서는 새로운 센서가 추가됨에 따라 변수가 증가하지만, 기존 연구는 고정된 변수를 전제로 했습니다. 이 논문에서는 데이터 구조의 불일치와 불균형한 시공간 학습이라는 두 가지 주요 문제와 이를 해결하기 위한 새로운 접근 방식을 다룹니다.

- **Technical Details**: 새로 제안된 STEV는 유연한 시공간 예측 프레임워크로, Flat Scheme을 통해 변수 차원을 따라 2D 샘플을 평탄화하여 1D 공간으로 확장합니다. 이러한 구조는 변수가 증가하더라도 모델의 동적 공간적 상관관계를 유지합니다. 또한, 시공간 Focal Learning 전략을 도입하여 대조 학습과 그래프 표현 간의 갈등을 해결하는 부정 필터를 포함하고 있습니다.

- **Performance Highlights**: 세 개의 실제 데이터셋을 사용한 벤치마킹 결과, STEV는 확장된 변수를 포함할 때 기존 SOTA MTSF 모델보다 우수한 성능을 보였습니다. 특히, 관측치가 5%에 불과한 경우에도 STEV가 완전 관측으로 훈련된 모델과 동등한 성능을 발휘하였습니다. 다양한 확장 전략을 탐색함으로써 실제 응용에 대한 STEV의 일반화 가능성이 강조되었습니다.



### Hyperspherical Normalization for Scalable Deep Reinforcement Learning (https://arxiv.org/abs/2502.15280)
Comments:
          50 pages. Preprint

- **What's New**: SimbaV2는 비정상적인 최적화를 안정화하기 위해 설계된 새로운 RL 아키텍처입니다. 모델의 무게(nom)와 특성( feature norm) 규제를 위해 하이퍼스피어 정규화를 사용하고, 리워드 스케일링을 통한 분포적 가치 추정으로 안정적인 그래디언트를 유지합니다. 이 접근법은 기존의 RL 커뮤니티에서 연구된 정규화 방법들의 한계를 극복하기 위해 제시되었습니다.

- **Technical Details**: SimbaV2는 Soft Actor-Critic(SAC)을 기반으로 하여 모델의 크기와 계산 능력을 효과적으로 확장할 수 있습니다. 뼈대 아키텍처는 사전 레이어 정규화를 가진 잔여 블록(pre-layernorm residual blocks)을 포함하며, 하이퍼스피어 정규화로 모든 레이어 정규화를 대체하여 일관된 효과적 학습률을 보장합니다. 또한 분포적 비평가(distributional critic)를 통합하고, 리워드 규모 변동을 위해 리워드를 스케일링하여 그래디언트 규범의 불안정성을 해결합니다.

- **Performance Highlights**: SimbaV2는 4개 도메인에서 57개의 연속 제어 작업을 수행하며, 최첨단 성능을 달성합니다. 알고리즘 수정이나 하이퍼파라미터 조정 없이도 안정적인 성능을 유지하며, 모델 크기와 계산량 증가에 비례해 효과적으로 확장 가능합니다. 기존의 주기적 재초기화 없이도 과적합을 방지하면서 훈련을 진행할 수 있는 특징이 있습니다.



### Towards a Reward-Free Reinforcement Learning Framework for Vehicle Contro (https://arxiv.org/abs/2502.15262)
- **What's New**: 본 논문에서는 차량 제어를 위한 새로운 보상 없는 강화학습 프레임워크(RFRLF)를 제안합니다. 이 프레임워크는 수동으로 설계된 보상 신호에 의존하지 않고, 목표 상태 예측 네트워크(TSPN)와 보상 없는 상태 안내 정책 네트워크(RFSGPN)를 통해 에이전트의 행동을 최적화합니다. 이를 통해 인간의 편향을 최소화하고, 보상의 적용이 불가능한 환경에서도 효율적으로 학습할 수 있습니다.

- **Technical Details**: RFRLF는 두 가지 주요 구성 요소로 나뉘며, 첫 번째는 환경의 상태를 예측하는 TSPN으로, 이는 에이전트의 현재 상태와 수행하는 동작에 따라 다음 상태를 예측합니다. 두 번째는 RFSGPN으로, 이 네트워크는 수동으로 설계된 보상 신호에 의존하지 않고 최적의 행동 결정을 할 수 있도록 설계되었습니다. 이 두 네트워크의 결합은 에이전트가 보상 피드백이 없는 상태에서도 정책을 효과적으로 최적화할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, RFRLF는 차량 운전 제어에서 높은 학습 효율성과 적응성을 보여 주었습니다. 본 연구에서 제안된 방법은 보상이 전혀 없거나 불확실한 환경에서도 우수한 성능을 발휘하며, 전문가의 행동 정보를 전달받지 못하는 상황에서도 효과적으로 정책을 학습할 수 있는 가능성을 보여줍니다.



### Real-Time Moving Flock Detection in Pedestrian Trajectories Using Sequential Deep Learning Models (https://arxiv.org/abs/2502.15252)
- **What's New**: 이 논문은 다중 보행자 궤적에서 실시간 조류 탐지를 위해 순차적인 딥러닝 모델인 RNN, LSTM, Transformer의 사용을 조사합니다. 제안된 방법은 두 단계로 구성되며, 첫 번째 단계에서는 사전 훈련된 이진 분류 모델이 쌍 궤적을 분류하고, 두 번째 단계에서는 학습된 표현을 적용하여 다중 에이전트 조류를 동적으로 식별합니다. 실제 그룹 이동 데이터셋으로 검증되어 다양한 이동 패턴에서도 높은 정확도와 안정성을 보여줍니다.

- **Technical Details**: 이 연구에서는 RNN, LSTM, Transformer와 같은 순차적인 딥러닝 모델을 활용하여 궤적 쌍이 조류를 형성하는지 여부를 분류합니다. 모델의 훈련은 보행자의 위치, 속도, 운동 각도 및 상호 간 거리와 같은 기본 이동 특징들을 기반으로 이뤄집니다. 이진 분류 프레임워크를 통해 보행자의 쌍 간 상호작용을 효과적으로 감지하여 동적인 보행자 환경에서도 수많은 조합을 처리할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 동적이고 혼잡한 환경에서도 보행자 조류를 높은 정확도로 지속적으로 탐지할 수 있음을 보여줍니다. 필터링 및 집계 단계를 통해 비즈니스 적인 통찰력을 제공할 수 있는 더 큰 조류 형성 추정이 가능해집니다. 이러한 접근법은 군집, 호송대, 그리고 벌떼와 같은 다양한 집단 이동 형태의 탐지로 확장되며, 다중 에이전트 행동 분석을 위한 보다 포괄적인 기반을 제공합니다.



### Multi-agent Multi-armed Bandits with Minimum Reward Guarantee Fairness (https://arxiv.org/abs/2502.15240)
- **What's New**: 본 논문에서는 다수의 에이전트와 다수의 암(arm)을 갖는 풀기(MA-MAB) 설정에서 사회적 복지(social welfare)를 극대화하면서 공정성(fairness)을 보장하는 문제를 다룹니다. 제안된 RewardFairUCB 알고리즘은 Upper Confidence Bound (UCB) 기술을 활용하여 공정성과 사회적 복지에 대한 비선형 후회(regret) 경계를 달성합니다. 각 에이전트는 최대 가능한 보상의 일정 비율 이상을 받는 것을 보장하도록 설계되었습니다.

- **Technical Details**: RewardFairUCB 알고리즘은 각각의 에이전트에 대한 최소 보장을 제공하는 명시적 제약 조건을 설정합니다. 공정성 후회(fairness regret)와 사회적 복지 후회(social welfare regret)를 측정하며, 비관적인 결과에 따라 사회적 복지 후회의 상한선은 O(T^(1/2)), 공정성 후회의 상한선은 O(T^(3/4))로 나타났습니다. 이러한 보장들은 보편적인 인스턴스 독립적 특성을 갖고 있어, 다양한 설정에 적용 가능합니다.

- **Performance Highlights**: RewardFairUCB 알고리즘의 성능은 다양한 기준선 및 휴리스틱 알고리즘과 비교되었습니다. 시뮬레이션 데이터와 실제 데이터를 활용하여 공정성과 사회적 복지 간의 균형을 평가하였고, 실험 결과 이러한 트레이드오프가 명확히 드러났습니다. 이는 알고리즘이 단순히 보상을 극대화하는 것 이상의 공정성을 고려함을 보여주었습니다.



### Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs (https://arxiv.org/abs/2502.15224)
Comments:
          13 pages

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 인간과 유사한 방식으로 과학적 연구를 수행하고 새로운 지식을 발견할 수 있는지를 탐구합니다. 이를 위해 연구자들은 LLM의 과학적 발견 능력을 평가하기 위한 새로운 벤치마크인 	extit{Auto-Bench}를 소개합니다. 이 벤치마크는 인과 그래프 발견 원칙을 기반으로 하여, 모델이 숨겨진 구조를 발견하고 최적의 결정을 내릴 수 있도록 도전합니다.

- **Technical Details**: 연구 방법론에서 LLM 모델은 화학 및 사회 네트워크를 포함한 두 가지 벤치마크를 통해 그 성능을 평가받습니다. 이들 벤치마크는 노드를 포함한 그래프로 구성되며, 각 노드에서의 개입이 연결된 노드에 미치는 영향을 분석합니다. LLM은 초기 가설을 바탕으로 반복적으로 실험을 수행하며, 새로운 관찰 데이터를 통해 가설을 수정합니다.

- **Performance Highlights**: 실험 결과, 현재의 LLM 모델들이 문제의 복잡성 증가에 따라 성능 저하를 겪는다는 것을 발견했습니다. 예를 들어, 화학 및 사회 네트워크에서 노드 수가 증가하면 평균적으로 정답을 얻는 데 필요한 사이클 수가 급격히 감소합니다. 이 연구는 기계와 인간 지능 간의 정보 처리에서의 중요한 차이를 강조하며, 향후 LLM 발전에 있어 이를 해결해야 할 필요성을 제기합니다.



### The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning (https://arxiv.org/abs/2502.15214)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 Reinforcement Learning (RL)과 Large Language Models (LLMs), Vision-Language Models (VLMs)의 통합을 탐구하는 설문 조사로, RL에서 자주 발생하는 도전 과제를 극복하기 위한 다양한 접근 방식을 리뷰합니다. RL에서는 인간이 설계한 보상, 샘플 비효율성, 일반화 부족, 해석 가능성 제한 등의 문제를 가지고 있으며, LLMs와 VLMs의 통합이 이를 해결할 기회를 제공합니다. 특히, 이 논문은 LLM/VLM을 에이전트, 플래너, 보상 역할로 체계적으로 분류한 새로운 분류 체계를 제시합니다.

- **Technical Details**: MDP (Markov Decision Process)는 상태 집합(S), 행동 집합(A), 전이 확률 함수(T), 보상 함수(R), 할인 계수(γ)로 정의됩니다. RL 에이전트는 환경과의 상호작용을 통해 정책(π)을 학습하고, 이는 상태를 행동으로 매핑합니다. 논문은 또한 LLM과 VLM이 RL에서 수반하는 역할에 대해 강조하며, 이는 효과적인 데이터 효율성, 일반화 및 해석 가능성을 향상시킬 수 있습니다.

- **Performance Highlights**: LLMs는 자연어 처리에서 혁신적인 발전을 이룩하였으며, VLMs는 이와 결합하여 이미지와 텍스트 간의 의미를 이해하고 해석하는 능력을 가집니다. 이 연구는 LLM/VLM과 RL의 통합이 에이전트의 행동과 학습 방식을 혁신적으로 변화시킬 수 있음을 강조합니다. 향후 연구 방향으로는 기초 모델(Foundation Models)과 RL의 통합에 있어 한계를 극복하고, 공정성, 편향 완화 및 개선된 표현과 같은 상징적인 문제를 다루는 것이 포함됩니다.



### PairBench: A Systematic Framework for Selecting Reliable Judge VLMs (https://arxiv.org/abs/2502.15210)
- **What's New**: 본 논문에서는 PairBench라는 새로운 프레임워크를 소개합니다. 이는 대규모 비전 언어 모델(VLMs)을 자동 평가자로 평가하기 위한 저비용 시스템으로, 다양한 모달리티(modality)와 시나리오에서 사용될 수 있습니다. PairBench는 유사성 점수의 핵심 요구 사항을 나타내는 네 가지 메트릭(metrics)을 도입하여 VLMs의 성능을 평가합니다.

- **Technical Details**: PairBench는 인간 주석(human annotations)과의 일치성, 데이터 쌍의 순서에 관계없는 일관성, 유사성 분포의 부드러움(smoothness), 프롬프트(prompting)를 통한 제어 가능성 등을 측정합니다. 분석 결과, 모든 메트릭에서 우수한 성능을 보이는 모델은 없었으며, 특정 평가자의 원하는 행동에 따라 최적의 선택이 달라질 수 있음을 보여주었습니다.

- **Performance Highlights**: 많은 VLM들이 순서와 상관없이 대칭 유사성 점수를 유지하는 데 어려움을 겪는 것으로 나타났습니다. PairBench에서의 성능은 기존의 인기 있는 벤치마크와 밀접하게 관련되어 있으며, 이를 통해 모델 순위에서 예측력을 보여줍니다. 이러한 결과는 VLMs를 평가자로 널리 사용하기 전에 철저한 평가가 필요함을 강조합니다.



### Graph-Based Deep Learning on Stereo EEG for Predicting Seizure Freedom in Epilepsy Patients (https://arxiv.org/abs/2502.15198)
- **What's New**: 이번 연구에서는 약물 저항성 epilepsy 환자의 stereo electroencephalography (sEEG) 데이터를 이용하여 발작 자유 예측을 위한 깊은 학습 기반의 그래프 신경망 (GNN) 모델을 개발하였습니다. 기존의 전통적인 방법으로는 환자 집단의 다양성 때문에 정확한 예측이 어려웠던 점이 해결되었습니다. 새로운 모델이 다양한 환자의 뇌 연결성을 이해하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 이 모델은 그래프 컨볼루션 (graph convolutions)과 다중 스케일 주의 메커니즘 (multi-scale attention mechanisms)을 통합하여, 연구하기 어려운 부위들(예: thalamus와 운동 영역) 간의 연결을 캡처합니다. 고품질의 sEEG 데이터는 15명의 소아 환자에게서 수집되어 학습에 사용되었습니다. GNN을 적용함으로써 발작 자유 결과 예측의 정밀도를 높일 수 있었습니다.

- **Performance Highlights**: 모델은 이진 분류 분석에서 92.4%, 환자별 분석에서 86.6%, 다중 분류 분석에서 81.4%의 정확도를 달성했습니다. 주요 기여 지역으로는 anterior cingulate와 frontal pole 영역이 확인되었으며, 이들 노드는 발작 시작 구역과 일치하는 경향이 있었습니다. 본 연구 결과는 GNN과 같은 신규 연결 기반의 깊은 학습 모델이 발작 자유 예측 및 개인 맞춤형 epilepsy 치료 계획에 기여할 수 있는 가능성을 강조합니다.



### Optimizing Product Provenance Verification using Data Valuation Methods (https://arxiv.org/abs/2502.15177)
- **What's New**: 이 연구에서는 Stable Isotope Ratio Analysis (SIRA)와 Gaussian 프로세스 회귀 기반의 isoscapes를 조합하여 제품의 출처를 직관적으로 검증하는 새로운 데이터 가치 평가 프레임워크를 소개합니다. 이 방법은 훈련 데이터를 보다 효과적으로 선택하고 활용하도록 설계되었으며, 주로 정보량이 높은 샘플을 우선시하여 다양한 데이터셋에서도 모델의 견고성과 예측 정확성을 향상시킵니다. 본 연구는 유럽의 법 집행 기관과 협력하여 불법 로그 및 제재된 목재 제품을 식별하고, 신뢰성 있는 샘플을 최적화하는 데 기여하고 있음을 보여줍니다.

- **Technical Details**: 본 연구에서 SIRA는 지리적 출처 확인을 위한 유용한 도구로, 환경, 대기 및 흙 등의 다양한 요인이 안정 동위 원소 비율에 미치는 영향을 분석합니다. 데이터 수집 위치와 이들 위치에서 측정된 안정 동위 원소 비율 값을 기반으로 하여, 훈련 데이터셋을 구성하고 이를 평가하기 위해 기계 학습 모델을 적용합니다. 이제까지 사용된 방법론에서의 데이터 희소성 및 부적절한 데이터셋 선택의 문제를 해결하기 위해 데이터의 가치를 평가하는 새로운 접근법을 도입하였으며, 데이터 샘플 수집의 효율성을 증대시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: 본 연구에서는 제안된 방법론을 통해 제품 출처 검증의 정확성을 크게 향상시킬 수 있음을 실험적으로 입증했습니다. 다양한 데이터셋에 대해 모델의 성능을 평가한 결과, 정보 가치가 높은 샘플을 우선적으로 고려함으로써 예측 정확성이 향상되었습니다. 이는 불법적인 거래 관행을 감소시키고, 글로벌 공급망의 규제 강화를 목표로 하는 데 있어 중요한 기여를 할 수 있습니다.



### Projection Optimization: A General Framework for Multi-Objective and Multi-Group RLHF (https://arxiv.org/abs/2502.15145)
- **What's New**: 이번 연구에서는 Multi-Objective Reinforcement Learning with Human Feedback (MORLHF) 접근법을 통해 각 목표에 대한 선호 피드백을 수집하고, 이를 통해 최적의 의사결정을 이끌어내는 방법을 제시합니다. 기존 연구와는 달리 선형 집계(linear aggregation) 방식을 벗어나 비선형 집계를 다루는 기존의 접근법의 한계를 극복하였습니다. 또한, 여러 그룹의 다양한 목표 가중치를 처리할 수 있는 확장된 프레임워크를 개발했습니다.

- **Technical Details**: 비선형 집계 문제를 해결하기 위해, 본 연구에서는 문제를 일련의 서브 문제(sub-problems)로 변환하였습니다. 각 서브 문제는 오직 선형 집계만을 포함하므로 계산적으로 효율적으로 해결할 수 있습니다. 이 프레임워크는 모든 그룹의 목표를 최대화하는 방법을 제공하여, 목표 간의 합의를 이끌어낼 수 있도록 돕습니다.

- **Performance Highlights**: 이론적으로, 제안된 알고리즘 프레임워크는 서브선형 유감(sublinear regret)을 달성할 수 있음을 입증하였습니다. 경험적으로, 개별 목표에 대한 최적 정책들이 얻어진 후, 거의 훈련이 필요 없는 알고리즘을 제시하여 효율성을 극대화할 수 있음을 보여줍니다.



### Data Complexity Measures for Quantum Circuits Architecture Recommendation (https://arxiv.org/abs/2502.15129)
- **What's New**: 이 논문에서는 데이터베이스 복잡성 측정을 활용하여 양자 회로 추천 아키텍처를 제안합니다. 이는 분류 문제 해결을 위한 최적의 양자 회로를 찾는 방법을 제공하고, 6개의 양자 회로를 14개의 데이터베이스에서 평가하였습니다. 본 연구는 기존의 연구에서는 다룬 적이 없는 데이터베이스 정보를 활용한 양자 회로 선택의 새로운 접근 방식을 보여줍니다.

- **Technical Details**: 파라미터화된 양자 회로(Parameterized Quantum Circuits, PQCs)는 분류, 예측 및 근사값 계산을 위해 고안된 구조로, 이를 통해 효율적인 양자 회로 디자인이 가능해집니다. 데이터베이스에서 추출한 복잡성 측정치는 각 데이터셋에 적합한 회로 레이아웃과 반복 횟수를 선택하는 데 유용하게 이용됩니다. 특히, 논문에서는 22개의 데이터베이스 복잡성 지표를 제시하고 이를 통해 회로를 최적화하는 방법을 설명하고 있습니다.

- **Performance Highlights**: 연구 결과, 최적의 양자 회로는 100% 정확도로 모든 분류 문제를 해결할 수 있음을 보였습니다. 평균 절대 오차는 0.80 ± 2.17로, 최대 3개의 레이어 추가 시 오차 범위 내에서 적정한 레이어 반복 횟수를 결정하는 데 도움이 되었습니다. 16개의 머신 러닝 모델과 12개의 고전 회귀 모델을 사용하여 양자 회로 선택이 각각의 데이터셋에 대해 수행되었습니다.



### MONSTER: Monash Scalable Time Series Evaluation Repository (https://arxiv.org/abs/2502.15122)
Comments:
          45 pages; 38 figures

- **What's New**: MONSTER(모너스터)라는 새로운 데이터셋을 소개합니다. 이는 MONash Scalable Time Series Evaluation Repository의 약자로, 시계열(classification) 분류를 위한 대규모 데이터셋의 모음입니다. 기존의 UCR과 UEA 데이터셋은 상대적으로 작은 데이터셋에 의존해 왔기 때문에, 대규모 데이터셋을 활용한 새로운 벤치마크를 도입하여 시계열 분류 분야를 다양화하고자 합니다.

- **Technical Details**: 이 논문에서는 우리 연구의 배경을 설명하고, MONSTER 데이터셋에 대한 세부사항을 제공하는 동시에, 현재 시계열 분류에서 사용되는 벤치마크의 한계를 지적합니다. 현재 벤치마크는 일반적으로 작고 다양한 데이터셋에 최적화된 저 분산(variance) 모델을 선호하기 때문에 대규모 데이터에서의 실제 학습 문제를 잘 반영하지 못하고 있습니다. 이로 인해 시계열 분류의 최신 기법들은 주로 작은 데이터셋에 해당하는 전략으로 한정될 위험이 존재합니다.

- **Performance Highlights**: MONSTER 데이터셋을 사용하면서, 벤치마크의 효과성이 향상되고 실제 문제를 더 잘 반영할 수 있을 것으로 기대합니다. 이 데이터셋은 시계열 분류의 새로운 발전 가능성을 제시하며, 다양한 머신러닝 모델이 대규모 데이터를 처리하는 도전 과제를 잘 이겨낼 수 있도록 합니다. 본 연구는 시계열 데이터에서 효과적으로 학습하는 방법론의 발전을 촉진할 것으로 보입니다.



### Assessing a Single Student's Concentration on Learning Platforms: A Machine Learning-Enhanced EEG-Based Framework (https://arxiv.org/abs/2502.15107)
- **What's New**: 이 연구는 온라인 학습 세션 중 학생의 집중 상태를 분류하기 위해 맞춤 설계된 기계 학습 모델을 사용하는 특화된 파이프라인을 소개합니다. EEG 데이터의 수집 및 전처리 프로토콜과 함께 5개의 EEG 신호 밴드에서 50개의 통계적 특성을 추출하는 방법이 상세히 설명되어 있습니다. 하이퍼파라미터 튜닝을 통해 학생의 집중 상태 분류 정확도를 향상시키는 이점을 탐구하며, VR 환경에서도 높은 정확성을 달성했습니다.

- **Technical Details**: 이 연구에서는 Muse S(Gen 2)라는 경량의 상용 EEG 센서를 사용하여 EEG 신호를 수집하였습니다. 신호는 Delta, Theta, Alpha, Beta, Gamma의 5개 주파수 밴드로 나뉘어 각 특성에서 평균, 분산, 표준편차 등의 통계적 특징을 추출하였습니다. Random Forest 모델을 이용하여 VR 및 비 VR 환경에서의 학생의 집중 상태 예측을 수행하였으며, 모델 성능 개선을 위한 피쳐 선택 및 하이퍼파라미터 최적화를 진행했습니다.

- **Performance Highlights**: 실험 결과, 컴퓨터 기반 학습 설정에서는 97.6%, 가상 현실 설정에서는 98%의 테스트 정확도를 기록하였습니다. 이는 온라인 교육 활동 중 학생의 집중에 대한 개인화된 통찰력을 제공하는 접근법의 효과를 강조합니다. 이러한 높은 정확도는 EEG 데이터와 기계 학습 기술의 결합이 어떻게 인지 상태 분류에 기여할 수 있는지를 보여줍니다.



### Leveraging ChatGPT for Sponsored Ad Detection and Keyword Extraction in YouTube Videos (https://arxiv.org/abs/2502.15102)
Comments:
          6 pages, 4 figures, accepted and presented in the 10th IEEE International Conference on Sustainable Technology and Engineering

- **What's New**: 이 연구는 YouTube 비디오에서 스폰서 광고 세그먼트를 탐지하기 위한 새로운 접근 방식을 제시합니다. 421개의 자동 생성 및 수동 텍스트를 수집한 후, 이를 Prompt-engineered GPT-4o에 입력하여 광고 탐지 및 키워드 추출을 진행했습니다. 새로운 방식으로 교육 콘텐츠에 내장된 광고 유형과 그 관련성에 대한 통찰력을 제공합니다.

- **Technical Details**: 이 방법론에서는 자동 생성된 421개의 트랜스크립트(transcript)를 수집하고 이를 GPT-4o에 입력하여 광고(ads) 탐지를 진행합니다. 이후, KeyBERT를 통해 키워드를 추출하고, ChatGPT의 또 다른 iteration을 통해 카테고리 식별을 수행합니다. 최종적으로 9개의 콘텐츠 카테고리와 4개의 광고 카테고리로 정제된 결과를 얻었습니다.

- **Performance Highlights**: 이 연구의 결과는 다양한 교육 주제에서 상품 관련 광고가 상당히 많이 나타나는 것을 보여줍니다. 기존의 광고 탐지 방법과 비교해 스케일 가능하고 효율적인 대안을 제공하며, LLMs가 광고 탐지 프로세스를 변화시키는 잠재력을 강조합니다. 이는 디지털 미디어 내 광고 전략을 이해하는 데 새로운 통찰력을 제공합니다.



### UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning (https://arxiv.org/abs/2502.15082)
Comments:
          Code: this https URL

- **What's New**: 본 연구에서는 사전 학습된 대형 언어 모델(LLM)에서 특정 정보를 삭제하는 과정 동안 모델의 다른 능력을 보존할 수 있는 효율적인 기술을 개발하고자 합니다. 이를 위해 데이터 선택을 통해 모델의 성능 저하를 최소화하는 UPCORE (Utility-Preserving Coreset Selection)라는 방법론을 제안합니다. UPCORE는 삭제 성과(deletion efficacy)와 모델 보존(model preservation) 간의 균형을 최적화하는 데 중점을 두고 있습니다.

- **Technical Details**: UPCORE는 비슷한 정보를 포함한 데이터 포인트를 식별하고 정리하여 Forget Set 내에서의 분산(variance)을 줄인다는 혁신적 접근법을 적용합니다. 이를 위해, 고립 숲(Isolation Forest) 알고리즘을 사용하여 이상 점을 찾아내고 이를 제거함으로써 전체 Forget Set의 분산을 줄입니다. 이러한 과정은 모델의 기능 저하를 최소화하면서 효율적으로 데이터 삭제를 수행할 수 있도록 합니다.

- **Performance Highlights**: UPCORE는 세 가지 표준 머신 언러닝 방법에 대해 평가되었으며, 항상 가장 높은 AUC (Area Under Curve)를 달성하여 성능이 우수함을 입증하였습니다. 또한, UPCORE는 비정상적인 데이터 포인트에 대한 긍정적인 전이 긍정적 효과를 극대화하여 삭제가 필요한 정보를 효과적으로 제거함과 동시에, 모델의 다른 능력에 대한 저하를 최소화했습니다. 결과적으로 UPCORE는 다양한 우선순위 간의 거래에서 최상의 결과를 보이며, 랜덤 선택 대비 더 나은 지표를 보이고 있습니다.



### More for Keys, Less for Values: Adaptive KV Cache Quantization (https://arxiv.org/abs/2502.15075)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 핵심-값(KV) 캐시를 적응적으로 압축하는 정보 인식 기반의 양자화 프레임워크를 소개합니다. 이전 연구들은 추론 시 키와 값 캐시의 역할이 다르다는 점을 강조했지만, 본 연구는 정규 분포, 특이값 분포 등을 분석하여 키 행렬이 값 행렬보다 정규값이 높고 양자화에 더 민감하다는 것을 처음으로 밝혔습니다. 이러한 분석을 바탕으로, 우리는 키에 대해 더 높은 비트 너비를 할당하고 값에 대해서는 더 낮은 비트 너비를 할당하는 혼합 정밀도 양자화 전략인 KV-AdaQuant를 제안합니다.

- **Technical Details**: KV-AdaQuant는 특이값, 스펙트럼 노름 및 프로베니우스 노름과 같은 메트릭을 활용하여 키에 더 많은 비트를 할당하고 값에 대해서는 적은 비트를 할당합니다. 이 방식은 고전적인 양자화 오류 전파를 줄이고 메모리 절약을 달성하는 데 효과적입니다. 특히, 키와 값이 자신만의 노름 분포를 가지고 있음을 기반으로 하여, 그에 맞춰 비트 할당이 이루어집니다.

- **Performance Highlights**: 실험 결과, KV-AdaQuant는 많은 LLM에서 좋은 성능을 유지하면서도 상당한 메모리 절약을 가능하게 합니다. 예를 들어, 4비트를 키에, 2비트를 값에 사용하는 방식은 75.2%의 정확도를 달성한 반면, 반대로 2비트를 키에, 4비트를 값에 할당한 경우 정확도는 54.7%로 떨어졌습니다. 이는 혼합 정밀도 양자화 방식이 성능을 유지하면서도 메모리를 효과적으로 절약할 수 있음을 보여줍니다.



### Visualizing Machine Learning Models for Enhanced Financial Decision-Making and Risk Managemen (https://arxiv.org/abs/2502.15073)
- **What's New**: 이 연구는 특히 금융 산업에서 머신 러닝 모델을 시각화하는 것이 얼마나 중요한지를 강조합니다. 이는 해석 가능성을 개선하고 고위험 금융 환경에서의 예측을 지원하는 데 기여합니다. 시각적 도구는 알고리즘 의사 결정 프로세스에 대한 중요한 통찰력을 제공함으로써 성능 개선과 혁신적인 금융 모델 생성을 촉진합니다.

- **Technical Details**: 연구는 금융 머신 러닝 프레임워크 내에서 시각적으로 안내된 실험을 사용하여 리스크 평가(risk assessment)와 포트폴리오 할당(portfolio allocation)과 같은 중요한 개념을 더 이해하기 쉽게 만듭니다. 이 과정에서 거래 전략의 차이와 리스크 성향(risk appetite)과의 관계를 분석하며, 포트폴리오 재조정 빈도(frequency of portfolio rebalancing)와 리스크 수용도(risk tolerance) 간의 부정적인 상관관계를 발견합니다.

- **Performance Highlights**: 이 연구의 핵심 발견 중 하나는 관련 정보를 시각화를 통해 발견할 수 있다는 점입니다. 마지막으로 연구는 로컬 확률적 자산 가중(local stochastic asset weighing)의 새로운 방법을 제시하며, 이 과정에서 데이터 추출과 검증(validation)을 용이하게 하는 점을 강조합니다. 이 연구는 금융 머신 러닝 분야의 발전에 기여할 수 있는 유용한 방법들을 부각하고 있습니다.



### GiGL: Large-Scale Graph Neural Networks at Snapcha (https://arxiv.org/abs/2502.15054)
- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)의 최근 발전을 활용하여, 대규모 비즈니스 애플리케이션에 GNN을 적용하는 방법을 소개합니다. 특히, Snapchat에서 GNN 훈련, 추론, 활용을 위한 접근 방식을 공유하며, 특히 GiGL (Gigantic Graph Learning)이라는 오픈 소스 라이브러리를 통해 대규모 분산 그래프 기계 학습을 지원합니다.

- **Technical Details**: GiGL는 관계형 데이터베이스에서 그래프 데이터 전처리, 서브그래프 샘플링, 분산 훈련, 추론, 조정(Orchestration)과 같은 GNN 워크플로우의 복잡한 작업을 관리하는 데 사용됩니다. 이 라이브러리는 PyTorch Geometric (PyG)와 같은 오픈 소스 GNN 모델링 라이브러리와의 원활한 인터페이스를 제공하며, 내장된 생산화(Productionization) 문제를 처리하여 내부 전문가들이 모델링에 집중할 수 있도록 합니다.

- **Performance Highlights**: GiGL는 여러 생산 환경에서 사용되며, 지난 2년 동안 친구 추천(friend recommendation), 콘텐츠 추천(content recommendation), 광고(advertising) 등 다양한 비즈니스 도메인에서 35개 이상의 런칭을 지원했습니다. 이 논문에서는 GiGL의 고급 설계, 제공 도구, 스케일링 속성 및 다양한 비즈니스 환경에서의 사례 연구를 포함해 대규모 그래프 기계 학습을 통한 중요한 교훈도 공유합니다.



### Approximating Latent Manifolds in Neural Networks via Vanishing Ideals (https://arxiv.org/abs/2502.15051)
Comments:
          26 pages (8 main body, rest appendix and references), 12 figures, 3 tables, 3 algorithms

- **What's New**: 이 논문에서는 고차원 데이터가 저차원 다양체에 놓여 있다는 다양체 가설(manifold hypothesis)을 바탕으로 심층 신경망(deep neural networks)의 잠재적 다양체(latent manifold)를 특성화하는 방법을 제시합니다. 특히, 소실 이상(vanishing ideal) 알고리즘을 활용하여 이러한 잠재적 다양체를 설명할 수 있는 새로운 신경망 아키텍처인 VI-Nets를 제안합니다. VI-Nets는 사전 훈련된 신경망(pretrained network)의 중간 층에서 잘라낸 후, 다항식 생성기를 사용하여 각 클래스 다양체를 근사화하는 방식으로 동작합니다.

- **Technical Details**: VI-Nets는 고차원 잠재 공간을 다루기 위한 효율적인 방법론을 도입하여, 개별 클래스의 소실 이상 생성을 위한 다항식 집합을 구축합니다. 이 접근법은 잠재 공간의 비선형 분리(linearly inseparable) 데이터를 선형적으로 분리 가능하게 변환할 수 있습니다. 또한, 알고리즘의 적용을 용이하게 하기 위해 데이터 차원 축소 및 재조정 기법을 활용하여, 소수의 생성자(generator) 다항식을 효율적으로 평가할 수 있도록 합니다.

- **Performance Highlights**: VI-Nets는 사전 훈련된 신경망에 비해 훨씬 적은 파라미터를 사용하면서 비슷한 정도의 정확도를 유지하며, 높은 연산 처리량을 달성하는 성과를 보입니다. 논문에서 수행한 다양한 실험 결과, VI-Nets가 경쟁력 있는 성능을 발휘하면서도 적은 자원을 소모함을 확인했습니다. 이러한 접근법은 심층 학습(context of Deep Learning)에서 계산적 대수(computational algebra)의 도구를 활용한 새로운 길을 제시합니다.



### GeoAggregator: An Efficient Transformer Model for Geo-Spatial Tabular Data (https://arxiv.org/abs/2502.15032)
Comments:
          Accepted in the main technical track of the AAAI 2025

- **What's New**: 본 논문에서는 지리적 데이터 모델링을 위해 특별히 설계된 가벼운 알고리즘인 GeoAggregator를 소개합니다. 이 알고리즘은 transformer 아키텍처를 기반으로 하여 공간 자기 상관(spatial autocorrelation) 및 공간 이질성(spatial heterogeneity)을 명확히 고려합니다. 특히, Cartesian product를 사용하는 새로운 attention 메커니즘을 도입하여 모델의 크기를 관리하면서도 강력한 표현력을 유지합니다.

- **Technical Details**: GeoAggregator는 Gaussian-biased local attention과 global positional awareness를 통해 지리적 테이블 데이터를 효율적으로 모델링합니다. 이 모델은 기존의 deep learning 모델들이 직면하는 확장성과 유연성 문제를 해결하기 위해 설계되었습니다. 또한, GeoAggregator는 다양한 지리적 회귀 작업에 대해 높은 효율성을 보여 줍니다.

- **Performance Highlights**: GeoAggregator는 synthetic 및 empirical 지리적 데이터셋을 활용하여 다른 공간 통계 모델과 XGBoost 그리고 여러 최신 deep learning 방법들과 성능 비교를 합니다. 실험 결과, GeoAggregator는 거의 모든 데이터셋에서 경쟁 모델들에 비해 최고의 성능을 기록하였으며, 작은 모델 크기로 인해 스케일 가능성과 경량성을 갖췄음을 강조합니다.



### Interpreting Adversarial Attacks and Defences using Architectures with Enhanced Interpretability (https://arxiv.org/abs/2502.15017)
Comments:
          Publication accepted at AAAI Deployable AI conference 2025 (proof - this https URL) Total 17 pages

- **What's New**: 이 논문은 딥러닝에서의 적대적 공격(adversarial attacks)에 대한 방어로서 효과적인 방법으로 Deep Linearly Gated Networks (DLGN) 아키텍처를 활용하고 있습니다. DLGN은 일반적인 딥 네트워크 아키텍처보다 해석 가능성(interpretation capabilities)이 뛰어난 특성을 가지고 있습니다. 이 연구에서는 PGD 적대적 훈련(PGD adversarial training)을 통해 훈련된 강건 모델(robust models)을 해석하고, 이를 표준 모델과 비교합니다.

- **Technical Details**: DLGN의 특징 네트워크(feature networks)는 모델 공격자의 유일한 접근 경로로 활용됩니다. 본 논문에서는 DLGN의 완전 연결 계층(fully connected layers)과 관련하여 하이퍼플레인(hyperplanes)의 정렬(alignment), PCA와의 관계, 클래스 간 서브 네트워크(sub-network) 중첩(overlap)과 같은 속성을 분석하고, 이를 강건 모델과 표준 모델 간에 비교합니다. 또한, CNN 계층을 포함한 이 아키텍처에서 강건 모델과 표준 모델 간의 게이팅 패턴(gating patterns)을 정성적 및 정량적으로(qualitatively and quantitatively) 대비합니다.

- **Performance Highlights**: PGD-AT 모델은 데이터 포인트에서 더 먼 위치에 정렬된 하이퍼플레인을 가지며, 이는 강건한 성능을 나타냅니다. 경로 활동 분석(path activity analysis)을 통해 PGD-AT 모델은 클래스 간의 다양한 비중첩(active subnetworks)을 생성하여 공격으로 인한 게이팅 중첩을 방지하는 것을 보여줍니다. 이 연구의 시각화(visualizations) 아이디어는 PGD-AT 및 STD-TR 모델이 학습한 표현의 특성을 보여줍니다.



### TimeDistill: Efficient Long-Term Time Series Forecasting with MLP via Cross-Architecture Distillation (https://arxiv.org/abs/2502.15016)
- **What's New**: 이번 연구에서 제안하는 TimeDistill은 경량 MLP와 고급 아키텍처(예: Transformers 및 CNN) 간의 지식 증류(Knowledge Distillation) 프레임워크로, 다차원 패턴(다중 스케일 및 다중 주기) 캡처에서 두 모델 간의 서로 보완적인 특성을 활용합니다. 이를 통해 MLP의 성능을 획기적으로 높이며, 특히 18.6% 향상을 지원하고 있습니다.

- **Technical Details**: TimeDistill 프레임워크는 다중 스케일 패턴과 다중 주기 패턴을 정렬하기 위해 시간 시계를 다운샘플링하고, 주파수 영역에서 주기 분포를 정렬하기 위해 Fast Fourier Transform(FFT)을 적용합니다. 이 과정은 기존의 지식 증류와는 달리 단순히 예측을 일치시키는 것이 아니라, 구조적으로 중요한 패턴을 전달합니다.

- **Performance Highlights**: TimeDistill은 MLP의 성능을 최대 18.6% 향상시키며, 대부분의 경우 교사 모델을 초월하는 성능을 보여줍니다. 또한 7배 더 빠른 추론 시간을 가능하게 하고, 교사 모델에 비해 130배 적은 파라미터를 요구하여 높은 효율성을 자랑합니다. 이러한 결과들은 시간 시계열 예측의 효과성과 효율성을 동시에 충족시키는 혁신적인 접근법으로 평가됩니다.



### Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition (https://arxiv.org/abs/2502.15015)
Comments:
          ICLR 2025; 23 pages, 5 figures, 8 tables

- **What's New**: AlgoPerf: Training Algorithms 대회는 신경망 훈련에서 알고리즘 개선만으로 실질적인 속도 향상을 평가하는 것이 목표입니다. 이 논문에서는 10개 팀의 18개의 다양한 제출물을 포함한 최초의 AlgoPerf 대회 결과를 발표합니다. 이 대회의 결과는 하이퍼파라미터(hyperparameter) 조정 방식과 비조정 방식에서 각각 새로운 접근법의 효과를 보여줍니다.

- **Technical Details**: 외부 조정 규칙에 따른 우승 제출물은 Distributed Shampoo를 사용하여 Adam과 같은 인기 방법들보다 비대각선 전처리(non-diagonal preconditioning)의 효과를 입증했습니다. 자가 조정(self-tuning) 규칙에서의 우승 제출물은 Schedule Free AdamW 알고리즘에 기반하여 완전 하이퍼파라미터-free 훈련 알고리즘의 새로운 수준의 효과를 입증합니다. 이 연구는 다양한 훈련 알고리즘 간의 공정한 비교를 보장하기 위해 직면한 엔지니어링 도전 과제를 논의합니다.

- **Performance Highlights**: 상위 점수를 받은 제출물들은 작업 부하(workload) 변화에 대해 놀라운 내구성을 나타냈습니다. 이 결과는 앞으로의 개선 가능성이 크다는 점과 함께 현재까지의 상당한 진전을 강조합니다. 대회의 결과는 다양한 과제와 함께 훈련 알고리즘 개선이 가져올 수 있는 새로운 기회를 제시합니다.



### Towards Physics-Guided Foundation Models (https://arxiv.org/abs/2502.15013)
- **What's New**: 이 논문에서는 물리학에 기반한 기초 모델(Physics-Guided Foundation Models, PGFM)을 제안하고 있습니다. 전통적인 기초 모델들이 데이터 기반 접근법에 한정되어 있어 물리적 원칙을 무시하고 비현실적인 출력을 생성하는 문제를 해결하고자 합니다. PGFM은 과학적 법칙과 물리적 제약을 통합하여 예측의 신뢰성과 견고성을 향상시킵니다.

- **Technical Details**: PGFM은 폭넓은 분야의 지식과 기본 개념을 통합하여 기초 모델을 개선합니다. 물리적 지식을 기초 모델에 포함시키기 위한 방법으로는 물리적 제약 학습과 아키텍처 수준 통합이 있습니다. 이러한 접근 방식은 모델이 신뢰할 수 있는 예측을 하도록 돕고, 다양한 다운스트림 작업에서의 성능을 향상시킵니다.

- **Performance Highlights**: PGFM은 다양한 응용 분야에서 성능, 견고성 및 해석 가능성을 개선하는 데 기여합니다. 향후 연구에서는 PGFM 모델을 개발하고 성능 샘플 복잡성 지표를 활용하여 도메인 지식 통합을 평가할 계획입니다. PGFM의 발전은 AI의 다양한 분야에서 가능성을 열어줄 것으로 기대됩니다.



### Understanding the Design Principles of Link Prediction in Directed Settings (https://arxiv.org/abs/2502.15008)
- **What's New**: 이 논문은 방향성 링크 예측(directed link prediction)의 문제를 다루고 있습니다. 기존의 그래프 표현 학습(Graph Representation Learning, GRL) 연구가 비대칭(adjacency matrix symmetry) 가정 하에 이루어져 왔음을 지적하며, 실제 데이터에서는 방향성이 중요한 역할을 한다고 주장합니다. 방향성을 이해하고 포함하지 않을 경우, 모델의 복잡한 상호작용을 완전히 포착할 수 없음을 강조하고 있습니다.

- **Technical Details**: 연구에서는 비대칭 그래프에서 링크 예측을 위한 기본 모델로 유사도 기반( similarity-based heuristics) 접근법, 다층 퍼셉트론(Multi-Layer Perceptron, MLP), Graph Neural Networks(GNNs) 기반 모델을 비교합니다. 저자들은 링크 예측 설계 원칙을 확장하고 방향성의 영향을 분석하기 위해 다양한 실험을 수행했습니다. 이를 통해 DirLP라는 새로운 모델을 개발하여 기존의 모델들을 능가하는 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 다수의 벤치마크에서 기존 최첨단 GNN 모델보다 우수한 성능을 보임으로써 방향성 링크 예측의 새로운 기준을 세웠습니다. 따라서 본 연구는 방향성과 비대칭성을 포함한 링크 예측의 설계 원칙을 수립하여 향후 연구의 기초를 제공합니다. 링크 예측의 방향성을 고려한 접근법으로, 다양한 실제 어플리케이션에 대한 실용성을 대폭 향상시킵니다.



### Digital implementations of deep feature extractors are intrinsically informativ (https://arxiv.org/abs/2502.15004)
Comments:
          6 pages

- **What's New**: 이 논문에서는 깊은 특징 추출기(deep feature extractors)에서 정보 전파의 속도를 제안하는 상한을 증명했습니다. 이 작업은 유클리드 및 비유클리드 도메인을 포함하는 다양한 신경망 모델에 대한 통합된 프레임워크를 제시합니다. 또한 신호 도메인에 대한 구조적 정보를 추가하여 감쇠 속도를 명시적으로 결정하거나 향상시키는 방법도 보여줍니다.

- **Technical Details**: 이 연구에서 다양한 신경망 아키텍처의 무한 프로퍼게이션(depth) 및 미세한 측정 공간(measure space)에서의 에너지 감쇠를 제시하였습니다. 연구자는 LCA 그룹을 활용한 산란 CNN(scattering CNN)에서 에너지 분포의 실험적 관찰을 검증하며, 다양한 유클리드 및 비유클리드 도메인에서의 깊은 특징 추출기에 대한 안정성 정보 전파를 정량적으로 분석합니다.

- **Performance Highlights**: 결과적으로 일반적인 깊은 특징 추출기에 대한 지수적 에너지 감쇠가 도출되었습니다. 이 연구는 깊은 특징 추출기의 디지털 구현이 본질적으로 정보를 갖고 있음을 결론짓는데, 이는 에너지 전파의 속도가 신경망의 깊이에 따라 지수적으로 증가함을 의미합니다.



### Generative Modeling of Individual Behavior at Sca (https://arxiv.org/abs/2502.14998)
- **What's New**: 이번 연구에서는 인공지능(AI)을 활용하여 개별 인간 행동을 모델링하는 방법에 대해 다루고 있습니다. 기존의 연구는 주로 집합적 행동 모델링에 중점을 두었으나, 저자는 퍼스널라이즈된 AI 솔루션을 통해 개별 사용자의 요구에 맞춘 접근 방식을 제안하고 있습니다. 이를 위해 행동 스타일 메트릭스(behavioral stylometry)를 다중 작업 학습(multi-task learning) 문제로 재구성하여 AI가 각 개인의 스타일 벡터(style vector)를 효율적으로 학습하고 생성할 수 있도록 하였습니다.

- **Technical Details**: 행동 스타일 메트릭스를 다중 작업 학습 문제로 모델링하기 위해, 저자는 Low Rank Adapters (LoRAs)와 라우팅 매트릭스를 사용하여 각 플레이어에 대한 매개변수를 효율적으로 공유합니다. 이러한 구조를 통해 저자는 체스와 로켓 리그(3D 축구 게임)에 적용하여 47,864명의 플레이어와 2,000명의 플레이어에 대해 스타일 벡터를 학습하고 생성했습니다. 스타일 벡터는 각 플레이어의 행동 패턴을 생성할 수 있으며, 이는 행동 모델링 및 스타일 조정(style steering)에 도움이 됩니다.

- **Performance Highlights**: 저자들은 100개의 게임을 바탕으로 한 정밀도에서 체스 94.4%, 로켓 리그 86.7%의 정확도를 기록하였습니다. 개별 플레이어를 위한 생성적 모델의 이동 일치 정확도는 체스에서 45-69%, 로켓 리그에서 44-72%로 나타났습니다. 또한 스타일 벡터를 통해 플레이어 스타일을 조정하는 새로운 방법을 제안하며, 이는 10,177명의 셀러브리티 이미지 생성을 통해 일반성을 실증하였습니다.



### EigenShield: Causal Subspace Filtering via Random Matrix Theory for Adversarially Robust Vision-Language Models (https://arxiv.org/abs/2502.14976)
- **What's New**: EigenShield는 기존의 방어 전략들이 가진 취약성을 극복하기 위해 Random Matrix Theory를 기반으로 한 새로운 방어 기법입니다. 이 모델은 VLM(비전-언어 모델)의 고차원 표현에서 적대적 방해를 정량화하여 이를 효과적으로 탐지하고 필터링합니다. 저자들은 특히 이동 평균 구조를 이용해 의미론적 정보가 포함된 인자와 적대적 인자 간의 구분을 시도합니다.

- **Technical Details**: EigenShield의 주요 기법은 spiked covariance model을 사용하여 VLM의 특성 표현에서 의미 있는 신호 구성 요소와 적대적 노이즈를 구분하는 것에 중점을 둡니다. Robustness 기반 비정규성 점수(RbNS)와 분위수 기반 임계값을 통해 의미론적 정보가 포함된 인자(eigenvectors)를 선별합니다. 이 방법은 모델의 매개변수를 수정하거나 적대적 훈련을 요구하지 않으면서도 높은 차원의 데이터를 효과적으로 관리할 수 있는 원리를 제공합니다.

- **Performance Highlights**: EigenShield는 전통적인 방어 방식들과 비교할 때 공격 성공률을 크게 줄이는 성능을 입증했습니다. 기존의 방어 전략들인 UNIGUARD, CIDER와 같은 방법들을 초월하여 VLM의 안정성을 높이는 데 기여합니다. 이 혁신적인 접근 방식은 처리 효율성 역시 갖추고 있어 실시간 방어 시스템에서도 활용 가능할 것으로 보입니다.



### P2W: From Power Traces to Weights Matrix -- An Unconventional Transfer Learning Approach (https://arxiv.org/abs/2502.14968)
- **What's New**: 본 논문에서는 기존의 전통적인 전이 학습(Transfer Learning) 접근법의 한계를 극복하는 새로운 방법인 P2W를 제시합니다. 이 접근법은 저자들이 전이 학습하기 위해 머신 러닝 모델의 직접적인 접근이 불가능한 상황에서 보고됩니다. 특히, 임베디드 시스템에서 실행 중인 머신 러닝 모델로부터 전력 소비 데이터를 활용하여 새로운 모델의 학습을 지원합니다.

- **Technical Details**: P2W 접근법은 학습에 필요한 고품질 데이터를 수집하기 어려운 민감한 도메인, 예를 들어 의료 분야에서 적용할 수 있습니다. 주요 기법으로는 전력 소비 측정을 통해 얻어지는 전력 추적(Power Trace)를 사용하여 기존 모델의 지식을 인코딩한 근사화된 가중치 행렬을 생성합니다. 이 가중치 행렬을 이용해 새로운 모델을 초기화하고, 제한된 데이터로 추가 훈련을 진행함으로써 모델의 성능을 극대화합니다.

- **Performance Highlights**: P2W 접근법을 통해 초기 37%의 정확도에서 최대 97%에 이르는 개선된 모델 성능을 확인하였습니다. 이러한 결과는 제한된 교육 데이터 사용 시에도 전이 학습이 가능함을 입증합니다. 또한, 전력 분석 기술과 Encoder-Decoder Deep Neural Network(EDNN)의 조합을 통해 전이 학습 과정에서의 효율성을 높였습니다.



### FLEKE: Federated Locate-then-Edit Knowledge Editing (https://arxiv.org/abs/2502.15677)
- **What's New**: 본 연구는 Federated Locate-then-Edit Knowledge Editing (FLEKE)라는 새로운 작업을 제안하여 여러 클라이언트가 개인 정보 보호를 보장하면서도 협력적으로 Knowledge Editing (LEKE)을 수행할 수 있도록 합니다. 기존의 LEKE 방법들은 단일 사용자 설정에 의존하여 다중 클라이언트 환경에서는 비효율적이었습니다. FLEKE는 다수의 클라이언트가 중복된 계산을 줄이고, 서로의 지식을 최적화하면서 독립적으로 업데이트할 수 있는 가능성을 제시합니다.

- **Technical Details**: FLEKE는 두 단계의 프레임워크인 FedEdit를 사용하여 Mediator Knowledge Vectors (MKVs)를 선택하고 재사용하는 방법을 최적화합니다. 첫 번째 단계에서 클라이언트는 로컬에서 LEKE를 적용하여 MKVs를 생성하고 중앙 서버에 업로드합니다. 두 번째 단계에서는 서버에 저장된 MKVs를 코사인 유사성을 기반으로 검색하여 재편집할 수 있도록 하여 중복 계산을 최소화합니다.

- **Performance Highlights**: 실험 결과 FedEdit는 비연합 환경의 최신 방법 성능의 96% 이상을 유지하면서도 FedAvg 기반의 기준보다 약 두 배 우수한 성과를 기록했습니다. 또한, FLEKE 작업에서 MEMIT이 PMET보다 더욱 일관된 성능을 보였습니다. 연구 결과는 ML 모델의 효율적인 지식 업데이트에 대한 중요한 기여를 명확히 보여줍니다.



### Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing (https://arxiv.org/abs/2502.15666)
Comments:
          17 pages, 17 figures

- **What's New**: 대형 언어 모델(LLMs)의 텍스트 생성 사용 증가로 인해, AI 생성 콘텐츠 탐지에 대한 우려가 커지고 있습니다. 그러나 간과된 문제는 AI 도구를 사용해 인간이 작성한 콘텐츠를 미세하게 다듬은 AI-폴리시드 텍스트입니다. 최소한으로 다듬어진 텍스트도 AI 생성으로 분류되어야 하는지에 대한 중요한 질문이 제기됩니다.

- **Technical Details**: 본 연구에서는 AI-폴리시드 텍스트 평가(AI-Polished-Text Evaluation, APT-Eval) 데이터셋을 사용해 11종의 최신 AI 텍스트 탐지기를 체계적으로 평가합니다. 이 데이터셋에는 다양한 AI 개입 수준에서 다듬어진 11.7K 샘플이 포함되어 있습니다. 연구 결과, 탐지기들은 최소한으로 다듬어진 텍스트조차 AI 생성으로 잘못 분류하며, AI 개입 정도를 구별하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: 탐지기들은 또한 기존 및 더 작은 모델에 대해 편향된 결과를 보이며, 이러한 한계는 보다 세분화된 탐지 방법론의 필요성을 강조합니다. 실제로, 잘못된 분류는 표절 혐의와 AI 콘텐츠의 확산에 대한 잘못된 주장을 초래할 수 있습니다.



### Automating Curriculum Learning for Reinforcement Learning using a Skill-Based Bayesian Network (https://arxiv.org/abs/2502.15662)
- **What's New**: 본 논문에서는 자동화된 커리큘럼 생성의 어려움을 해결하기 위해 SEBNs(Skill-Environment Bayesian Networks)를 소개합니다. SEBN은 기술, 보상 구조와 관련된 목표, 환경 특징 간의 확률적 관계를 모델링하여 정책 성능을 예측합니다. 이 방법을 통해 기존의 자동 커리큘럼에 비해 더 나은 성과를 거둘 수 있는 가능성을 보여줍니다.

- **Technical Details**: SEBN은 과거 데이터를 기반으로 기술적 역량(laten competency)과 환경 특징 간의 관계를 모델링 합니다. 이를 통해 에이전트의 성공률을 예측하고, 새로운 환경에서의 성공 가능성을 바탕으로 다음 훈련 과제를 선택하는 SEBN 가이드 자동 커리큘럼을 구성합니다. 이러한 접근 방식은 각 환경에 대한 명시적 평가 없이도 가능하다는 점에서 큰 장점을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, SEBN을 활용한 커리큘럼이 DoorKey(이산격자), BipedalWalker(연속제어), robosuite(로봇 시뮬레이션) 등의 다양한 환경에서 다른 기준 모델들보다 더 빠르고 안정적인 정책을 달성하는 것으로 나타났습니다. 이는 SEBN 기반의 커리큘럼이 로봇 에이전트의 학습 성과를 극대화하는 데 효과적임을 보여줍니다.



### Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path? (https://arxiv.org/abs/2502.15657)
- **What's New**: 이번 논문에서는 일반ist AI 에이전트(agents)에 대한 위험성을 강조하고 있습니다. 특정 목표를 자율적으로 추구할 수 있는 시스템은 유용할 수 있지만, 악의적인 행위자에 의한 남용이나 인간 통제의 상실과 같은 문제를 야기할 수 있습니다. 이러한 위험에 대응하기 위해, 저자들은 비 에이전틱(non-agentic) AI 시스템, 즉 Scientist AI를 개발할 필요성을 제안합니다.

- **Technical Details**: Scientist AI는 관찰(observations)에서 세계를 설명하는 시스템으로 설계되었습니다. 이 시스템은 데이터(data)를 설명하기 위한 이론을 생성하는 세계 모델(world model)과 질문-응답 추론 기계(question-answering inference machine)로 구성됩니다. 또한, 이러한 두 구성 요소는 예측의 과신(overconfident predictions)을 완화하기 위한 명확한 불확실성(notion of uncertainty)을 바탕으로 작동합니다.

- **Performance Highlights**: Scientist AI는 AI 안전성을 포함한 과학적 진전을 가속화하는 데 인간 연구자를 지원할 수 있는 잠재력을 가지고 있습니다. 또한, 저자는 위험이 따르는 AI 에이전트에 대한 안전 장치(guardrail)로 사용할 수 있는 가능성도 제시합니다. 이러한 비 에이전틱 AI에 대한 초점을 맞추는 것은 AI 혁신의 이점을 누리면서 현재 진행되고 있는 경로에 따른 위험을 피할 수 있는 방법이 될 수 있습니다.



### Machine-generated text detection prevents language model collaps (https://arxiv.org/abs/2502.15654)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)의 훈련에서 생성된 데이터의 기원(인간 또는 합성)이 불확실할 때 모델 붕괴(model collapse)를 방지하기 위한 새로운 방법론을 제안합니다. 이를 위해 기계 생성 텍스트 탐지기에서 중요 가중치를 사용하여 데이터 분포를 재표본화(resampling)하는 방법을 설계하였습니다. 이는 모델의 성능 저하를 줄이고, 또한 인간 작성 데이터가 충분할 경우 성능을 향상시키는 결과를 보여줍니다.

- **Technical Details**: 본 연구에서는 디코딩 전략(decoding strategy)이 모델 붕괴의 정도에 미치는 영향을 분석하고, 재귀적 훈련(recursive training) 동안 생성된 데이터의 특성과 인간 참조(human references)와의 유사성을 살펴봅니다. 또한, 모델 붕괴를 평가하기 위해 세 가지 관점—작업 성능(task performance), 모델 생성 질(model generation quality), 그리고 인간 텍스트와의 의미적 유사성(semantic similarity)—에서 평가를 진행했습니다. 이러한 평가를 통해, 우리는 디코딩 전략이 모델 붕괴에 미치는 심각한 영향을 강조합니다.

- **Performance Highlights**: 제안된 방법론은 두 개의 LLM 변형(GPT-2와 SmolLM2)에서 검증되어, 인공지능 생성 텍스트 탐지기가 제공하는 확률적 추정치를 바탕으로 훈련 데이터의 재표본화가 성공적으로 모델 붕괴를 방지함을 확인했습니다. 결과적으로, 해당 방법을 통해 훈련 데이터셋에 인간 작성 데이터가 충분히 포함되어 있을 때, 모델 성능이 개선되는 것을 확인할 수 있었습니다. 이러한 실험은 텍스트 생성 작업에서 생성된 데이터의 품질 및 다양성을 보장하는 데 기여합니다.



### Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models (https://arxiv.org/abs/2502.15639)
Comments:
          34 pages

- **What's New**: 이번 연구에서는 다국어 대형 언어 모델(mLLMs)에서 언어 간 정렬(cross-lingual alignment)이 효과적으로 향상될 수 있는 방법에 대해 분석합니다. 데이터 효율적인 방법으로서 모델 개입(model interventions)을 통해 언어 모델의 출력 방향을 조정하는 기법인 '전문가 찾기(finding experts)'를 사용합니다. 연구 결과, 이 개입이 대형 언어 모델의 임베딩 공간(embedding space)을 변형시키고, 결과적으로 언어 간 정렬이 개선됨을 확인했습니다.

- **Technical Details**: 연구에서는 Aya-8B, PolyLM-13B, Bloom-7B와 같은 세 가지 오픈소스 다국어 대형 언어 모델을 선택하여, 전문가 뉴런(expert neurons)을 식별하고 개입하는 방식을 사용합니다. Flores200 데이터셋을 이용하여 특정 목표 언어에 해당하는 전문가 뉴런을 찾아내며, 이에 대한 조작 후 언어 간 임베딩 공간과 변화를 분석합니다. 이 과정에서 제안된 개입이 언어 모델의 난이도(perplexity)에 미치는 영향도 관찰됩니다.

- **Performance Highlights**: 연구 결과, 개입 이후 목표 언어의 생성 확률이 전반적으로 증가하며, 이는 개입의 성공적인 결과로 해석됩니다. 특히, 언어 간 임베딩의 거리가 줄어드는 현상이 나타났으며, 이는 교차 언어 검색(cross-lingual retrieval) 성능에서 최대 2배의 정확도 개선으로 이어졌습니다. 이러한 성과는 기존 모델들의 하위 작업(performance on downstream tasks)에서도 긍정적인 영향을 미치는 것으로 보입니다.



### Sparks of cognitive flexibility: self-guided context inference for flexible stimulus-response mapping by attentional routing (https://arxiv.org/abs/2502.15634)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 연구에서는 Wisconsin Neural Network (WiNN)를 introduced하여 빠르고 느린 학습(fast-and-slow learning) 알고리즘을 현실의 복잡한 과제에 적용했습니다. WiNN는 사전 훈련된 convolutional neural network를 활용하여 시각적으로 중요한 특징에 주의를 기울이는 'context state'를 조정할 수 있습니다. 또한 WiNN는 잘못된 반응을 한 경우, 반복적으로 context state를 업데이트하여 주의를 재집중시키고, 최소한의 매개변수 업데이트로 주의(layer)와 판독(readout layer) 층을 수정하여 동작합니다.

- **Technical Details**: WiNN의 설계는 Wisconsin Card Sorting Task (WCST)의 이미지 기반 확장에 대한 유연성 있는 인지(cognitive flexibility)를 평가합니다. WiNN는 숨겨진 규칙을 자율적으로 추론하고, 대규모 매개변수 업데이트에 의존하는 기존 모델보다 적은 예시로도 학습이 가능합니다. context-state 조정만으로 규칙 추론을 수행할 수 있으며, 이는 주의 및 판독 매개변수의 느린 업데이트에 의해 더욱 강화됩니다.

- **Performance Highlights**: 실험 결과, WiNN는 이전에 접하지 않은 조합 규칙에 대해서도 전이 가능성을 보여줍니다. WiNN는 조정된 context state만으로 규칙을 추론하고, 빠른 맥락 기반의 규칙 재매핑을 통해 눈에 띄는 유연성을 발휘합니다. 이러한 접근 방식은 복잡한 규칙 기반 과제에 빠르게 적응하는 지식 유지가 가능한 맥락 민감적 모델 개발의 길을 열어줍니다.



### Paradigms of AI Evaluation: Mapping Goals, Methodologies and Cultur (https://arxiv.org/abs/2502.15620)
- **What's New**: 최근 AI 평가 연구는 다양해진 배경과 목표를 가진 연구자들로부터 많은 관심을 받고 있으며, 이를 통해 복합적인 평가 패러다임이 등장했습니다. 이 논문은 AI 평가의 다양한 접근 방식을 조사하여 6가지 주요 패러다임을 정의하고, 각 패러다임이 지닌 목표, 방법론 및 연구 문화에 대한 최근 기여를 특징짓고 있습니다. 이를 통해 AI 시스템의 평가 접근법에 대한 인식을 높이고, 다른 패러다임 간의 상호작용을 촉진하는 것을 목표로 합니다.

- **Technical Details**: AI 평가를 정의할 때, 행동 속성과 사회적 영향을 측정하는 과정을 중점적으로 다루며, 125개 이상의 논문을 조사하여 다채로운 AI 평가 방법론을 포괄하는 관점을 제공하고 있습니다. 각 논문은 목표, 방법론, 문화와 관련된 분석 차원을 체계적으로 주석 처리하였으며, 이러한 주석을 통해 각 패러다임이 다루는 질문과 접근 방식을 명확히 하고 있습니다. 이는 연구자들이 다양한 평가 접근법을 탐색하고 비판적으로 평가할 수 있도록 돕기 위한 작업입니다.

- **Performance Highlights**: AI 평가 분야는 고유의 요소와 문제를 가지고 있으며, 이를 통해 다양한 접점을 탐색할 수 있습니다. 연구자와 실무자들 사이의 비교와 협력을 촉진하고, 향후 연구 방향을 제시하기 위해 현재 AI 평가의 간극을 식별하는 것이 중요합니다. 이러한 작업은 AI 시스템의 안전한 배포와 책임 있는 적용을 보장하는 데 기여할 잠재력을 지니고 있습니다.



### Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing (https://arxiv.org/abs/2502.15618)
Comments:
          ICLR 2025

- **What's New**: 이번 논문에서는 Probe Pruning (PP)이라는 새로운 프레임워크를 소개합니다. 이는 대규모 언어 모델(LLM)에서 온라인으로 동적으로 구조적 프루닝을 수행할 수 있으며, 각 배치의 중요한 가중치를 효율적으로 식별할 수 있는 전략을 제공합니다. PP는 1.5%의 FLOPs만 사용하여 LLM의 구조적 프루닝 효율성을 크게 개선할 수 있음을 보여줍니다.

- **Technical Details**: PP는 세 가지 주요 단계로 구성되며, 각 단계는 프로빙(probing), 기록 기반 프루닝(history-informed pruning), 전체 추론(full inference)입니다. 프로빙 단계에서는 잔여 중요성(residual importance)에 따라 숨겨진 상태의 소규모 세트를 선택하고, 이후 기록 기반 프루닝 단계에서는 선정된 프로빙 상태를 과거 상태와 통합합니다. 최종적으로 통합된 정보를 바탕으로 중요도 점수를 사용하여 가중치를 구조적으로 프루닝합니다.

- **Performance Highlights**: LLaMA-2-7B 모델에 대한 평가 결과, PP는 기존에 비해 40% 프루닝 비율에서 성능 저하율을 2.56배 낮추는 성과를 내었습니다. 실험을 통해 PP가 최소한의 프로빙을 통해도 모델의 성능을 유지할 수 있으며, 다양한 입력 배치에서 동적 아울라이어를 효과적으로 처리할 수 있음을 확인하였습니다. 이는 실제 응용에서 리소스를 절약하면서도 높은 성능을 유지하는 데 기여할 것입니다.



### On the Robustness of Transformers against Context Hijacking for Linear Classification (https://arxiv.org/abs/2502.15609)
- **What's New**: 이번 논문에서는 Transformer 기반의 Large Language Models (LLMs)의 context hijacking 현상을 다루고 있습니다. 이 현상은 사실상 올바른 정보가 포함된 context가 LLM의 예측을 방해하는 문제로, 모델의 강인성에 중대한 이슈를 제기합니다. 연구자들은 최근의 Linear Transformers 발전을 바탕으로 이 현상을 이론적으로 분석하였으며, 이러한 분석은 Transformer 아키텍처에 대한 깊은 이해를 제공합니다.

- **Technical Details**: 이 논문은 context hijacking에 대한 robustness (강인성)을 분석하기 위해 다층 linear transformer 모델을 활용했습니다. 연구자들은 hijacking 예제를 반대 라벨을 가진 query-answer 쌍으로 설정하고 multi-step gradient descent를 이용하여 최적의 학습률과 초기화를 관찰했습니다. 그 결과, 깊은 Transformer는 더 섬세한 최적화 단계를 수행할 수 있어 hijacking 예제의 영향을 줄일 수 있다는 것을 입증했습니다.

- **Performance Highlights**: 실험 결과, 더 깊은 Transformer 구조가 더 높은 강인성을 보임을 확인하였습니다. 이는 더 많은 prepended context 예제가 사용될 때 모델의 예측이 바뀌기 어려워진다는 것을 의미합니다. 본 연구는 Linear Problems에 대한 Gradient Descent 방법을 사용하는 타 분야에서도 독립적인 관심을 가질 수 있는 결과를 제시합니다.



### Do Multilingual LLMs Think In English? (https://arxiv.org/abs/2502.15603)
Comments:
          Main paper 9 pages; including appendix 48 pages

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 다국어 작업을 수행하더라도 가장 중요한 결정을 영어에 가까운 표현 공간에서 내린다는 사실을 보여줍니다. 연구진은 프랑스어, 독일어, 네덜란드어, 중국어 문장에 대한 내부 표현을 탐구하며, LLM이 의미적으로 중요한 단어에 대해서 먼저 영어에 가까운 표현을 생성한 후 그것을 목표 언어로 번역하는 과정을 따릅니다. 이 결과는 다국어 LLM이 영어에 의해 크게 형성된 방식으로 이유를 공개하지 않고 작동함을 시사합니다.

- **Technical Details**: 이 연구에서는 4개의 오픈 소스 모델(Llama-3.1-70B, Gemma-2-27b, Aya-23-35B, Mixtral-8x22B)의 성능을 비교 분석하며, 이 모델들이 내부 표현 공간에서 어떻게 작동하는지를 상세히 설명합니다. 로그잇 렌즈(logit lens)와 인과 추적(causal tracing), 그리고 스티어링 벡터(steering vectors)의 세 가지 해석 방법으로 모델의 내부 표현을 조사합니다. 특히, 영어에서 생성된 스티어링 벡터를 사용했을 때 비영어 문장의 결과를 더 효과적으로 조정할 수 있다는 점이 중요합니다.

- **Performance Highlights**: 이 연구는 LLM들이 영어 표현 공간에서 결정을 내리기 때문에 비영어 문장 생성에서 성능이 저하된다는 점을 강조합니다. 특히, 모델의 언어 범위에 따라 성능 차이를 분석하면서, 결과적으로 LLM의 영어 중심 행동이 다양한 언어 설정에서의 모델의 공정성, 신뢰성, 그리고 강인성에 영향을 미친다고 설명합니다. 이번 연구는 LLM의 다국어 처리에서의 한계를 이해하는 데 중요한 통찰력을 제공합니다.



### KAD: No More FAD! An Effective and Efficient Evaluation Metric for Audio Generation (https://arxiv.org/abs/2502.15602)
- **What's New**: 이번 논문에서는 Fréchet Audio Distance (FAD)의 한계를 극복하기 위한 새로운 메트릭, Kernel Audio Distance (KAD)를 소개합니다. KAD는 Maximum Mean Discrepancy (MMD) 기반으로 설계되어, 분포에 대한 가정 없이도 오디오 샘플 간의 유사성을 평가할 수 있습니다. KAD는 더 작은 샘플 수로도 빠른 수렴을 보여주며, 낮은 계산 비용과 더 강한 인간 인지 판단과의 정렬성을 가지고 있습니다.

- **Technical Details**: KAD는 Gaussian 분포 가정이 필요하지 않으며, 오디오 임베딩의 더 복잡한 특성을 잘 포착합니다. KAD는 표준 오디오 인코더 모델인 VGGish를 통해 추출된 임베딩을 사용하여 실질적인 특성을 반영하여 신뢰할 수 있는 평가를 제공합니다. 도구킷 kadtk로 오픈소스화되어 개발자들이 쉽게 활용할 수 있습니다.

- **Performance Highlights**: KAD는 샘플 크기에 따른 수렴 속도가 빠르며, GPU 가속을 통해 계산 비용이 낮아집니다.의사용 예측과의 상관관계 또한 더욱 뛰어난 성능을 보입니다. KAD는 생성된 오디오 모델의 품질 평가를 위한 효율적이고 신뢰할 수 있는 기준을 제공합니다.



### LightThinker: Thinking Step-by-Step Compression (https://arxiv.org/abs/2502.15589)
- **What's New**: 이번 논문에서는 LightThinker라는 새로운 방법을 제안합니다. 이 방법은 대형 언어 모델(LLMs)이 추론 중 중간 사고를 동적으로 압축할 수 있게 해줍니다. 사람의 인지 과정에서 영감을 받아, 경량화된 표현으로 불필요한 추론 단계를 제거함으로써 메모리 사용량과 연산 비용을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: LightThinker는 LLM이 언제 어떻게 압축을 수행할지 학습하도록 훈련시킵니다. 데이터 구성, 숨겨진 상태(hidden states) 매핑, 전문적인 주의 마스크(attention masks)를 사용하여 압축 로직을 조정하고, Dependency (Dep) 메트릭을 도입하여 압축 정도를 정량화합니다. 이 메트릭은 생성된 각 토큰이 얼마나 많은 역사적 토큰에 의존하는지를 측정합니다.

- **Performance Highlights**: 실험 결과, LightThinker는 Qwen 모델에서 피크 토큰 사용량을 70% 줄이고, 추론 시간을 26% 감소시키며, 정확도를 크게 저하시키지 않고(1% 감소) 효율성을 높였습니다. 이는 복잡한 추론 작업에서 LLM의 효율성을 향상시킬 수 있는 새로운 방향을 제시합니다.



### Context-Aware Doubly-Robust Semi-Supervised Learning (https://arxiv.org/abs/2502.15577)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 통신 시스템에서 인공지능(AI)의 활용에 있어, 네트워크 디지털 트윈(NDT)을 활용한 고급 데이터 보강 기술을 소개합니다. 새로운 기법인 컨텍스트-어웨어 더블 로버스트(CDR) 학습은 다양한 맥락에서 NDT의 정확도에 따라 의존성을 조절하며, 이는 기존의 반지도 학습 방법보다 뛰어난 성능을 보입니다.

- **Technical Details**: CDR 학습은 반지도 학습 설정을 기반으로 하며, 라벨이 있는 데이터와 라벨이 없는 데이터를 함께 사용할 수 있습니다. 컨텍스트 변수 C가 주어지는 상황에서, CDR은 정확한 의사 레이블을 기반으로 가중치를 조절하여 더 정확한 학습이 가능하게 합니다. 이 방법은 기존의 P-ERM(의사 경험적 위험 최소화) 접근 방식의 한계를 극복합니다.

- **Performance Highlights**: CDR 기법은 다운링크 빔포밍(downlink beamforming) 작업에서 기존의 반지도 학습 방법들과 비교했을 때 더 우수한 성능을 보였습니다. 다양한 시뮬레이션 실험에서 CDR은 높은 정확도와 신뢰성을 유지하며, 맥락에 따라 의사 레이블의 영향력을 조절할 수 있는 능력을 입증하였습니다.



### Feature maps for the Laplacian kernel and its generalizations (https://arxiv.org/abs/2502.15575)
- **What's New**: 최근 머신 러닝의 커널 방법(Kernel methods) 분야에서 Laplacian kernel의 재조명되고 있습니다. 이는 Gaussian kernel과 비교하여 bandwidth hyperparameter에 대한 안정성이 뛰어난 점과, 심층 완전 연결 네트워크의 neural tangent kernel과 동등한 표현력을 가지기 때문입니다. 하지만 Laplacian kernel은 Gaussian kernel과 달리 비분리형으로, 이를 근사하는 데 있어 여러 도전 과제가 있습니다.

- **Technical Details**: 본 연구에서는 Laplacian kernel 및 Matérn kernel과 Exponential power kernel 등 두 가지 일반화를 위한 random features를 제시합니다. 우리는 랜덤 피처가 이들 커널을 근사할 수 있도록 효율적으로 무작위 가중치 행렬을 샘플링하는 방안을 제공합니다. 가중치 행렬은 약하게 결합된 heavy-tailed randomness를 가지며, 이는 연구의 핵심 기술적 부분을 포괄합니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 수치 실험을 통해 제공된 랜덤 피처 맵(random feature maps)의 효과성을 입증하였습니다. 연구는 Laplacian kernel의 근사를 통한 성능 향상을 목표로 하며, 이는 머신 러닝 모델의 효율적인 구현에 기여할 것으로 기대됩니다.



### Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation (https://arxiv.org/abs/2502.15563)
- **What's New**: AI 모델의 신뢰성 있는 평가 방법이 과학적 발전과 실제 응용을 위한 중요한 요소로 부각되고 있습니다. 기존의 VLM 벤치마크는 모델 능력에 대한 일반적인 통찰력을 제공하지만, 이질적인 설계와 특정 이미지 도메인에 국한된 접근 방식 때문에 여러 문제에 직면하고 있습니다. 이를 해결하기 위해, 저자는 리소스 효율적인 도메인 특화 VLM 벤치마크 생성 프레임워크와 함께 새로운 VLM 벤치마크를 제공하며, 22개의 최신 VLM 모델에 대한 광범위한 벤치마킹을 실시했습니다.

- **Technical Details**: 아이디어의 핵심은 단일 이미지로부터 다양한 과제를 생성할 수 있는 작업 증강(task augmentation) 프레임워크에 있습니다. 이 프레임워크는 인스턴스 분할(instance segmentation) 주석을 다양한 인식 과제들로 변환하여, 인식 능력을 테스트하는 여러 문제를 만듭니다. 저자들은 이 방법론을 적용하여 7개의 새 도메인 특화 VLM 벤치마크를 생성하고, 37,171개의 과제에 대해 22개의 모델을 포괄적으로 평가했습니다.

- **Performance Highlights**: 이 연구의 결과는 벤치마크의 설계가 도메인과 과제에 따라 성능 차이를 드러내며, 개인화된 벤치마크가 필요함을 시사합니다. 162,946개의 인간 검증 응답을 수집하여, 모델 평가의 강력한 기준점을 설정했습니다. 이로 인해 연구자들은 특정 도메인에 맞춤화된 모델 선택을 할 수 있는 길을 열고, 향후 연구 방향을 개선할 수 있습니다.



### A Defensive Framework Against Adversarial Attacks on Machine Learning-Based Network Intrusion Detection Systems (https://arxiv.org/abs/2502.15561)
Comments:
          Accepted to IEEE AI+ TrustCom 2024

- **What's New**: 이번 연구에서는 최신 사이버 공격에 대응하기 위해 새로운 방어 프레임워크를 제안합니다. 이 프레임워크는 adversarial training, dataset balancing 기법, 고급 feature engineering, ensemble learning 및 모델 세부 조정을 통합하여 ML 기반의 NIDS의 강인성을 강화합니다. 이를 통해 사이버 공격에 대한 탐지 정확도를 35% 향상시키고, false positive 비율을 12.5% 감소시킬 수 있음을 입증했습니다.

- **Technical Details**: ML 기반의 NIDS는 정상적인 네트워크 트래픽의 행동을 학습하여 이상 상황을 감지합니다. 전통적인 signature 기반 시스템과 달리, ML 모델들은 새로운 제로데이 공격과 같은 알려지지 않은 위협을 탐지하는 데 유리합니다. 그러나 adversarial 공격에 매우 취약하여, 공격자가 트래픽을 조작하여 악성 활동을 정상으로 오인하게 만드는 문제가 발생합니다.

- **Performance Highlights**: 제안된 방어 프레임워크는 두 개의 데이터셋(NSL-KDD, UNSW-NB15)에서 실험을 통해 검증되었습니다. 실험 결과, adversarial 공격 상황에서도 탐지 정확도가 평균 35% 향상되었고, false positive 비율이 12.5% 감소했습니다. 이러한 실적은 ML 기반의 NIDS가 실제 네트워크 환경에서 보다 효과적으로 배치될 수 있도록 하는 중요한 단계를 나타냅니다.



### Estimating Vehicle Speed on Roadways Using RNNs and Transformers: A Video-based Approach (https://arxiv.org/abs/2502.15545)
- **What's New**: 이번 프로젝트는 고급 기계 학습 모델, 특히 Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), 그리고 Transformers를 활용하여 비디오 데이터를 통한 차량 속도 추정 작업을 탐구합니다. 전통적인 속도 추정 방법들은 높은 비용과 제한된 범위로 인해 제약이 있는 반면, 기존의 감시 인프라와 최첨단 신경망 구조를 활용하여 비침습적이고 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: 이 연구는 LSTM 및 GRU 모델을 통해 비디오 프레임의 시간 순서 내에서 장기 의존성을 효과적으로 관리하며, Transformers는 자체 주의 메커니즘을 사용하여 전체 시퀀스를 병렬로 처리하며 데이터의 가장 유의미한 부분에 집중합니다. 입력 데이터의 시퀀스 길이를 늘리면 모델 정확도가 일관되게 향상되고, 특히 Transformers는 다양한 시퀀스 길이와 복잡성에서 뛰어난 적응력과 강건성을 보여주어 다양한 교통 조건의 실시간 애플리케이션에 적합합니다.

- **Performance Highlights**: 모델 결과는 LSTM과 GRU가 일반적인 Recurrent Neural Networks (RNNs)보다 뛰어난 성능을 보이며, 컨텍스트 정보를 중요시하는 동적 환경에서의 정확성이 향상됨을 보여줍니다. 이러한 정교한 신경망 모델의 통합은 자동화된 속도 감지 시스템의 정확성과 신뢰성을 크게 향상시킬 수 있으며, 이는 교통 관리 및 도로 안전의 혁신적인 발전을 전망하게 합니다.



### Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors (https://arxiv.org/abs/2502.15540)
Comments:
          Accepted as a Spotlight Paper at ICLR 2025

- **What's New**: 본 연구는 representation learning 알고리즘의 일반화 오류에 대한 새로운 기대값(in-expectation) 및 꼬리(tail) 경계를 확립하였습니다. 이 경계는 훈련과 테스트 데이터 세트에서 추출된 표현의 분포와 데이터 종속적인 대칭 prior, 즉 Minimum Description Length (MDL) 간의 상대적인 엔트로피를 기준으로 설정됩니다. 이는 인코더의 구조(structure)와 단순성(simplicity)을 반영하며, 기존의 연구에 비해 상당히 개선된 결과를 나타냅니다.

- **Technical Details**: 연구진은 기대값 경계를 활용하여 데이터 종속적인 정규화 기법(regularizer)을 제안하며, prior 선택에 관한 중요한 질문을 깊이 탐구하였습니다. 이를 통해 데이터 종속적인 Gaussian mixture prior를 동시에 학습하고 이를 정규화 기법으로 사용하는 체계적인 접근 방식을 제안합니다. 흥미롭게도, 이 과정에서 가중치 주의 메커니즘(weighted attention mechanism)이 자연스럽게 나타나는 것을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 현재 인기 있는 Variational Information Bottleneck (VIB) 기법과 최근의 Category-Dependent VIB (CDVIB) 방법보다 뛰어난 성능을 보임을 입증하였습니다. 이러한 성능 향상은 제안된 정규화 기법과 관련하여 효율적인 데이터 종속적 학습의 중요성을 강조합니다.



### Sheaf theory: from deep geometry to deep learning (https://arxiv.org/abs/2502.15476)
Comments:
          117 pages, 8 figures

- **What's New**: 이번 논문은 sheaf 이론의 적용이 심층 학습과 데이터 과학, 컴퓨터 과학 전반에 걸쳐 어떻게 이루어지는지를 다루고 있습니다. 저자들은 이 논문이 수학적 개념에 대한 친절한 소개 역할을 한다고 강조하며, 기초적인 수학적 지식이 있는 독자들이 접근할 수 있도록 구성되어 있습니다. 특히, 이론적 연구자와 실용적 연구자 모두가 공유하는 직관과 동기를 설명하며, 클래식한 수학적 이론과 최신 응용 간의 다리를 놓고 있습니다.

- **Technical Details**: sheaf 이론은 복잡한 수학의 한 분야로, 이론은 Jean Leray가 개발하였으며, Alexander Grothendieck에 의해 개념화되었습니다. 이 이론은 일반적인 posets에서의 sheaf의 적용을 탐구하고, 새로운 알고리즘을 제안하여 유한한 posets에서 sheaf cohomology를 계산합니다. 문서는 sheaf 이론이 데이터 과학 및 심층 학습에서 어떻게 활용될 수 있는지에 대한 체계적인 정리를 제공합니다.

- **Performance Highlights**: sheaf 이론은 최근 몇 년 동안 흥미로운 방법론의 융합을 통해 심층 학습에 중요한 기여를 하고 있습니다. 특히, sheaf Laplacians 및 heat diffusion 같은 개념을 통해 새로운 신경망 아키텍처가 제안되었습니다. 이 논문은 sheaf 이론의 다양한 실용적 적용과 더불어, 대수적 도구를 통해 기하학적 직관을 확립할 수 있는 강력한 도구를 제공함을 강조합니다.



### PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System (https://arxiv.org/abs/2502.15470)
Comments:
          To appear in ASPLOS 2025

- **What's New**: 이 논문은 LLM(대형 언어 모델) 디코딩의 성능을 개선하기 위해 동적인 특성을 고려한 PAPI(PArallel Decoding with PIM)를 제안합니다. PAPI는 메모리 집약적 또는 계산 집약적 커널에 대한 동적 스케줄링을 통해 다양한 하드웨어 유닛으로 최적의 커널 할당을 가능하게 합니다. 이를 통해 기존의 정적 커널 매핑의 비효율성을 극복하고 보다 효율적인 LLM 추론을 실현합니다.

- **Technical Details**: PAPI는 실행 시간에 커널의 특성을 동적으로 식별하여 적합한 하드웨어 유닛에 커널을 할당하는 온라인 커널 특성화 메커니즘을 포함하고 있습니다. 또한, PIM(Processing-in-Memory) 유닛과 계산 중심의 가속기가 조화롭게 동작하는 이기종 컴퓨팅 시스템을 구현하여 서로 다른 컴퓨팅 및 메모리 수요를 충족합니다. 이러한 시스템은 다양한 메모리 집약적 및 계산 집약적 커널의 요구를 적절히 충족시키는 구조를 갖추고 있습니다.

- **Performance Highlights**: 실험 결과 PAPI는 기존의 최첨단 이기종 LLM 가속기보다 1.8배, PIM 전용 LLM 가속기보다 11.1배 빠른 성능을 기록했습니다. 이러한 성과는 LLM 디코딩의 동적으로 변화하는 특성을 최대한 활용한 결과로, PAPI가 에너지 효율성에서도 현저한 개선을 보여줍니다. 향후 LLM 추론 시스템의 성능을 더욱 향상시키는 데 기여할 것으로 기대됩니다.



### Dimension-free bounds in high-dimensional linear regression via error-in-operator approach (https://arxiv.org/abs/2502.15437)
Comments:
          100 pages

- **What's New**: 이번 논문에서는 랜덤 디자인(random design)의 고차원 선형 회귀(high-dimensional linear regression) 문제를 다루고 있습니다. 제안된 새로운 접근 방식인 error-in-operator는 설계 공분산(design covariance) $	au$를 직접 추정하지 않고 경험적 위험 최소화(empirical risk minimization)에 통합하는 방식을 채택합니다. 이 접근법은 문제의 유효 차원이 보조 변수(auxiliary variables)에 의해 증가하지 않음을 입증하는 데 기여합니다.

- **Technical Details**: 먼저, 과도한 예측 위험의 확장을 제시하며, 주요 항과 나머지에 대한 비대칭적(dimension-free) 경계를 도출합니다. 이 이론적 분석을 통해 적절히 조정된 절차의 파라미터가 있을 경우 보조 변수가 문제의 유효 차원에 미치는 영향을 최소화할 수 있음을 보여줍니다. 또한, 이 방법의 계산적인 측면(computational aspects)을 논의합니다.

- **Performance Highlights**: 마지막으로, 제안된 방법의 성능을 수치 실험(numerical experiments)으로 설명하며, 기존 방법들과의 비교를 통해 우수성을 입증합니다. 이 연구는 고차원 회귀 문제에서의 효율성을 높이는 데 중요한 통찰을 제공합니다.



### Adversarial Prompt Evaluation: Systematic Benchmarking of Guardrails Against Prompt Input Attacks on LLMs (https://arxiv.org/abs/2502.15427)
Comments:
          NeurIPS 2024, Safe Generative AI Workshop

- **What's New**: 대형 언어 모델(LLMs)이 일상 애플리케이션에 통합됨에 따라, 이러한 모델의 견고성과 보안을 보장하는 것이 점점 더 중요해지고 있습니다. 특히, jailbreaks라 불리는 프롬프트에 의해 LLM이 안전하지 않은 행동을 해버리는 경우가 많습니다. 본 연구는 15가지의 다양한 방어 기법을 체계적으로 벤치마킹하여 다양한 악의적 및 무해한 데이터셋에서 성능을 비교하였으며, 현재 이용 가능한 데이터셋을 기반으로 간단한 기준선 모델이 많은 최신 방어 기법들과 경쟁력 있는 성능을 보여줌을 밝혔습니다.

- **Technical Details**: 본 연구는 ‘안전 교육(safety training)’ 또는 ‘정렬(alignment)’이라고 불리는 과정을 통해 LLM의 안전한 출력을 보장하기 위한 다양한 방어 수단인 guardrails를 다룹니다. 연구에서는 perplexity 필터, 입력 프롬프트 패러프레이징 도구, 키워드 탐지기 등을 포함한 여러 guardrails의 성능을 평가했습니다. 또한 방어 기법의 효율성을 평가하기 위해 다양한 공격 스타일에 대한 방어 성능을 시스템적으로 측정하였으며, 데이터셋의 구성과 정확한 평가 방법이 방어 기법의 성능에 큰 영향을 미친다고 지적합니다.

- **Performance Highlights**: 연구 결과, 다양한 jailbreak 스타일에 따라 방어 기법의 성능이 크게 달라진다는 점이 확인되었습니다. 또한, 현재 사용 가능한 데이터셋을 활용했을 때, 몇몇 기본적인 모델이 최신 방어 기법에 비해 경쟁력 있는 OOD(out-of-distribution) 성능을 보여주었습니다. 향후 연구 방향으로는 더 정교한 공격을 방어하기 위한 효과적인 guardrails의 개발과 지원을 제안하고 있습니다.



### TAG: A Decentralized Framework for Multi-Agent Hierarchical Reinforcement Learning (https://arxiv.org/abs/2502.15425)
- **What's New**: 기존의 단일 구조가 아니라, 완전히 분산화된 계층적 다중 에이전트 시스템을 구축할 수 있는 TAME Agent Framework (TAG)를 소개합니다. TAG는 새로운 LevelEnv 개념을 통해 임의 깊이의 계층을 가능하게 하여 각각의 계층을 에이전트가 상호작용하는 환경으로 추상화합니다. 이를 통해 다양한 에이전트 타입의 원활한 통합이 가능해지며, 정보 흐름을 표준화하면서 느슨한 결합을 유지합니다.

- **Technical Details**: TAG의 핵심 혁신은 각 계층 수준을 환경으로 간주하는 LevelEnv 추상화입니다. 이 접근 방식은 여러 수평적 계층으로 구성된 시스템을 만들어, 에이전트가 아래 계층과 상호작용할 때의 복잡성을 줄여줍니다. 또한, 중앙 집중식 제어 없이 효율적인 조정을 가능하게 하는 유연한 통신 프로토콜을 지원하며, 이질적인 에이전트도 다양한 학습 알고리즘을 통해 사용할 수 있도록 합니다.

- **Performance Highlights**: TAG의 효과는 기존의 멀티-에이전트 강화 학습(MARL) 벤치마크에서의 실험을 통해 입증되었습니다. 여러 두 개 및 세 개의 수준의 계층을 구현하여 표준 벤치마크에 대해 샘플 효율성과 최종 성능이 개선된 결과를 보여주었습니다. 이러한 성과는 분산된 계층적 조직이 학습 속도와 최종 성능을 모두 향상시킬 수 있음을 시사합니다.



### Fréchet Cumulative Covariance Net for Deep Nonlinear Sufficient Dimension Reduction with Random Objects (https://arxiv.org/abs/2502.15374)
- **What's New**: 이 논문은 비선형 충족 차원 축소(nonlinear sufficient dimension reduction, SDR)를 위한 새로운 통계적 의존성 척도인 Fréchet 누적 공분산(Fréchet Cumulative Covariance, FCCov)을 도입하며, FCCov을 기반으로 한 비선형 SDR 프레임워크를 개발하였습니다. 이 방법은 복잡한 비유클리드 데이터에 적합하며, 이상치에 대한 견고성 또한 제공합니다. 또한, Feedforward Neural Networks (FNNs) 및 Convolutional Neural Networks (CNNs)를 활용하여 샘플 수준에서 비선형 충분 방향을 추정하는 방안을 제시합니다.

- **Technical Details**: FCCov는 비유클리드 응답 객체와 유클리드 예측 변수 간의 조건부 평균 독립성을 정량화하기 위한 새로운 통계적 측정 방법입니다. 이 새로운 척도를 적용하여 비선형 SDR을 제한된 최적화 프레임워크로 재구성하였습니다. 이 방법은 squared Frobenius norm 정규화를 활용하여 비선형 SDR의 편향성을 특정 σ-필드 수준에서 달성할 수 있음을 이론적으로 입증하였습니다.

- **Performance Highlights**: 이 논문에서 제시된 방법은 광범위한 시뮬레이션 연구를 통해 유클리드 및 비유클리드 환경에서의 성능을 검증하였습니다. 인간의 얼굴 표정 인식과 같은 실제 데이터셋을 적용하여 결과를 도출하였고, 이 방법이 보다 현실적이고 넓은 적용 가능성을 지닌다는 것을 입증하였습니다. 또한, 제안하는 방법은 기존 방법들과의 포괄적인 비교를 통해 이상치에 대한 견고성을 평가하였습니다.



### AttentionEngine: A Versatile Framework for Efficient Attention Mechanisms on Diverse Hardware Platforms (https://arxiv.org/abs/2502.15349)
Comments:
          15 pages

- **What's New**: 이번 논문에서는 AttentionEngine이라는 포괄적인 프레임워크를 소개합니다. 이 프레임워크는 다양한 하드웨어 플랫폼에서 Attention 메커니즘의 최적화를 지원하며, 수동 조작을 최소화하여 성능 향상을 도모합니다. AttentionEngine은 모듈화된 연산으로 Attention 계산을 분해하여 다양한 알고리즘 요구 사항에 유연하게 적응할 수 있도록 합니다.

- **Technical Details**: AttentionEngine은 두 가지 기본 연산, 즉 relevance scoring과 aggregation으로 Attention 메커니즘을 추상화합니다. 이를 통해 상관성 점수를 계산하고, 사용자 정의 기능을 추가하여 사용자가 원하는 주의 변형을 디자인할 수 있는 템플릿을 제공합니다. 이 프레임워크는 입력 구성 및 하드웨어 제약에 맞춰 자동으로 최적화를 수행하여 다양한 Attention 변형과 하드웨어 플랫폼을 지원합니다.

- **Performance Highlights**: 실험 결과, AttentionEngine은 기존의 수작업 최적화된 커널과 동급의 성능을 발휘하며, 지원되지 않는 구성에서는 최대 10.4배의 속도 향상을 자랑합니다. 또한, AttentionEngine는 사용자 정의 Attention 메커니즘의 설계 및 최적화에 대한 유례없는 유연성을 제공하여 확장 가능하고 일반화 가능한 Attention 계산의 중요한 진전을 나타냅니다.



### Drug-Target Interaction/Affinity Prediction: Deep Learning Models and Advances Review (https://arxiv.org/abs/2502.15346)
Comments:
          64 pages, 7 figures, 10 tables

- **What's New**: 이 논문은 약물 발견 (Drug Discovery) 절차와 그 과정에서의 약물-타겟 상호작용 (Drug-Target Interaction, DTI) 예측의 중요성에 대해 설명합니다. 딥러닝 모델이 약물-타겟 상호작용 예측의 복잡한 관계를 포착하는 데 어떻게 기여할 수 있는지를 보여주며, 2016년부터 2025년까지 180개의 예측 방법을 분석하였습니다. 이를 통해 연구자들에게 더욱 효율적이고 정확한 방법들을 제안하고 있습니다.

- **Technical Details**: 논문은 DTI/DTA 예측에 대한 다양한 입력 표현 (Input Representations) 및 기계 학습 (Machine Learning) 기술을 기술하고 있습니다. 특히, SMILES, fingerprints 및 분자 그래프 (Molecular Graphs)와 같은 약물에 대한 입력 표현을 다루고, 단백질에 대해서는 기본 시퀀스와 단백질 그래프를 포함한 여러 표현 방식을 소개합니다. 이러한 다양한 표현 방식은 모델들이 효과적으로 학습할 수 있도록 돕습니다.

- **Performance Highlights**: 기존의 방법들이 한정된 표현 방식을 사용하는 반면, 본 논문은 다채로운 구조 및 표현을 활용하여 DTI 예측 모델의 성능을 향상시키고자 합니다. 연구자들이 사용할 수 있는 다양한 수준의 입력 표현과 데이터 세트를 제공하여, 향후 약물 개발 수명 주기를 빠르게 진행할 수 있도록 기여할 것입니다. 최종적으로, 성공적인 약물 개발을 위한 도전 과제와 미래 연구 방향에 대한 통찰력을 공유하고 있습니다.



### Attention Eclipse: Manipulating Attention to Bypass LLM Safety-Alignmen (https://arxiv.org/abs/2502.15334)
- **What's New**: 최근 연구에 따르면 정교하게 설계된 jailbreak 입력이 큰 언어 모델(Large Language Models, LLMs)로 하여금 유해한 출력을 생성하도록 유도할 수 있다는 것이 밝혀졌습니다. 본 논문에서는 모델의 attention을 조작하여 특정 프롬프트의 일부에 대한 주의를 강화하거나 약화시키는 새로운 jailbreak 공격 생성 방법을 제안합니다. 이를 통해 attention 손실을 활용하여 기존의 공격 알고리즘보다 더 효과적이고 전이 가능한 공격을 개발했습니다.

- **Technical Details**: 작성된 공격은 두 가지 방식으로 attention을 조작합니다. 첫째, 스캐폴딩 기법을 통해 안전한 내용 부분과 악의적 의도가 포함된 부분의 의존성을 형성하여 공격자가 해로운 내용을 허가적인 프롬프트 내에 삽입할 수 있도록 합니다. 둘째, 악의적 접미사를 숨기기 위해 프롬프트 내에서 모델의 attention 분포를 제어하여 악성 프롬프트가 alignment 제약을 우회하도록 하는 기술을 사용합니다.

- **Performance Highlights**: 이 공격 기법은 여러 최근 모델에 적용되며, 공격 성공률을 현저히 향상시킵니다. 특히 기존의 GCG, AutoDAN, ReNeLLM 등의 공격과 결합할 때, 공격 생성 소요 시간을 감소시키며 성공적인 jailbreak를 위한 전체 반복 횟수를 줄입니다. 전반적으로 제안된 방법은 다양한 LLM 구조에 대한 기존의 jailbreak 기술을 향상시키는 일반화 가능한 프레임워크를 제공하며, 모델 패밀리 간의 전이성을 강조합니다.



### Lightweight yet Efficient: An External Attentive Graph Convolutional Network with Positional Prompts for Sequential Recommendation (https://arxiv.org/abs/2502.15331)
Comments:
          26 pages, 8 figures, journal paper, accepted by TOIS at 20th February, 2025

- **What's New**: 본 논문은 External Attentive Graph convolutional network with Positional prompts for Sequential recommendation (EA-GPS)을 제안합니다. EA-GPS는 사용자와 아이템 간의 상호작용 및 아이템 간의 순차적 관계를 효과적으로 처리하는 그래프 기반의 시스템입니다. 이 시스템은 외부 메모리 유닛을 통해 글로벌 연관성을 선형으로 측정하고, 포지션 프롬프트 기반의 디코더를 통해 복잡한 위치 종속성을 명시적으로 다룹니다.

- **Technical Details**: EA-GPS는 두 개의 외부 메모리 유닛을 통한 외부 에디터(External Attention, EA)를 포함하여, 아이템의 절대 위치를 프롬프트로 간주합니다. 또한, 길이 적응형 순차 마스킹(length-adaptive sequential masking)과 소프트 어텐션 네트워크를 채택하여 모델이 장기적인 위치 종속성과 문맥 관계를 포착할 수 있도록 지원합니다. 이러한 기술적 접근은 그래프 인코더의 높은 계산 복잡도를 완화합니다.

- **Performance Highlights**: 다섯 개의 실제 데이터셋에서 EA-GPS의 성능을 평가한 결과, 기존의 선진 모델들에 비해 우수한 성능이 입증되었습니다. 특히, EA-GPS는 더 적은 파라미터 수와 낮은 학습 오버헤드로 뛰어난 성능을 유지하며, 자원 제약이 있는 엣지 디바이스에서도 효율적으로 사용할 수 있도록 설계되었습니다.



### Utilizing Sequential Information of General Lab-test Results and Diagnoses History for Differential Diagnosis of Dementia (https://arxiv.org/abs/2502.15317)
Comments:
          7 pages, 6 figures. This work has been submitted to the IEEE for possible publication

- **What's New**: 알츠하이머병(Alzheimer's Disease, AD)의 조기 진단을 위한 새로운 접근 방식이 소개되었습니다. 기존의 제한된 데이터 접근성과 단일 지표에 대한 과도한 의존도를 극복하기 위해, 일반적으로 수집되는 혈액 검사 이력을 활용하여 AD를 조기에 탐지하고 감별 진단을 할 수 있는 방법론이 제안됩니다.

- **Technical Details**: 이 연구에서는 검사 결과를 '문장(sentence)'으로 모델링하여, 단어 임베딩(word embedding) 기법을 사용해 검사 간의 잠재적 관계를 포착합니다. 또한, LSTM(long-short-term memory)과 Transformer 네트워크와 같은 심층 시계열 모델을 활용하여 환자 기록의 시간적 패턴을 모델링합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 진단 정확성을 개선하였으며, 다양한 임상 환경에서 AD 선별 검사의 확장성과 비용 효율성을 높일 수 있다는 것을 입증하였습니다.



### A Data-Driven Real-Time Optimal Power Flow Algorithm Using Local Feedback (https://arxiv.org/abs/2502.15306)
- **What's New**: 이 논문에서는 지역 정보를 기반으로 분산 에너지 자원(DER)을 활용하여 실시간 최적 전력 흐름(Optimal Power Flow, OPF)을 해결하는 데이터 기반 알고리즘을 제안합니다. 기존 방법들의 한계점을 보완하기 위해, 저자는 지역 피드백을 사용하여 시간 변동성이 있는 OPF 솔루션을 지속적으로 추적할 수 있는 업데이트 전략을 세웠습니다. 이로 인해 실시간 최적화와 제어가 동시에 이루어질 수 있는 새로운 접근 방식이 제시되었습니다.

- **Technical Details**: 본 알고리즘은 학습 가능한 함수(learnable function)를 설계하여 입력으로 지역 피드백을 받아들이고, OPF 문제를 함수의 파라미터를 최적화하는 형태로 변환합니다. 또한, 비선형 전력 흐름 모델의 복잡한 기울기 계산을 우회할 수 있는 기울기 없는 접근법을 개발했습니다. 알고리즘의 안정성을 보장하는 충분조건을 제시하고, 깊은 신경망(deep neural network, DNN)을 사용한 파라미터화와 통계적 재구성을 통해 문제를 해결하도록 설계되었습니다.

- **Performance Highlights**: IEEE 37-bus 테스트 피더에서의 수치 결과에 따르면, 제안된 방법은 기존 벤치마크 방법들에 비해 시간 변동성이 있는 OPF 솔루션을 더 높은 정확도로 보다 빠르게 추적할 수 있음을 보여줍니다. 또한, 알고리즘은 전압 안전 한계를 준수하는 동시에, OPF 목표의 경제성을 유지하는 데 성공적인 성과를 자아냈습니다. 이러한 성능 개선은 알고리즘의 확장성과 실제 전력망 시스템에의 적용 가능성을 높이는 기반을 만듭니다.



### Comparative Analysis of Black Hole Mass Estimation in Type-2 AGNs: Classical vs. Quantum Machine Learning and Deep Learning Approaches (https://arxiv.org/abs/2502.15297)
Comments:
          29 pages, 12 Figures, 6 Tables

- **What's New**: 이 연구는 Type-2 AGN의 블랙홀 질량 추정에 대한 다양한 고전 및 양자 머신러닝 알고리즘을 비교합니다. 이 논문은 고전 알고리즘, 예를 들어 Linear Regression, XGBoost Regression, LSTM 등과 함께 Hybrid Quantum Neural Networks 및 Variational Quantum Regressor와 같은 양자 알고리즘을 사용하여 새로운 방법을 제시합니다. 특히, 블랙홀 질량을 추정하는 과정에서 양자 알고리즘이 고전 알고리즘에 비해 성능이 다소 떨어짐을 발견했습니다.

- **Technical Details**: Type-2 AGNs의 블랙홀 질량 추정에는 여러 고전 방법론과 양자 머신러닝(QML) 방법론이 비교되었습니다. 연구에서 사용된 고전 알고리즘으로는 LSTM, XGBoost, Random Forest 등이 있으며, 양자 알고리즘으로는 Estimator-QNN, Q-LSTM, HoneyQNN 등이 포함되었습니다. 결과적으로, LSTM이 정확도 99.77%로 가장 우수한 성능을 보였고, 양자 알고리즘 중에서는 Estimator-QNN이 MSE 0.0124 및 정확도 99.75%로 가장 뛰어난 성능을 발휘했습니다.

- **Performance Highlights**: 성능 평가 결과, 고전 알고리즘은 R², MAE, MSE, RMSE 모든 지표에서 양자 모델보다 우수한 성능을 보였습니다. LSTM 모델이 가장 높은 정확도를 기록한 반면, Estimator-QNN 또한 매우 경쟁력 있는 성능을 발휘하여 양자 컴퓨팅의 잠재력을 보여주었습니다. 저자는 양자 머신러닝이 향후 천체 물리학 데이터 분석에 혁신적인 전환을 가져올 것으로 기대하고 있습니다.



### Steganographic Embeddings as an Effective Data Augmentation (https://arxiv.org/abs/2502.15245)
Comments:
          10 pages, 4 figures. For associated code and experiments, see this http URL this https URL

- **What's New**: 이번 연구에서는 이미지 스테가노그래피(Image Steganography)를 데이터 증강(data augmentation) 전략으로 사용하여 딥 뉴럴 네트워크(deep neural networks)의 훈련 효율성을 높일 수 있음을 보여주고 있습니다. 특히 Least Significant Bit (LSB) 기법을 활용하여 숨겨진 정보를 이미지에 포함시키고, 이를 통해 데이터의 다양성을 확장하는 동시에 훈련 시간이 증가하지 않도록 합니다. 이 접근 방식은 이미지 분류와 같은 하류 컴퓨터 비전 작업에서 성능을 향상시키는 새로운 방향을 제시합니다.

- **Technical Details**: LSB Steganography를 통해 한 이미지를 다른 이미지에 삽입하는 과정이 이 연구의 핵심입니다. 이 방법은 비트의 수가 고정된 입력 데이터의 크기를 유지하면서 두 개의 이미지 정보로서의 유용성을 극대화합니다. 훈련 데이터 준비 시, 이미지를 배치(batch)로 처리하고 특정 확률(p)에 따라 두 이미지를 선택하여 결합하여 새로운 복합 이미지를 생성합니다.

- **Performance Highlights**: 이번 연구에서 제안하는 스테가노그래피 기반 데이터 증강의 효율성은 전통적인 방법에 비해 훈련 효율성을 크게 개선할 수 있음을 입증하고 있습니다. 또한 이 기법을 통해 색상 변경(color altering) 증강을 추가 비용 없이 제공받으며, 이러한 증강의 수학적 동등성을 통해 이 방법이 기존의 데이터 증강 방법에 비해 뛰어난 장점을 갖고 있음을 강조합니다.



### FormalSpecCpp: A Dataset of C++ Formal Specifications created using LLMs (https://arxiv.org/abs/2502.15217)
Comments:
          Accepted at the 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR)

- **What's New**: FormalSpecCpp는 C++ 프로그램의 formal specification을 검사하기 위한 표준화된 벤치마크가 부족한 문제를 해결하기 위해 설계된 데이터셋입니다. 이는 잘 정의된 precondition과 postcondition을 가진 C++ 프로그램의 최초의 종합적인 모음으로, 연구자와 개발자가 specification inference tool을 벤치마킹하고 생성된 specification의 정확성을 테스트하는 데 사용할 수 있습니다. 이 데이터셋의 공개는 프로그램 검증, specification inference, AI 지원 소프트웨어 개발의 연구를 진전시키는 목표를 가지고 있습니다.

- **Technical Details**: FormalSpecCpp 데이터셋은 verified Dafny 프로그램을 기반으로 하여 OpenAI의 GPT-4-turbo 모델을 사용하여 C++로 번역됩니다. 이 과정은 formal specification을 유지하면서 Dafny 프로그램을 번역하는 구조화된 접근 방식을 채택합니다. 또한, prompt engineering 기법을 활용하여 정확한 타입 매핑, assertion 변환 및 안전 제약 조건 처리를 통해 데이터셋의 품질을 향상시킵니다.

- **Performance Highlights**: 본 연구에서 변환된 105개의 C++ 파일은 모두 최초 시도에서 성공적으로 컴파일되었으며, 전체 과정은 약 27분만에 완료되었습니다. 변환 비용은 총 $2.07로, 파일당 평균 $0.02가 소요되었습니다. 이와 함께 관련 테스트 케이스를 생성하는 데 추가 $1.31이 들었고, 약 15분이 소요되었습니다.



### Tensor Product Neural Networks for Functional ANOVA Mod (https://arxiv.org/abs/2502.15215)
Comments:
          45 pages

- **What's New**: 본 논문은 고차원 기능을 저차원 함수의 합인 컴포넌트로 분해하는 기능적 ANOVA 모델을 기반으로 하여 독창적인 해석 가능 모델을 제안합니다. 이 모델은 유일한 기능적 ANOVA 분해(unique functional ANOVA decomposition)를 보장하여 각 컴포넌트를 안정적으로 추정할 수 있습니다. 제안된 모델의 이름은 ANOVA-NODE로, Neural Oblivious Decision Ensembles (NODE)의 기능적 ANOVA 모델에 대한 수정판입니다. 이론적으로 ANOVA-NODE가 평활 함수(smooth function)를 잘 근사할 수 있음을 증명합니다.

- **Technical Details**: ANOVA-NODE는 기존의 신경망 모델들이 가지는 불확정성(unidentifiability) 문제를 해결하여 각 구성 요소(component)가 식별 가능(identifiable)하게 만들어집니다. 이는 표준 확률적 경량 경량 최적화(stochastic gradient descent) 알고리즘으로 학습될 수 있기 때문에 큰 데이터셋에 적용 가능성을 높입니다. 텐서 곱 기저 확장(tensor product basis expansion) 기법을 바탕으로 각 기저 함수를 특수 설계된 신경망으로 대체하여 각 컴포넌트가 식별 가능하고 이상치(outlier)에 강하게 만듭니다. 최종적으로 ANOVA-TPNN을 제안하여 기능적 ANOVA 모델에서 각 컴포넌트를 추정합니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 분석한 결과, ANOVA-TPNN이 NAM, NBM, NODE-GAM, XGB 등과 같은 기준 모델들에 비해 더 정확하고 안정적인 컴포넌트 추정 및 해석을 제공함을 입증합니다. 기존 모델들이 가지는 예측 정확도를 잃지 않고도 각각의 컴포넌트를 안정적이고 정확하게 추정할 수 있습니다. 이로 인해 모형 해석 가능성(interpretable AI)이 향상되어 다양한 응용 분야에서의 활용 가능성이 높아집니다.



### CoT-ICL Lab: A Petri Dish for Studying Chain-of-Thought Learning from In-Context Demonstrations (https://arxiv.org/abs/2502.15132)
Comments:
          22 pages, 27 figures, 3 tables

- **What's New**: CoT-ICL Lab는 언어 모델에서 체인-오브-스Thought(Chain-of-Thought, CoT)와 인-컨텍스트 학습(In-Context Learning, ICL)을 연구하기 위한 새로운 프레임워크이자 방법론입니다. 이 연구는 토큰화된 합성 데이터 세트를 생성하고, 인-컨텍스트 예제의 복잡성에 대한 세밀한 제어를 제공합니다. 기존의 연구와는 달리, CoT-ICL Lab는 입력과 체인 토큰을 이산 토큰 공간에서 처리하여 자연어 처리와 밀접하게 연결됩니다.

- **Technical Details**: CoT-ICL Lab는 인과 구조(causal structure)와 토큰 처리 함수(token processing functions)를 분리하여 복잡한 문제를 처리할 수 있는 유연성을 제공합니다. 이 프레임워크는 방향 비순환 그래프(Directed Acyclic Graph, DAG)를 사용하여 체인 생성을 조정하고, 다양한 수준의 토큰 변환을 위해 다층 퍼셉트론(Multi-Layer Perceptron, MLP)을 활용합니다. 또한, 연구자들이 복잡한 문제를 더 잘 이해할 수 있도록 다양한 구성 요소를 조정할 수 있도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, CoT가 있는 인-컨텍스트 학습은 모델 성능의 전환을 가속화하는 데 기여함을 보여주었습니다. 특히, 모델의 깊이가 적은 예제에서 CoT의 활용에 중요한 것으로 나타났고, 더 많은 예제가 얕은 모델이 깊은 모델의 성능을 따라가는 데 도움이 되었습니다. 이를 통해 CoT-ICL Lab가 언어 모델에서 ICL과 CoT에 대한 이론적 및 경험적 통찰력을 제공하는 단순하면서도 강력한 테스트베드로 작용함을 강조했습니다.



### Optimal and Provable Calibration in High-Dimensional Binary Classification: Angular Calibration and Platt Scaling (https://arxiv.org/abs/2502.15131)
- **What's New**: 논문에서는 고차원 문제에서 리니어 이진 분류기의 보정(calibration) 문제를 다루고 있습니다. 기존의 알고리즘들은 주로 전통적인 비대칭 이론이나 유한 샘플 학습 이론에 치중하였으나, 이 연구는 고차원에서 효과적인 보정 방법을 제시합니다. 특히, 기하학적인 각도(geometry angle) 기반의 보정 접근법을 통해 정말 신뢰할 수 있는 예측기를 구축하는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 고차원 특성(feature)에서 샘플 수와 특성 수가 비례적으로 다루어지는 상황에서 보정된 예측기를 제안합니다. 이러한 예측기는 추정된 선형 가중치(estimated linear weight) 대 실제 가중치(true weight) 사이의 각도를 바탕으로 interpolation을 사용하여 구성합니다. 또한, 제안한 예측기가 Bregman 최적(Bregman-optimal)임을 입증하며, 이는 진짜 레이블 생성 확률에 대한 Bregman 발산을 최소화합니다.

- **Performance Highlights**: 제안된 보정 방법은 고차원 설정에서 확실하게 효과적인 특성을 보여줍니다. 이러한 성과는 Platt scaling과 같은 기존 방법과 비교했을 때, 명확하게 보정된 솔루션으로 수렴하는 조건을 제시합니다. 실험에서도 기존 Gaussian 가정에 대한 강인성을 보여주며, 이는 현대 통계 및 기계 학습 방법론에 기여하는 바가 큽니다.



### Curvature Corrected Nonnegative Manifold Data Factorization (https://arxiv.org/abs/2502.15124)
- **What's New**: 본 연구에서는 곡률 보정(nonnegative manifold data factorization, CC-NMDF)을 제안하여 매니폴드 값 데이터에서 해석 가능한 요소를 추출하는 방법을 개발하였습니다. 이는 기존의 비음수 행렬 분해(nonnegative matrix factorization, NMF)에 기초한 기하학적 접근 방식으로, 다양한 과학적 데이터에서 발생하는 매니폴드 데이터 분석에 적절하게 적용될 수 있습니다.

- **Technical Details**: CC-NMDF는 대칭 리만 매니폴드(symmetric Riemannian manifold)에서 수집된 데이터의 해석 가능한 분해를 공식화하고, 빠른 접선 공간(tangent space) 기반 알고리즘에 곡률 보정을 적용하는 방식으로 구현됩니다. 이 알고리즘은 매니폴드 값 데이터에 효과적으로 적용할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: CC-NMDF 방법은 실제 확산 텐서 자기공명영상(diffusion tensor magnetic resonance imaging) 데이터에 적용되었습니다. 이 연구는 매니폴드 값 데이터의 분석에 유용한 저차원 근사(low rank approximation)의 새로운 길을 제시하며, 기존의 NMF 기법을 매니폴드 환경에 맞게 발전시킨 사례를 보여줍니다.



### Do we really need the Rademacher complexities? (https://arxiv.org/abs/2502.15118)
- **What's New**: 이번 연구에서는 Rademacher 복잡도의 필요성에 대한 기존 신념에 도전합니다. 저자들은 최소한의 가정 하에 샘플 복잡성이 Rademacher 복잡도가 아닌 한계 Gaussian 프로세스의 행동에 의해 결정된다는 것을 증명했습니다. 이 결과는 일반 볼록 학습 문제에서의 첫 번째 보편성 결과로, 동일한 $L_2$ 구조를 가진 모든 학습 문제는 같은 샘플 복잡성을 공유합니다.

- **Technical Details**: 연구에서는 새로운 학습 절차에 기반하여, 실제 값 랜덤 변수에 대한 최적 평균 추정 기법과 Talagrand의 일반 체이닝 방법을 결합해 성능을 분석합니다. 이를 통해 한계 Gaussian 프로세스의 행동을 이용하여 최적 학습 절차의 오차를 결정할 수 있게 됩니다. 저자들은 특히 무거운 꼬리 분포를 포함한 볼록 함수 집합에서의 학습 문제에 대한 새로운 관점을 제시합니다.

- **Performance Highlights**: 이 연구는 전통적인 Rademacher 복잡도가 주장하는 것과 달리, 한계 Gaussian 프로세스가 학습 문제의 성능 지표와 관련이 있음을 강조합니다. 결과적으로 샘플 복잡성이 Rademacher 복잡도보다 훨씬 작은 경우가 자주 발생하며, 이러한 발견은 머신러닝 연구의 기초를 흔드는 중요한 통찰을 제공합니다.



### Variational phylogenetic inference with products over bipartitions (https://arxiv.org/abs/2502.15110)
Comments:
          20 pages, 5 figures

- **What's New**: 이번 연구에서는 ultrametric phylogenetic trees를 위한 새로운 변별적 베이지안 접근법을 제안합니다. VIPR(Variational phylogenetic Inference with PRoducts over bipartitions)라는 방법을 통해 모든 나무 공간(tree space)에서 추론을 수행하며, Markov chain Monte Carlo(MCMC) 서브루틴을 필요로 하지 않습니다. 또한, 이 방법은 분지 길이(branch lengths)에 시간 제약을 포함시켜 정확한 분류를 가능하게 합니다.

- **Technical Details**: 우리는 단일 연결(cluster) 클러스터링을 기반으로 한 변별 가족을 제시하고, 그에 따른 밀도(density)를 나무에 대해 국소적인 게재를 통해 도출합니다. VIPR는 coalescent 이론을 바탕으로 하여 변별 정보를 통해 ultrametric 나무에 대한 변별 추론을 자연스럽게 수행합니다. VIPR는 distance matrix에 대해 변별 분포를 매개변수화하여 이를 이용해 differentiable variational density를 도출합니다.

- **Performance Highlights**: COVID-19 데이터와 여러 기준 유전체 데이터셋에서 실험을 통해 VIPR의 정확도가 기존 최첨단 기술과 유사함을 입증하였습니다. 특히, VIPR는 기존 방법보다 훨씬 적은 gradient 평가로도 경쟁력 있는 결과를 달성할 수 있음을 보여줍니다. 이를 통해, VIPR는 ultrametric trees의 추론 분야에서 중요한 발전을 이룬 것으로 평가됩니다.



### Social Genome: Grounded Social Reasoning Abilities of Multimodal Models (https://arxiv.org/abs/2502.15109)
Comments:
          Under Review, 22 pages

- **What's New**: 이번 논문에서는 멀티모달 모델의 기초적이고 구체적인 사회적 추론 능력을 평가하기 위한 새로운 벤치마크인 Social Genome을 소개합니다. Social Genome은 272개의 상호작용 비디오와 1,486개의 인간 주석이 포함된 추론 경로(reasoning traces)를 포함하며, 이들 경로는 5,777개의 추론 단계로 구성되어 있습니다. 이 벤치마크는 시각적 단서, 언어적 단서, 음성 단서 및 외부 지식을 참조하여 사회적 정보를 해석하는 데 필요한 능력을 평가합니다.

- **Technical Details**: Social Genome은 비디오의 시각적, 언어적, 음성적 단서와 외부 지식을 참고하여 사회적 추론을 수행하는 알고리즘 개발을 위한 중요한 기초 자료입니다. 각 추론 단계는 정보의 참조 방식에 따라 태그가 지정되어 있으며, 이는 11,000개 이상의 개체, 5,000개 이상의 멀티모달 단서 및 2,900개 이상의 외부 지식을 포함합니다. 이 논문은 모델이 생성한 사회적 추론 경로의 의미론적 및 구조적 측면을 평가할 수 있는 메트릭(metrics)을 정의합니다.

- **Performance Highlights**: 실험 결과, 최신 모델들이 사회적 추론에서의 차이를 여전히 보이고 있으며, 제로샷(zero-shot) 및 인컨텍스트 학습(in-context learning) 설정에서 성능이 부족하다는 것을 알 수 있었습니다. Social Genome 벤치마크를 통해 연구자들은 멀티모달 모델의 기초적인 사회적 추론 능력을 향상시키기 위한 갭과 기회를 식별할 수 있었습니다. 이러한 발견들은 AI 시스템의 사회적 상호작용 처리 능력을 향상시키는 데 기여할 것으로 예상됩니다.



### LUME: LLM Unlearning with Multitask Evaluations (https://arxiv.org/abs/2502.15097)
- **What's New**: 최근의 데이터 보호 규제 및 개인 정보 요구에 따라 LLMs(대형 언어 모델)의 유효한 unlearning 알고리즘에 대한 필요성이 커지고 있습니다. 이 논문에서는 LUME(LLM Unlearning with Multitask Evaluations)라는 새로운 멀티태스크 unlearning 벤치마크를 개발하여 창작물과 민감한 정보를 포함하는 데이터 세트를 다룹니다. 다음으로, 1B 및 7B 매개변수를 가진 두 개의 세부 튜닝된 모델을 공개하여 unlearning 알고리즘의 효율성을 평가합니다.

- **Technical Details**: LUME 벤치마크에는 세 가지 작업이 포함됩니다: (1) 합성 창작 단편 소설, (2) 개인 정보가 포함된 합성 전기, (3) 공개 전기. unlearning의 효과성을 측정하기 위해 기억화, 개인 정보 유출, 모델 유틸리티 테스트와 같은 정교한 메트릭을 사용합니다. 이 방법은 프라이버시와 유틸리티를 모두 고려하면서 특정 정보 세트를 기억하지 못하도록 하는 것을 목표로 합니다.

- **Performance Highlights**: 여러 최신 unlearning 알고리즘을 평가한 결과, 민감한 정보를 효과적으로 제거하더라도 모델의 유틸리티가 크게 저하되는 경향이 있음을 발견했습니다. 제안된 벤치마크는 공개적으로 사용 가능하며, 관련 데이터 및 정보에 대한 유효한 unlearning 솔루션 개발에 기여할 것으로 기대됩니다. LUME는 LLM의 유용성을 테스트하기 위한 중요한 자원으로 자리 잡을 것으로 보입니다.



### Forecasting Local Ionospheric Parameters Using Transformers (https://arxiv.org/abs/2502.15093)
Comments:
          47 pages, 42 figures

- **What's New**: 이번 논문에서는 transformer 기반의 신경망을 이용하여 주요 전리층 매개변수를 예측하는 혁신적인 방법론을 제시합니다. 이 모델은 특정 지리적 위치에 대한 F2층의 피크 플라즈마 주파수(foF2), 피크 밀도 높이(hmF2), 총 전자 밀도(TEC)의 정확한 예측과 불확실성 정량화를 제공합니다. Local Ionospheric Forecast Transformer(LIFT)라 불리는 이 방법은 외부 변수와 기후에서의 순진한 예측값을 결합하여 24시간 예측을 생성하는 방식으로 학습됩니다.

- **Technical Details**: 이 모델은 128개의 이온소너 스테이션에서 얻은 데이터를 기반으로 학습되며, 이 데이터는 ARTIST5 알고리즘을 통해 가공됩니다. 예측 창은 72시간 길이의 데이터 윈도우를 사용하여 다음 24시간의 예측을 생성합니다. 각 지리적 위치에서의 데이터를 분석하여, 모델이 새로운 지리적 지역에서도 일반화될 수 있는지를 평가하였습니다.

- **Performance Highlights**: LIFT 모델은 기존의 예측 모델인 국제 기준 전리층(IRI)와 비교하여 뛰어난 성능을 나타났으며, 새로운 지리적 위치에 대한 예측에서도 효과적으로 작동하였습니다. 본 논문의 결과는 향후 데이터 기반의 전리층 모델에서 transformer 모델이 중요한 역할을 할 수 있음을 암시합니다.



### Modifying Final Splits of Classification Tree for Fine-tuning Subpopulation Target in Policy Making (https://arxiv.org/abs/2502.15072)
- **What's New**: 이 논문에서 제안된 두 가지 방법, Penalized Final Split (PFS)와 Maximizing Distance Final Split (MDFS)는 기존의 Classification and Regression Trees (CART) 방법론의 한계를 극복하기 위한 것으로, 주어진 이진 사건의 잠재적 확률을 더욱 정확하게 분류하기 위해 설계되었습니다. PFS는 표준 CART 분할 기준 함수에 조정 가능한 패널티를 추가하여 잘못 분류될 위험을 줄이고, MDFS는 노드 평균과 임계치 간의 가중치 합의 거리를 극대화하여 최적의 분할을 식별합니다.

- **Technical Details**: 기술적으로 PFS는 CART의 분할 기준을 수정하여 잘못된 분류를 최소화하고자 하며, MDFS는 고유한 교차 잠재(probability) 가정을 통해 최적 분할을 지향합니다. MDFS의 이론적 결과는 제로(asymptotic) 위험을 가지고 있으며, 여러 시뮬레이션 연구를 통해 두 방법 모두 기존의 CART와 KD-CART를 뛰어넘는 성과를 보였습니다. 이 방법들은 정책적 맥락에서 사용되는 Latent Probability Classification (LPC) 문제에 맞춰 개발되었습니다.

- **Performance Highlights**: 그 결과, PFS와 MDFS는 잘못된 분류 오류 면에서 기존 방법들보다 뛰어난 성능을 보여주었으며, 정책 입안자들이 목표 집단에 대해 더욱 깊은 통찰을 제공하였습니다. 실증 평가를 통해 두 새로운 방법이 주어진 기준에 맞게 특정 집단을 효과적으로 타겟팅 할 수 있게 도와주는 것으로 나타났습니다. 이러한 개선된 성능은 정책 자원 분배의 효율성을 높이며 궁극적으로 보다 정교한 의사결정을 가능하게 합니다.



### An Interpretable Machine Learning Approach to Understanding the Relationships between Solar Flares and Source Active Regions (https://arxiv.org/abs/2502.15066)
Comments:
          16 pages, 9 figures

- **What's New**: 이번 연구는 2011년부터 2021년까지의 관측 데이터를 활용하여 태양 플레어를 예측하는 머신러닝 모델을 개발했습니다. 특히 랜덤 포레스트(Random Forest) 알고리즘을 통해 태양 활동 영역(active regions, ARs)의 물리적 특성과 태양 플레어 간의 관계를 분석했습니다. 이를 통해 태양 플레어 발생을 예측하기 위한 이진 분류 작업을 진행하였으며, AR 유형과 클래스 간의 상관관계를 밝히는 것을 목표로 했습니다.

- **Technical Details**: 연구에서는 AR에 대한 데이터 수집 및 랜덤 포레스트 모델을 구축하여 태양 플레어에 영향을 미치는 주요 물리적 특성을 식별했습니다. AR_Type_Today는 긍정적 분류에서 90% 이상의 사례에서 중요한 특성으로 확인되었으며, Hale_Class_Yesterday는 가장 영향력이 적은 특성으로 나타났습니다. 또한 NoS_Difference 특성은 모델의 결정 과정에 있어 글로벌 및 로컬 해석에서 상당한 영향을 미쳤습니다.

- **Performance Highlights**: 모델의 성과 측정 지표로는 0.81의 재현율(recall), 0.82의 정밀도(precision), 0.74의 정확도(accuracy), 0.82의 F1 점수를 기록했습니다. 이는 태양 플레어 발생 예측에 있어 랜덤 포레스트 모델이 유의미한 성능을 보임을 나타냅니다. 이러한 결과는 태양 플레어의 조기 경고 시스템 개발에 기여할 수 있을 것입니다.



### Low degree conjecture implies sharp computational thresholds in stochastic block mod (https://arxiv.org/abs/2502.15024)
Comments:
          33 pages

- **What's New**: 이 논문은 최근 공식화된 저차 추측(low-degree conjecture)의 적용을 연구하며, 대칭 확률 블록 모델(symmetric stochastic block model)의 맥락에서 커뮤니티 레이블의 약회복(weak recovery) 문제를 다룹니다. 저자들은 저차 추측이 성립할 경우, Kesten-Stigum(KS) 경계를 아래로 커뮤니티 레이블을 약하게 회복하는 다항식 시간(polynomial-time) 알고리즘이 존재하지 않음을 수립합니다. 본 연구는 KS 임계점에서 다항식 시간 알고리즘의 회복률의 급격한 전환에 대한 첫 번째 엄격한 증거를 제공합니다.

- **Technical Details**: 이 연구는 스타카스틱 블록 모델(stochastic block model)에서 커뮤니티 구조를 회복하는 문제를 다루며, KS 임계점 이하에서 다항식 시간 알고리즘의 성능 한계를 제시합니다. 그래프 분할(graph splitting) 및 교차 검증(cross-validation) 기법을 활용하여 회복 알고리즘의 일반성을 배제하는 방법론을 사용하였습니다. 저자는 기존의 연구와 비교하여 강력한 하한(lower bound)을 제시하며, 저차 추측의 강력한 버전이 충족될 경우 블록 수가 발산하는 경우에도 이 하한의 유효성을 강조합니다.

- **Performance Highlights**: 이 연구는 KS 임계점 이상에서 다항식 시간 알고리즘이 진정한 커뮤니티와 높은 확률로 상수 상관(u)관계를 달성할 수 있음을 보여줍니다. 그러나 KS 임계점 이하에서는 확률적으로 진정한 커뮤니티와 유의미한 상관관계를 달성하는 다항식 시간 추정량이 존재하지 않음을 입증하였습니다. 이는 기존 연구와는 대조적으로, 통계적-계산적 격차(computational-to-statistical gap) 증거를 제공합니다.



### Obliviate: Efficient Unmemorization for Protecting Intellectual Property in Large Language Models (https://arxiv.org/abs/2502.15010)
- **What's New**: 최근 AI 기업들과 콘텐츠 제작자 사이의 저작권 협약으로 인해 언어 모델의 저작권 콘텐츠 재현 능력에 대한 정밀한 제어 필요성이 강조되었습니다. 우리는 Obliviate라는 새로운 포스트 트레이닝(post-training) 기법을 제안하며, 이 방법은 특정 텍스트의 정확한 복제를 선택적으로 방지하면서 의미론적 이해를 유지합니다. Obliviate는 기억된 시퀀스에서 특정 토큰을 선택하고 모델의 확률 분포를 수정하여 정확한 복제를 방지하는 방식으로 작동합니다.

- **Technical Details**: Obliviate는 사전 훈련된 언어 모델에서 특정 시퀀스를 비기억하도록 선택적으로 수정하는 기술입니다. 이 기술은 Kullback-Leibler (KL) 발산 손실을 사용하여 목표 토큰이 재현될 확률을 줄이면서 유창성과 일관성을 유지하도록 조정합니다. 비목표 토큰에 대해서도 KL 발산을 적용하여 상위 k 토큰 분포의 일관성을 유지하여 모델의 전반적인 성능을 보존합니다.

- **Performance Highlights**: Obliviate는 여러 대형 언어 모델(LLaMA-3.1 8B, Qwen-2.5-7B 등)에 대해 평가된 결과, 100배의 저작권 콘텐츠의 Verbatim memorization 감소를 보여주면서도 표준 벤치마크에서 성능 저하가 1% 이내로 유지되었습니다. 이 결과는 Obliviate가 선택적으로 목표 콘텐츠를 비기억하면서 모델 성능을 유지하는 데 효과적임을 강조합니다.



### Few-shot Species Range Estimation (https://arxiv.org/abs/2502.14977)
- **What's New**: 본 논문에서 제안하는 FS-SINR은 제한된 데이터로부터 종의 범위를 추정하는 새로운 방법을 소개합니다. 기존의 모델들은 많은 관측 데이터에 의존했으나, FS-SINR은 적은 수의 관측자료로도 신뢰할 수 있는 예측을 가능하게 합니다. 이 모델은 또한 훈련 세트에 포함되지 않은 종에 대한 예측도 가능하여 대화식 탐색 및 모델링이 가능합니다.

- **Technical Details**: FS-SINR은 Transformer 기반 모델로, 관측된 지리적 위치 세트를 입력으로 받아 종의 범위를 추정합니다. 이 과정에서 추가적인 메타데이터(예: 텍스트 요약 또는 이미지)를 유연하게 통합하여 예측 품질을 더욱 향상시킬 수 있습니다. 이 방법은 IUCN과 S&T 벤치마크 데이터셋에서 검증되었으며, 우수한 성과를 보였습니다.

- **Performance Highlights**: FS-SINR은 적은 수의 샘플에서 최신 성능을 발휘함을 보여주며, 이전의 알고리즘에 비해 월등한 결과를 달성하였습니다. 특히, 373,000종의 대부분이 적은 관측 데이터를 가진 경우에도 높은 정확도로 범위를 예측할 수 있는 가능성을 제시합니다. 다양한 관측 데이터를 통합하여 더 나은 예측을 이끌어내는 것이 FS-SINR의 큰 강점입니다.



### Symmetric observations without symmetric causal explanations (https://arxiv.org/abs/2502.14950)
Comments:
          8 pages, 4 figures, RevTeX 4.2. The computational appendix is available at this https URL

- **What's New**: 이번 논문은 관찰된 상관관계로부터 인과 모델을 유추하는 과정의 어려움을 탐구하며, 관찰의 대칭성이 기본적인 인과 모델의 대칭성과 반드시 일치하지 않는다는 점을 명확하게 밝혔다. 이를 통해 기존의 이론적 틀에 대한 새로운 통찰을 제시하며, 고전 물리학을 바탕으로 한 다양한 시스템들이 어떤 한계들에 부딪히는지를 논의한다.

- **Technical Details**: 논문에서는 세 개의 독립적인 고전적 무작위 소스로부터 유래된 삼원 이항 사건에 대한 확률 분포를 통해 본 주제를 설명한다. 특히, 여기서는 마르코프 조건(Markov condition)과 보른 법칙(Born's rule)과 같은 확률 이론의 원리를 적용하여 관찰을 기술하는 방법을 설명하고, 집합적인 확률 이론의 관점에서 이 문제를 접근한다.

- **Performance Highlights**: 결과적으로, 이 논문은 클래스 및 양자 시스템 모두에 관련한 여러 가지 배포 모델에 대해 실제적인 제약 조건을 설정하고, 특정 대칭성이 유지되는 가운데 이러한 모델을 생성하는 것이 불가능하다는 강력한 입증을 진행했다. 이는 단순한 이론을 넘어서, 초광자 시스템 등에서의 응용 가능성을 높이는 데 기여할 수 있는 내용을 담고 있다.



### KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding (https://arxiv.org/abs/2502.14949)
Comments:
          17 pages, 5 figures, ACL 2025

- **What's New**: 이 논문은 KITAB-Bench라는 포괄적인 아랍어 OCR 벤치마크를 소개합니다. 아랍어 OCR의 평가 시스템에서의 공백을 메우기 위해 8,809개의 샘플을 수집하여 9개의 주요 도메인과 36개의 하위 도메인으로 구성하였습니다. 이 벤치마크는 다양한 문서 유형을 포함하여, 손글씨 텍스트, 구조화된 표, 그리고 비즈니스 인텔리전스를 위한 21가지 차트 유형에 대한 특화된 내용을 제공합니다.

- **Technical Details**: KITAB-Bench는 레이아웃 탐지(text blocks, tables, figures), 다중 형식 인식(printed/handwritten text, charts, diagrams), 그리고 구조화된 출력 생성(HTML tables, DataFrame charts, markdown)을 평가하는 시스템을 채택합니다. 이 방법론은 OCR 성능을 정밀하게 평가할 수 있는 프레임워크를 제공하며, 챠트 추출(CharTeX)과 도표 추출(CODM) 평가 지표를 포함하여 다양한 문서 이해 과제를 다룹니다.

- **Performance Highlights**: 최신 비전-언어 모델(GPT-4, Gemini, Qwen)이 전통적인 OCR 방법(EasyOCR, PaddleOCR, Surya)에 비해 평균 60% 더 높은 Character Error Rate(CER)에서 우수성을 보였습니다. 그러나 현재 아랍어 OCR 모델의 한계로는 PDF에서 Markdown으로의 변환에서 가장 우수한 모델인 Gemini-2.0-Flash가 단 65%의 정확도만을 달성했습니다. 이러한 결과는 아랍어 텍스트 인식을 위한 복잡한 글꼴, 숫자 인식 오류, 단어 신장 및 표 구조 감지의 문제를 강조합니다.



### Reward-Guided Iterative Refinement in Diffusion Models at Test-Time with Applications to Protein and DNA Design (https://arxiv.org/abs/2502.14944)
Comments:
          Under review. If you have any suggestions/missing references, please let us know

- **What's New**: 이 논문에서는 확산 모델(diffusion models)과 관련해 새로운 테스트 시간 보상 최적화(test-time reward optimization) 프레임워크를 제안합니다. 이 프레임워크는 진화 알고리즘(evolutionary algorithms)에서 영감을 받았으며, 각 반복(iteration)에서 노이즈 처리와 보상 기반 디노이징(reward-guided denoising)을 포함하는 단계적 정제(iterative refinement)의 두 가지 단계를 활용합니다. 이를 통해 보상 최적화 과정에서 발생할 수 있는 오류를 점진적으로 수정할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 각 반복에서 파슬 노이징(partial noising)과 보상 기반 디노이징(reward-guided denoising)으로 구성되며, 이를 통해 복잡한 보상 함수(reward functions)를 최적화할 수 있습니다. 특히 생물학적 서열이나 분자 설계에서 자주 발생하는 하드 제약 조건(hard constraints)을 처리하는 데 효과적입니다. 이 프레임워크의 이론적 보장을 제공하며, 보상 함수와 사전 훈련(pre-trained) 분포의 통합을 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘이 단백질(protein) 및 DNA 설계에서 기존 방법보다 뛰어난 성능을 보인다는 것을 확인했습니다. 이 연구는 보상 기반 생성(generation) 과정에서의 반복적 접근 방식을 통해 출력 결과를 점진적으로 개선하는 방법을 제시하며, 이는 다양한 생물학적 응용 분야에 큰 의미를 가집니다. 코드도 제공되어 연구 결과의 재현이 용이합니다.



### Fast and Accurate Blind Flexible Docking (https://arxiv.org/abs/2502.14934)
Comments:
          25 pages, Accepted by ICLR 2025

- **What's New**: FABFlex는 단백질의 유연성과 결합 부위가 알려지지 않은 상황에서, 빠르고 정확한 다중 작업 학습 모델로 설계되었습니다. 이 모델은 3개 모듈로 구성되어 있으며, 결합 포켓 예측, 리간드 도킹, 포켓 도킹을 통해 실질적인 '블라인드 유연 도킹'(blind flexible docking) 시나리오의 도전 과제를 해결합니다. FABFlex는 리간드와 포켓 도킹 모듈 간의 지속적인 구조 개선 가능성을 제공하는 반복 업데이트 메커니즘을 포함하여 통합적이고 일관된 프레임워크를 구성합니다.

- **Technical Details**: FABFlex는 E(3)-등가 그래프 신경망 레이어인 'FABind 레이어'를 활용해 각 모듈을 설계하였으며, 이는 리간드-단백질 이종 그래프를 처리할 수 있도록 조정되었습니다. 포켓 예측 모듈에서 시작하여, 예측된 포켓 정보를 활용하여 각각의 리간드와 포켓의 홀로 구조를 예측합니다. 각 모듈은 상호 작용하며 기능적으로 통합되어 있으며, 동시에 리간드와 포켓의 홀로 구조를 단일 작업으로 예측하게 됩니다.

- **Performance Highlights**: FABFlex는 PDBBind 공개 벤치마크에서 기존의 다양한 도킹 방법들과 비교하여 리간드 RMSD가 2Å 이하인 비율을 40.59%로 증가시키고, 포켓 RMSD를 1.10Å까지 낮추는 성과를 보였습니다. 특히, FABFlex는 최근의 최첨단 유연 도킹 방법인 DynamicBind보다 약 208배 빠른 계산 속도를 자랑하여 실질적으로 시간과 자원을 절약할 수 있습니다.



### The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Tex (https://arxiv.org/abs/2502.14921)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)로 생성된 합성 데이터(synthetic data)로부터 훈련 샘플(training samples)의 정보가 얼마나 유출될 수 있는지를 분석합니다. 합성 데이터 생성 파이프라인에서 정보 흐름(information flow)의 미세한 부분을 간과할 경우 사생활(privacy)에 대한 잘못된 인식을 초래할 수 있습니다. 저자들은 특히 적대자가 미세 조정된 모델(fine-tuned model)에 접근하지 못하고 합성 데이터만 가지고 있는 경우에 대한 멤버십 추론 공격(membership inference attacks, MIAs)을 설계했습니다.

- **Technical Details**: 이 논문에서는 합성 데이터에 대한 MIAs가 무작위 추측(random guess)보다 훨씬 더 높은 성능을 보인다는 사실을 발견했습니다. 이는 합성 데이터가 훈련 데이터(training data)에 대한 정보를 유출하는 것을 의미합니다. 또한, 모델 기반 MIA에 대한 취약성을 극대화하기 위해 제작된 카나리(canaries)는 합성 데이터만 공개될 경우 프라이버시 감사를 위한 최적의 방법이 아님을 보였습니다.

- **Performance Highlights**: 저자들은 자동 회귀(autoregressive) 모델의 메커니즘을 활용하여 합성 데이터에서 탐지 가능한 흔적을 남기는 in-distribution prefix와 high-perplexity suffix를 가진 카나리를 설계했습니다. 이러한 접근 방식은 데이터 기반 MIAs의 효과를 향상시키고 LLMs가 생성한 합성 데이터의 프라이버시 위험을 보다 잘 평가할 수 있도록 합니다.



### Optimizing Gene-Based Testing for Antibiotic Resistance Prediction (https://arxiv.org/abs/2502.14919)
Comments:
          Accepted to AAAI-25 AISI

- **What's New**: 이번 연구는 GenoARM이라는 새로운 프레임워크를 도입하여 강화 학습(reinforcement learning)과 transformer 기반 모델을 통합하여 PCR 유전자 검사 최적화를 통해 항생제 내성 예측을 개선합니다. 또한, 메타데이터를 활용하여 정확도를 높이는 방법을 제안합니다. 이전 연구들과 비교하여, GenoARM은 메타데이터 사용 시 보다 정교한 성과를 보여주어 많은 임상 환경에서의 진단 도구 최적화에 큰 발전을 가져올 수 있습니다.

- **Technical Details**: GenoARM은 두 가지 주요 모듈로 구성되어 있으며, 하나는 항생제 내성(prediction of antibiotic resistance)을 예측하는 모델이고, 다른 하나는 최적의 PCR 검사 유전자 집합을 선택하는 정책 네트워크(policy network)입니다. PCR 검사의 최적 유전자 집합을 찾는 문제를 정의하고, 다양한 실험을 통해 강화 학습 기반으로 모델을 훈련시키는 방식을 구현했습니다. 이 과정에서 알고리즘은 방대한 NCBI 데이터 세트를 사용하여 여러 병원체를 아우르는 테스트의 효율성을 점검했습니다.

- **Performance Highlights**: GenoARM은 메타데이터가 도입되었을 때 이전 방법들보다 우수한 성능을 발휘하는 것으로 나타났습니다. 특히, 전체 모델은 데이터의 부족함과 고차원 상태-행동 공간(high-dimensional state-action space)에서도 뛰어난 예측력을 보여줘서 클리니컬한 적용 가능성을 강조하고 있습니다. 연구 결과는 모든 평가된 방법들이 메타데이터가 없을 때도 강력하고 신뢰할 수 있는 성능을 보여주었음을 입증하였습니다.



### What Is a Good Caption? A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Coverage of MLLMs (https://arxiv.org/abs/2502.14914)
Comments:
          Work in progress

- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전으로 기존 비주얼 캡셔닝 기준이 뒤처지게 되었습니다. 기존 기준들은 주로 짧은 설명과 오래된 메트릭으로 한정되어 있었습니다. 이 문제를 해결하기 위해 CV-CapBench라는 종합적인 비주얼 캡션 기준을 제안하며, 6개 관점과 13개 차원에 걸쳐 캡션 품질을 체계적으로 평가합니다.

- **Technical Details**: CV-CapBench는 각 차원에 대해 정확도(accuracy), 재현율(recall), 적중률(hit rate) 메트릭을 도입하여 결과의 정확성과 coverage을 독특하게 평가합니다. 특히, 동적 및 지식 집약적 차원에서 MLLMs의 성능 격차가 뚜렷하게 나타났습니다. 이 구성이 정적(static) 차원 9개와 동적(dynamic) 차원 4개로 나뉘어지며, 직관적으로 시각적 캡션의 포괄성을 강화하고 있습니다.

- **Performance Highlights**: CV-CapBench를 활용한 실험 결과, 여러 주요 MLLMs가 여전히 특정 차원에서 성과를 내기 어려운 것으로 나타났습니다. 이러한 발견은 향후 비주얼 캡셔닝 개선을 위한 실질적인 통찰을 제공합니다. 논문에서 제안된 코드는 공개될 예정으로, 심도 있는 연구가 기대됩니다.



### Universal Semantic Embeddings of Chemical Elements for Enhanced Materials Inference and Discovery (https://arxiv.org/abs/2502.14912)
Comments:
          5 figures

- **What's New**: 이번 논문에서는 재료 추론(material inference) 및 발견(discovery)을 촉진하기 위한 보편적인 의미 임베딩(semantic embeddings) 프레임워크를 제시합니다. 이 프레임워크는 합금(alloy) 관련 과학 논문 129만 개의 초록을 기반으로 학습된 BERT 기반 자연어 처리(NLP) 모델인 ElementBERT를 활용하여 합금에 특화된 잠재 지식(latent knowledge)과 맥락적 관계를 포착합니다.

- **Technical Details**: ElementBERT는 전통적인 경험적(descriptors) 대신 사용할 수 있는 강력한 원소 기술(descriptors)로, 여러 하위 작업(downstream tasks)에서 현저한 성능 향상을 보여줍니다. 이 연구에서는 기계적 성질(mechanical properties) 예측, 상 구조(classifying phase structures) 분류, 베이지안 최적화(Bayesian optimization)를 통한 재료 성질 최적화 등의 작업에서 성능을 평가하였습니다.

- **Performance Highlights**: 타이타늄 합금(titanium alloys), 고엔트로피 합금(high-entropy alloys), 형태 기억 합금(shape memory alloys)에 대한 적용을 통해 예측 정확성(prediction accuracy)에서 최대 23% 향상이 이루어졌습니다. 또한, ElementBERT는 일반용 BERT 변형보다 특화된 합금 지식을 인코딩하여 더 나은 성능을 발휘함을 보여 줍니다.



### KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models (https://arxiv.org/abs/2502.14908)
- **What's New**: 이번 연구에서는 Vision Language Models (VLMs)의 멀티모달 환경에서의 지식 갈등에 대한 영향을 조사하기 위해 \

- **Technical Details**: \

- **Performance Highlights**: \



### Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherenc (https://arxiv.org/abs/2502.14905)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 생성 과정에서 엄격한 스키마 준수를 강제하는 문제를 다룹니다. DeepSeek R1 강화 학습 프레임워크를 기반으로, 15억 매개변수 모델의 구조적 추론 기술을 훈련시키기 위한 새로운 파이프라인을 제안합니다. 특히, 비구조화된 데이터에서 구조화된 데이터로의 변환을 위한 2만 개의 샘플 데이터셋을 활용해 모델의 핵심 추론 능력을 구축하였으며, 이후 1만 개의 추론 샘플 데이터셋에서 감독 학습을 통해 스키마 준수를 더욱 정교하게 다듬었습니다.

- **Technical Details**: 이 접근법은 합성(문서 생성) 데이터셋 구축과 사용자 정의 보상 함수를 결합하여 Group Relative Policy Optimization(GRPO) 아래서 훈련됩니다. 모델은 강력한 추론 능력을 함양하기 위해 비구조화된 데이터와 구조화된 데이터 모두를 생성하는 작업을 수행하며, 사용자 정의 보상 메커니즘을 통해 출력의 스키마 준수를 직접 평가합니다. 즉, 모델 훈련은 모든 관련 기준에서 높은 점수를 획득하는 출력을 만들어내는 방향으로 이루어집니다.

- **Performance Highlights**: 우리의 ThinkJSON 접근법은 DeepSeek R1(671B), Qwen-1.5B와 Qwen-7B의 축약 버전, Gemini 2.0 Flash(70B)와 비교하여 실용적인 응용에서의 효과를 보여주었습니다. 훈련 범위가 상대적으로 modest했음에도 불구하고, GRPO 훈련을 위해 8xH100 GPU 클러스터에서 약 20시간, SFT를 위해 1xA100 GPU에서 3시간이 소요되는 동안, 우리의 모델은 스키마 일관성을 보장하는 데 강력한 성능을 발휘합니다. 이 연구는 스키마가 제약되는 텍스트 생성에 대한 자원 효율적인 프레임워크의 실용적 유용성을 강조합니다.



### Towards an automated workflow in materials science for combining multi-modal simulative and experimental information using data mining and large language models (https://arxiv.org/abs/2502.14904)
- **What's New**: 이 논문에서는 자료 과학(materials science) 시뮬레이션과 실험 데이터를 쉽게 접근 가능하고 기계가 읽을 수 있는 구조로 변환하는 자동화된 작업 흐름을 소개합니다. 최근 오픈 사이언스(open science)의 발전이 데이터 접근성을 더욱 높이고 있으나, 많은 정보가 과학 문서에 암호화돼 있어 원하는 문헌 및 물질 특성을 찾는 데 어려움이 있었습니다. 이 연구는 자연어 처리(natural language processing)와 비전 트랜스포머(vision transformer) 모델을 활용하여 문헌에 담긴 정보를 기계가 읽을 수 있는 데이터 구조로 변환하는 방법을 보여줍니다.

- **Technical Details**: 자동화된 작업 흐름은 텍스트, 그림, 표, 방정식 및 메타데이터를 포함한 다양한 자료를 기계가 읽을 수 있는 형식으로 변환하는 데 중점을 두고 있습니다. 이 과정에서 기계 가독성 데이터베이스는 지역 데이터, 예를 들어 비공식적 또는 개인적 물질 데이터를 추가하여 지식의 합성(knowledge synthesis)을 이끌어낼 수 있습니다. 연구는 특히 면심 입방 결정(face-centered cubic single crystals)의 미세구조 분석 연구 분야를 통해 다중 모달 입력 데이터(multi-modal input data)에 대한 정보 검색, 근접 맥락 감지 및 물질 특성 추출의 가속화를 보여줍니다.

- **Performance Highlights**: 마지막으로, Retrieval-Augmented Generation (RAG) 기반의 대규모 언어 모델(large language model, LLM)은 빠르고 효율적인 질문 답변(chat bot) 기능을 지원함으로써 자동화된 작업의 유용성을 강조합니다. 이 시스템은 연구자들이 필요한 정보를 더 빠르게 찾을 수 있도록 도와줄 뿐만 아니라, 기계 학습 및 인공지능(AI) 기술을 통해 데이터의 활용 범위를 넓히는 데 기여합니다.



### Reading the unreadable: Creating a dataset of 19th century English newspapers using image-to-text language models (https://arxiv.org/abs/2502.14901)
- **What's New**: 이 논문에서는 19세기 언론의 디지털 아카이브 접근성을 높이기 위해 Pixtral 12B라는 사전 훈련된 이미지-텍스트 변환 모델을 활용한 Optical Character Recognition (OCR) 기술을 소개합니다. 전통적인 방법과 달리, Pixtral의 OCR 성능은 4개의 다른 OCR 접근 방식과 비교하여 가장 낮은 문자 오류율 1%를 기록했습니다. 이러한 성과는 NCSE v2.0 데이터셋을 생성하게 되었으며, 이는 향후 역사적 및 사회학적 연구에 기여할 것입니다.

- **Technical Details**: NCSE(v2.0)는 19세기 영국 신문 및 정기간행물의 84,000 페이지를 포함한 1.4백만 개 항목과 3.21억 개의 단어로 구성됩니다. 해당 데이터셋은 네 가지 유형의 텍스트 및 열일곱 가지 주제로 분류되어 높은 품질의 OCR과 기사를 식별하기 쉽게 만듭니다. Pixtral 12B 모델은 새로운 언어로의 적용이 용이한 형태로 고안되었습니다.

- **Performance Highlights**: NCSE v2.0 데이터셋은 주제 유사성 분석, 가독성 평가 및 사건 추적과 같은 다양한 활용 사례를 통해 현대 독자들이 19세기 저널리즘을 쉽게 접근할 수 있도록 합니다. OCR 접근 방식을 통해 21세기 독자들은 이제 오스카 와일드가 언급한 불리한 저널리즘 기준을 이해하고 재발견할 수 있는 기회를 갖게 되었습니다.



### UPCMR: A Universal Prompt-guided Model for Random Sampling Cardiac MRI Reconstruction (https://arxiv.org/abs/2502.14899)
Comments:
          Accepted paper for STACOM 2024

- **What's New**: 이 논문에서는 심장 자기공명영상(CMR) 복원을 위한 보편적인 모델인 UPCMR을 소개합니다. UPCMR은 두 가지 유형의 학습 가능한 프롬프트를 통합하여 다양한 언더샘플링(k-space에서의 불완전 샘플링) 방식에 적응할 수 있도록 설계되었습니다. 이 접근 방식은 UNet 구조와 결합된 각 블록에서 효과적인 품질 향상을 보여줍니다.

- **Technical Details**: UPCMR 모델은 다중 코일 언더샘플링 k-space 측정값으로부터 복잡한 값을 가지는 MR 영상 시퀀스를 재구성하는 것을 목표로 합니다. 이 모델은 언더샘플링 특화 프롬프트와 공간 특화 프롬프트를 제공합니다. 또한 모델은 k-space 궤적 및 가속 요인을 사전 정보로 활용하여 학습 가능한 프롬프트 풀이 효율적으로 설계되었습니다.

- **Performance Highlights**: UPCMR 모델은 CMRxRecon2024 챌린지 데이터셋을 사용하여 모든 무작위 샘플링 시나리오에서 재구성된 이미지 품질이 크게 향상되었습니다. 기존의 전통적인 방법들과 비교하여 강력한 적응 가능성을 보여주며, 이는 다양한 샘플링 모드에서 효과적인 훈련 전략 덕분입니다. 결과적으로 UPCMR은 심장 MRI 재구성 작업에 있어 뛰어난 성능을 입증하였습니다.



### Revisiting Financial Sentiment Analysis: A Language Model Approach (https://arxiv.org/abs/2502.14897)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 연구는 전통적인 금융 감정 분석(Financial Sentiment Analysis, FSA)의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 기존의 인간 주석에 의존하는 감정 레이블 대신, 시장에서 직접 유도된 레이블을 사용하여 텍스트 신호와 시장 동학 간의 관계를 명확히 합니다. 연구 결과는 여러 시장 상황에서 금융 트윗의 효과적인 예측이 가능하다는 것을 보여주며, 이러한 패러다임 전환은 금융 의사 결정에 있어 언어 모델의 미개척 가능성을 강조합니다.

- **Technical Details**: 연구에서는 짧은 시간 가격 경향을 바탕으로 트윗 레이블을 지정하는 시장 기반 레이블링 방법을 제안합니다. 이 방법은 Triple Barrier Labeling (TBL) 접근법을 활용하여 상한, 하한, 그리고 시간적 제약을 통해 시장 반응을 그대로 반영합니다. 또한, CryptoBERT 및 FinBERT와 같은 도메인 특화된 언어 모델을 기반으로 하여, 시장 정보를 효과적으로 통합하는 프롬프트 튜닝 기술로 모델의 예측 정확성을 향상시킵니다.

- **Performance Highlights**: 본 연구에서 제안한 언어 모델은 전통적인 감정 기반 기준선에 비해 단기 추세 예측 정확도가 최대 11% 향상되었으며, 특정 비트코인 관련 뉴스 사건에 대해 89.6%의 높은 정확도를 기록했습니다. 일일 트윗 예측을 통한 거래 신호 생성이 전통적인 융합 모델에 비해 우수한 결과를 보여주었고, 여러 시장 환경에서 샤프 비율이 각각 5.07 및 3.73에 달했습니다. 이러한 결과는 언어 모델이 효과적인 단기 시장 예측 도구로 사용될 수 있음을 증명합니다.



### FOCUS on Contamination: A Geospatial Deep Learning Framework with a Noise-Aware Loss for Surface Water PFAS Prediction (https://arxiv.org/abs/2502.14894)
- **What's New**: 이 연구는 PFAS 오염 예측을 위한 지리공간 딥러닝 프레임워크인 FOCUS를 소개합니다. FOCUS는 label noise-aware loss function을 활용하여 방대한 지역에서 PFAS 오염을 예측할 수 있도록 설계되어 있습니다. 이 모델은 수문학적 흐름 데이터, 토지 이용 정보, 그리고 이미 알려진 PFAS 원천과의 거리 데이터를 통합하여 예측 정확성을 개선합니다.

- **Technical Details**: FOCUS는 PFAS 오염의 세분화를 위해 래스터 데이터를 사용하는 지리공간 딥러닝 모델을 적용합니다. 이 프레임워크는 PFAS 확산에 대한 도메인 전문 지식을 활용하여 데이터 격차를 자동으로 보완하고, 이러한 가정을 신뢰도에 따라 가중치를 두어 왼쪽으로 캐싱하는 점이 특징입니다. 모델의 성능은 다양한 AI 모델 및 기존의 과학적 방법들과 비교하여 검증되었습니다.

- **Performance Highlights**: FOCUS의 평가 결과는 대규모 PFAS 모니터링을 위한 스케일 가능성, 정확성 및 강건성을 강조합니다. 이 연구는 환경 연구에 적극적으로 참여하는 비영리 단체와 수질 모델 전문 학자와 함께 진행되어, 최신 전문성과 실제 환경 문제를 반영합니다. FOCUS는 PFAS 오염의 복잡성을 해결하기 위해 AI와 지리공간 모델링의 최근 발전을 통합하여 새로운 발견을 가능하게 합니다.



### NOTA: Multimodal Music Notation Understanding for Visual Large Language Mod (https://arxiv.org/abs/2502.14893)
- **What's New**: 이번 논문에서는 NOTA라는 이름의 대규모 종합 멀티모달 음악 표기법 데이터셋을 처음으로 제안합니다. 이 데이터셋은 1,019,237개의 레코드로 구성되어 있으며, 세 가지 전 세계 지역에서 수집되었습니다. 데이터셋은 음악 정보 추출, 크로스 모달 정렬 테스트, 음악 표기법 분석의 세 가지 주요 작업을 다룹니다.

- **Technical Details**: NOTA 데이터셋은 ABC 표기법을 사용하여 음악 점수를 표현합니다. 데이터셋은 헤더와 본체로 구성되어 있으며, 헤더에는 참조 번호와 제목, 박자 기호, 기본 음표 길이 등이 포함됩니다. 본체는 주로 음표와 마침표를 포함하여 음악의 구조적 요소를 표현합니다.

- **Performance Highlights**: NotaGPT-7B 모델은 17개의 주요 멀티모달 대형 언어 모델과의 실험에서 음악 이해에 있어 두드러진 개선을 보였습니다. 기존 최고 성능 모델인 Gemini는 33.34%의 음악 정보 추출률을 달성한 반면, 우리 모델은 67.84%의 성과를 기록했습니다. 이는 멀티모달 음악 데이터셋의 중요성과 NOTA 데이터셋의 효과성을 보여줍니다.



### SEM-CLIP: Precise Few-Shot Learning for Nanoscale Defect Detection in Scanning Electron Microscope Imag (https://arxiv.org/abs/2502.14884)
Comments:
          Published in ACM/IEEE International Conference on Computer-Aided Design (ICCAD), 2024

- **What's New**: 본 논문에서는 SEM 이미지 결함 분류 및 세분화를 위한 새로운 few-shot learning 접근 방식인 SEM-CLIP을 제안합니다. 이 방법은 기존의 데이터 및 라벨 요구 사항을 최소화하여 반도체 산업의 라벨링 문제를 해결하고 스마트 제조를 촉진할 수 있는 기반을 마련합니다. 특히 SEM-CLIP은 CLIP 모델을 사용자 정의하여 결함 영역에 집중하고 배경의 복잡성을 최소화한 기능 추출 방법을 도입했습니다.

- **Technical Details**: SEM-CLIP 모델은 Contrastive Language-Image Pretraining(CLIP) 모델을 조정하여 결함 영역에 더욱 집중하고 세분화 정확도를 향상시키기 위해 V𝑉Vitalic_V-V𝑉Vitalic_V 어텐션 블록을 추가하여 복잡한 배경 방해 요소를 최소화합니다. 또한, 전문 지식을 활용한 텍스트 프롬프트를 사전 정보로 활용하여 명확한 결함 분석을 지원합니다. 이를 통해 충분한 주석 데이터 없이도 모델이 복잡한 시각적 개념을 이해하도록 돕습니다.

- **Performance Highlights**: SEM-CLIP은 여러 few-shot 설정에서 테스트된 결과, 다른 방법들보다 우수한 클래시피케이션(iAUROC, pAUROC, F1-mmax) 성능을 보여줍니다. 특히, SEM-CLIP은 최근의 SOTA 방법인 PromptAD보다 각각 2.0%, 1.3%, 21.1% 향상된 결과를 기록했습니다. 이러한 결과는 반도체 산업에서 결함 감지의 효율과 정확성을 크게 높일 것으로 기대됩니다.



### Applications of Random Matrix Theory in Machine Learning and Brain Mapping (https://arxiv.org/abs/2502.14878)
- **What's New**: 본 논문은 Random Matrix Theory (RMT)와 머신러닝 (Machine Learning, ML)을 뇌 매핑에 적용하는 새로운 방법을 탐구하고 있습니다. RMT를 통해 뇌의 기능적인 영역들 간의 상관관계를 분석하여, 신경 질환과 장애를 발견하는 데 중요한 진전을 이루고 있습니다. 이 연구는 머신러닝의 정확성을 높이고, 인공지능의 응용 가능성을 확장하려는 목표를 가지고 있습니다.

- **Technical Details**: RMT는 선형대수학과 확률 이론의 조합으로, 다차원 난수 데이터를 2차원이나 3차원으로 투영하여 분석할 수 있게 해줍니다. 기존 알고리즘을 통해 voxel 신호의 강도를 타임 인터벌에 따라 시뮬레이션함으로써, 뇌 네트워크의 동적 변화를 탐구합니다. 특히, Marchenko-Pastur 법칙을 적용하여 잡음이 섞인 경우에도 유의미한 eigenvalue 분포를 유지하는 점이 강조됩니다.

- **Performance Highlights**: 이 연구 결과는 RMT가 잡음에 대해 높은 테스트-재테스트 신뢰성을 가진다는 것을 증명하며, 뇌의 기능적 영역 간의 강한 상관관계가 존재함을 보여줍니다. 또한, 머신러닝을 통한 뇌 스캔의 미래 해석이 가능해짐으로써, 이미지 분석의 정확성을 높이고 병원 자원을 더 효율적으로 재분배할 수 있는 가능성을 열어줍니다.



### Exploring Quantum Control Landscape and Solution Space Complexity through Dimensionality Reduction & Optimization Algorithms (https://arxiv.org/abs/2502.11905)
- **What's New**: 이 연구에서는 단일 두 수준 양자 시스템(qubit)의 양자 제어 경관(Quantum Control Landscape, QCL)에 대한 다양한 제어 전략을 분석하였습니다. 새로운 분석 방법인 Principal Component Analysis (PCA)를 사용하여 고차원 제어 매개변수의 QCL을 시각화하고 이해할 수 있는 기법을 제시합니다. 특히, PCA와 같은 차원 축소 기법이 양자 제어의 복잡성을 이해하는 데 중요한 역할을 할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구의 주요 목표는 최적의 목표 상태를 달성하기 위한 최적의 제어 매개변수를 찾는 것이며, 이를 위해 Genetic Algorithms (GA), Stochastic Gradient Descent (SGD) 및 Q-learning (QL) 등의 전통적인 최적화 알고리즘을 평가합니다. 또한, Cluster Density Index (CDI)를 사용하여 해결 공간의 복잡성을 분석하며, 이는 알고리즘이 고충실도의 영역을 생성하는지를 나타냅니다. 이 과정에서 즉각적인 보상 함수 설계의 중요성을 강조하고, DQN 및 PPO를 사용할 때 즉각적인 보상이 시스템 성능에 미치는 긍정적인 영향을 보여줍니다.

- **Performance Highlights**: 실험 결과, GA가 SGD에 비해 우수한 성능을 보이는 것으로 나타났으며, Q-learning은 DQN 및 Proximal Policy Optimization (PPO)보다 좋은 성과를 보였습니다. 또한, 최적화 알고리즘의 성능을 비교하며, 특히 짧은 시간 단계의 양자 제어 문제에서 효과적으로 작동하는 RL 네트워크 설계의 중요성을 강조합니다. 이 연구에서는 양자 제어 전략의 효율성을 높이기 위한 여러 알고리즘 탐색의 성과를 분석하여, 최적 해결을 찾는 데 있어 어느 알고리즘이 가장 적합한지에 대한 인사이트를 제공합니다.



### Automating Customer Needs Analysis: A Comparative Study of Large Language Models in the Travel Industry (https://arxiv.org/abs/2404.17975)
- **What's New**: 이번 연구에서는 자연어 처리(Natural Language Processing)의 발전 속에서 대형 언어 모델(Large Language Models, LLMs)이 여행 고객 요구사항 추출에 효과적임을 조명했습니다. 특히, TripAdvisor 게시물에서 고객의 요구를 파악하는 데 있어서 여러 모델의 성능을 비교 분석하였습니다. 이 과정에서 오픈소스와 독점 모델(GPT-4, Gemini 등)을 포함하여 다양한 모델들이 사용되었습니다.

- **Technical Details**: 연구에서는 BERTScore, ROUGE, BLEU와 같은 다양한 평가 지표를 통해 각 모델의 성능을 객관적으로 분석하였습니다. 가장 주목할 만한 결과는 Mistral 7B 오픈소스 모델이 대형 폐쇄형 모델에 비해 경쟁력 있는 성능을 보이고, 경제성과 커스터마이즈 가능성을 동시에 제공한다는 점입니다. 이는 고객 요구 분석 작업에 적합한 모델 선택 시 고려해야 할 요소를 강조합니다.

- **Performance Highlights**: 연구 결과, 오픈소스 LLM 모델들이 대형 상용 모델들과 유사한 성능을 나타내면서도 비용 효율적이고 맞춤형 솔루션을 제공함을 확인했습니다. 이는 여행 산업에서 고객 경험을 향상시키고 운영 효율성을 높이기 위한 고급 NLP 기술 활용을 원하는 기업들에게 중요한 통찰력을 제공합니다.



### Temporal Misalignment in ANN-SNN Conversion and Its Mitigation via Probabilistic Spiking Neurons (https://arxiv.org/abs/2502.14487)
- **What's New**: 이 논문에서는 Spiking Neural Networks (SNNs)의 ANN-SNN 변환 과정에서 발생하는 'temporal misalignment'라는 현상을 밝혀냈습니다. 이 현상은 SNN 레이어 전반에서 랜덤 스파이크 재배치가 성능 향상을 초래한다는 것을 보여줍니다. 이를 기반으로, 두 단계 확률적 스파이킹 뉴런(Two-Phase Probabilistic spiking neurons)을 도입하여 변환 과정을 더욱 개선하고 있습니다.

- **Technical Details**: 기존의 Integrate-and-Fire (IF) 스파이킹 뉴런 모델을 기반으로 하여 ANN과 SNN 간의 변환 과정을 수식으로 설명합니다. 이 과정은 변환의 정확성을 높이며, 헤비스타이드 함수와 같은 수학적 요소를 통해 SNN의 내부 동역학을 모델링합니다. 논문에서는 ANN에서 SNN으로 가중치와 편향을 전송하는 방법을 자세히 다루고 있으며, 이 과정의 효용성을 강조합니다.

- **Performance Highlights**: CIFAR-10/100, CIFAR10-DVS, ImageNet 데이터셋을 통한 실험에서 제안한 방법이 최신 SOTA 변환 및 다른 훈련 방법들보다 우수한 정확도를 달성했습니다. 다양한 아키텍처를 통한 포괄적 실험을 통해 이 방법의 효과가 입증되었으며, 이로써 에너지 효율적인 AI 시스템 구축의 가능성을 제시합니다.



