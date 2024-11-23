New uploads on arXiv(cs.CL)

### Predictive Insights into LGBTQ+ Minority Stress: A Transductive Exploration of Social Media Discours (https://arxiv.org/abs/2411.13534)
Comments:
          This paper is accepted in 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA)

- **What's New**: 이번 연구에서는 LGBTQ+ 커뮤니티에서의 소셜 미디어에서 표현된 소수 집단 스트레스(minority stress)를 식별하기 위한 새로운 하이브리드 모델이 소개되었습니다. 이 모델은 Graph Neural Networks (GNN)와 Bidirectional Encoder Representations from Transformers (BERT)를 통합하여, 기존의 자연어 처리(NLP) 방법의 한계를 극복하는 데 목적을 두고 있습니다. 특히, 성 정체성을 가진 개인들이 언어적으로 복잡한 방식으로 소수 집단 스트레스를 표현하는 방식을 더 정확하게 포착할 수 있도록 설계되었습니다.

- **Technical Details**: 연구진은 LGBTQ+ MiSSoM+라는 소셜 미디어 데이터 세트를 사용하여 모델을 시험했으며, 이 데이터 세트는 5,789개의 인간이 주석을 단 Reddit 게시물로 구성되어 있습니다. 하이브리드 모델인 RoBERTa-GCN은 0.86의 정확도 및 F1 점수를 달성하여, 다른 기준 모델들을 능가하는 성과를 보였습니다. 이 모델은 대규모 원시 데이터에 대한 사전 훈련을 통해 숨겨진 언어적 뉘앙스를 추출하고, 레이블이 지정된 훈련 데이터와 레이블이 없는 테스트 데이터의 표현을 동시에 개발할 수 있도록 합니다.

- **Performance Highlights**: RoBERTa-GCN 모델은 LGBTQ+ 소수 집단 스트레스 예측 분야에서 다른 기존 모델들보다 우수한 성능을 보였습니다. 이러한 예측 성능 향상은 LGBTQ+ 커뮤니티의 건강 문제를 해결하기 위한 디지털 건강 개입의 기회를 제공할 수 있습니다. 건강 불균형을 줄이기 위한 예측력이 향상됨에 따라, 연구 결과는 소수 집단 스트레스 관련 건강 개입을 설계하는 데 있어 중요한 통찰력을 제공합니다.



### Advancing Complex Medical Communication in Arabic with Sporo AraSum: Surpassing Existing Large Language Models (https://arxiv.org/abs/2411.13518)
Comments:
          arXiv admin note: text overlap with arXiv:2411.06713

- **What's New**: 이 연구는 다국어 기능에 대한 의료 분야의 증가하는 수요를 반영하여 아랍어에 맞춤화된 Sporo AraSum 모델을 평가합니다. 기존의 JAIS 모델과 비교하여, Sporo AraSum은 의료 문서화에서 아랍어의 복잡한 특성을 처리하는 데 적합한 성능을 보입니다. 이 연구는 특히 임상 문서화를 위한 AI 모델의 필요성을 강조하고 있습니다.

- **Technical Details**: 연구는 합성 데이터셋(synthetic datasets)과 의료 관련 지표인 PDQI-9를 수정하여 Sporo AraSum과 JAIS의 성능을 평가합니다. 모델의 성능은 환자와 의사의 상호작용을 요약하는 데 중점을 두며, 정확성(accuracy), 포괄성(comprehensiveness), 임상 유용성(clinical utility), 언어-문화적 역량(linguistic-cultural competence) 등의 요소가 포함됩니다. Sporo AraSum의 아키텍처는 아랍어의 미세한 언어적 뉘앙스를 고려하여 정확하고 문화적으로 민감한 문서화를 가능하게 합니다.

- **Performance Highlights**: Sporo AraSum은 AI 중심의 정량적 지표와 모든 정성적 측면에서 JAIS 모델을 크게 능가하는 성과를 보였습니다. 특히, Sporo AraSum은 AI 망상(hallucination)과 같은 리스크를 줄이며 아랍어가 필요한 의료 환경에서 더 적합한 모델로 나타났습니다. 향후 연구는 실제 데이터를 포함하여 이러한 결과를 검증하고 의료 시스템에의 통합 가능성을 탐색할 것을 제안합니다.



### Disentangling Memory and Reasoning Ability in Large Language Models (https://arxiv.org/abs/2411.13504)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 추론 과정에서 지식 검색과 추론 단계를 명확하게 분리하는 새로운 패러다임을 제안합니다. 이 과정에서 메모리(knowledge recall)와 추론(reasoning)이라는 두 가지 명확한 행동으로 복잡한 추론을 분해하는 데 중점을 두었습니다. 또한, 새로운 제어 신호로 작동하는 특별한 토큰인 ⟨memory⟩와 ⟨reason⟩을 도입하여 모델의 성능과 해석 가능성을 향상시킵니다.

- **Technical Details**: 이 연구에서 제안한 방법은 새로운 훈련 데이터셋을 통해 진행됩니다. 강력한 LLM인 GPT-4o를 활용하여 다양한 질문-응답 작업에 대한 항목별 행동을 생성하고, 이 행동들을 각각 메모리나 추론으로 레이블을 붙입니다. 이후, 레이블이 붙은 데이터셋을 사용하여 맞춤형 LLM을 훈련시키며, 학습 과정을 통해 이 두 가지 제어 토큰이 모델이 관련 지식을 회상하고 그에 기반하여 추론을 수행하도록 유도합니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식은 학생 모델의 성능을 개선했으며, 특정 벤치마크에서는 78.6%와 78.0%의 정확도를 달성했습니다. 특히 TruthfulQA 데이터셋에서는 LLaMA-3.1-8B가 GPT-4o보다 높은 86.6% 정확도를 기록하였으며, 세 개의 벤치마크 데이터셋 평균에서는 1.9%의 성능 격차를 보였습니다. 이러한 성과에 덧붙여, 모델의 추론 과정을 파악할 수 있는 해체 정확도도 83.3%에 달해, 오류의 주요 원인을 식별하는 데 기여했습니다.



### Utilizing Large Language Models to Synthesize Product Desirability Datasets (https://arxiv.org/abs/2411.13485)
Comments:
          9 pages, 2 figures, 6 tables

- **What's New**: 이번 연구는 Product Desirability Toolkit (PDT) 테스트를 위한 합성 데이터셋 생성을 대규모 언어 모델(LLMs)로 수행하는 방법을 탐구합니다. 특히 비용 효율적인 gpt-4o-mini 모델을 사용하여 1000개의 제품 리뷰를 생성하는 세 가지 방법(Word+Review, Review+Word, Supply-Word)을 제안합니다. 본 연구는 합성 데이터셋이 실제 데이터가 제한된 상황에서 비용 효과적이고 확장 가능한 옵션이 될 수 있는지를 평가합니다.

- **Technical Details**: 합성 데이터셋을 생성할 때, 높은 감정 일치를 보여주었고 Pearson 상관계수는 0.93에서 0.97에 이릅니다. Supply-Word 방법은 가장 높은 텍스트 다양성과 PDT 용어의 범위를 보였지만, 데이터 생성 비용이 증가했습니다. 이는 LLM이 생성한 합성 데이터가 특히 적은 테스트 데이터가 필요할 때의 이점을 제공함을 보여줍니다.

- **Performance Highlights**: 연구 결과는 개발된 세 가지 합성 데이터 생성 방법이 높은 감정 일치를 달성하며, 다양한 텍스트 형태를 제공한다고 보고합니다. 그러나 모든 방법에서 약간의 긍정적 편향이 관찰되었으며, 이는 미래 연구에서 해결해야 할 필요성을 제기합니다. 합성 데이터의 생산 비용 분석 또한 성과 강조 점으로, 이점은 확장 가능성 및 비용 절감 효과가 있음이 확인되었습니다.



### PatentEdits: Framing Patent Novelty as Textual Entailmen (https://arxiv.org/abs/2411.13477)
- **What's New**: 본 연구에서는 창작권 확보를 위해 필요한 특허 수정 예측 문제를 새로운 학습 가능한 작업으로 설정하고, 이를 위한 PatentEdits 데이터셋을 소개합니다. 이 데이터셋은 105,000개의 수정 예시를 포함하고 있으며, 기존 문서 수정 예측 연구와 달리 주목할 만한 이전 문헌과의 관계를 고려합니다. 특히, 대형 언어 모델(LLM)을 활용하여 인용된 참조와 초안 문장 간의 텍스트 포함 관계를 평가하는 방식을 제안합니다.

- **Technical Details**: PatentEdits 데이터셋은 2007년부터 2014년까지의 미국 공인 발명 특허 105,000건으로 구성되어 있습니다. 데이터셋에 포함된 초안, 인용된 참조 및 최종 특허 텍스트를 정렬하여 어떤 문장이 Kept, Edited 또는 Deleted 되는지를 자동적으로 라벨링하는 알고리즘을 설계했습니다. 향상된 클레임 예측을 위해 LLM을 활용하여 인용된 문장을 긍정 예로, 초안 문장을 앵커로, 최종 문장을 부정 예로 설정하고, triplet loss로 파인튜닝합니다.

- **Performance Highlights**: 실험 결과, 인용된 참조를 포함시키고 초안 특허 클레임과 인용된 특허 클레임 간의 포함 관계에 집중할 경우, 특허 클레임의 새로움을 평가하는 데 효과적임을 보여주었습니다. 특히, BLEU-4 및 METEOR와 같은 다양한 유사도 평가 방법을 활용하여 초안 및 최종 문장 간의 문서 매칭 품질을 검증하였습니다. 이러한 접근 방식은 특허 출원자와 미국 특허 심사원이 필요한 수정을 예측하는 데 실질적인 기여를 할 수 있을 것으로 기대됩니다.



### When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training (https://arxiv.org/abs/2411.13476)
- **What's New**: 최근 논문에서는 대형 언어 모델(LLMs)의 긴 시퀀스를 처리하고 복잡한 작업을 수행하기 위해 컨텍스트 창 크기를 확장하는 필요성을 강조하고 있습니다. 이 모델은 Rotary Positional Embedding (RoPE)의 상대적인 위치 인코딩 특성 덕분에 긴 맥락 훈련에서 효과를 보여주지만, BFloat16 포맷 사용 시 수치적 문제를 일으킨다는 점을 지적합니다. 특히 긴 컨텍스트 시나리오에서 RoPE의 상대적인 위치 인코딩이 깨지는 문제가 발생합니다.

- **Technical Details**: BFloat16은 GPU 메모리 사용을 줄여주지만, 그 한계로 인해 RoPE의 상대적 위치 인코딩 속성이 손상됩니다. 논문에서는 AnchorAttention이라는 플러그 앤 플레이(plug-and-play) 주의(attention) 메커니즘을 제안하여, 첫 번째 토큰을 공유 앵커(anchor)로 취급하고 일관된 위치 ID를 부여합니다. 이는 문서 간의 불일치를 제거하고, 토큰 간의 엄청난 수치적 오류 누적을 방지하는 방식으로 구성됩니다.

- **Performance Highlights**: AnchorAttention을 사용하면 표준 전체 주의 메커니즘에 비해 긴 컨텍스트 성능이 크게 향상되며, 학습 시간도 50% 이상 단축됩니다. 실제 긴 컨텍스트 벤치마크에서 성능이 개선되었고, MMLU와 HellaSwag와 같은 일반 작업에서도 원래 모델의 능력을 유지하는 데 성공했습니다. 이 모든 결과가 AnchorAttention의 높은 효과성을 입증하고 있습니다.



### LIMBA: An Open-Source Framework for the Preservation and Valorization of Low-Resource Languages using Generative Models (https://arxiv.org/abs/2411.13453)
- **What's New**: 이번 백서에서는 소수 언어의 디지털 자원 부족 문제를 해결하기 위한 언어 모델 생성을 위한 프레임워크를 제안하고 있습니다. 특히, 고급 언어 모델과 대조적으로 데이터가 부족한 언어의 보존을 지원하기 위해 데이터 생성에 중점을 두고 있습니다. 사르디니아어와 같은 멸종 위기 언어를 사례 연구로 사용하여 프레임워크의 효과성을 입증하려고 합니다. 이 연구는 언어 다양성을 촉진하고 언어 표준화 및 재생을 지원하는 데 기여하고자 합니다.

- **Technical Details**: 소수 언어는 디지털 자원 확보가 어려워 고급 언어 처리 기술(advanced linguistic technologies)의 개발이 제한됩니다. 본 연구에서는 언어 모델을 구축하기 위해 새로운 언어 도구를 생성하기 위한 방법론을 제시하는데, 사르디니아어를 활용하여 효과성을 검증하고 있습니다. 또한, 데이터 수집을 촉진하기 위해 구글과 모질라와 같은 다양한 기관이 프로그램을 시작했음을 언급하고 있습니다.

- **Performance Highlights**: 프레임워크를 통해 소수 언어를 위한 언어 모델이 생성될 가능성이 높아집니다. 데이터가 부족한 언어에서도 새로운 데이터 생성을 지원하여 AI 모델의 효과적인 훈련을 도울 수 있을 것으로 기대됩니다. 이러한 접근법은 사회적 포용성을 높이고, 향후 소수 언어의 보존과 활성화에 긍정적인 영향을 미칠 것으로 예상됩니다.



### Unification of Balti and trans-border sister dialects in the essence of LLMs and AI Technology (https://arxiv.org/abs/2411.13409)
Comments:
          Accepted by IEEE conference ISCSLP 2024

- **What's New**: 이번 논문은 Balti 언어의 중요성과 그 방언들의 통합 노력을 다룹니다. 이 언어는 중국어-티베트어족(Sino-Tibetan) 소속으로, 인도, 중국, 파키스탄 등 여러 국가에서 다양한 방언이 존재합니다. 인공지능(AI) 기술의 발전에 따라, 방언 통합을 통한 공통성 이해의 필요성이 강조됩니다.

- **Technical Details**: Balti 언어의 분석 및 문서화는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 수행됩니다. 이 연구에서는 다양한 방언 간의 기본적인 어근(root), 어휘(lexica), 음운론적 관점에서의 공통점을 찾아내는 방법을 제시합니다. 이러한 접근 방식은 언어 보존에 기여할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: AI를 활용한 Balti 언어의 분류 및 표준화 노력은 여러 방언들의 문서화와 분석을 통해 이루어집니다. 글로벌화가 진행되는 현재, AI 기술은 언어 다양성 이해 및 방언 통합에 중요한 역할을 할 수 있습니다. 이 연구는 인공지능 기술이 어떻게 언어 보존에 기여할 수 있는지를 탐구합니다.



### Transformer-Based Contextualized Language Models Joint with Neural Networks for Natural Language Inference in Vietnames (https://arxiv.org/abs/2411.13407)
- **What's New**: 이번 연구는 베트남어에서 자연어 추론(Natural Language Inference, NLI)의 성능 향상을 추구하는 새로운 접근 방식을 제안합니다. 기존의 연구들은 영어 중심으로 이루어졌으며, 베트남어에 적합한 합동 모델(joint model)에 대한 연구가 부족했습니다. 연구팀은 다양한 유형의 맥락화된 언어 모델(Contextualized Language Model, CLM)과 신경망(Neural Networks)을 결합하여 성능을 극대화하고, 각 모델의 장단점을 평가하였습니다.

- **Technical Details**: 연구에서는 CLM을 사용하여 베트남어 NLI 데이터셋을 위한 맥락화된 표현을 생성하고, 신경망을 통해 분류 작업을 수행합니다. XLM-R 모델이 (355M) 가장 큰 규모로 실험에 사용되었으며, 이 결합 방식은 성능 면에서 PhoBERT, mBERT 및 XLM-R과 같은 기존의 강력한 언어 모델을 능가하였습니다. 연구의 결과로, 단일 베트남어 NLI 태스크에서 82.78%의 F1 점수를 기록했습니다.

- **Performance Highlights**: 이는 기존의 SOTA 모델과 비교했을 때 뚜렷한 성과를 나타냅니다. 예를 들어, CLM과 신경망을 통합하여 부여된 벡터의 성능이 비맥락화된 벡터에 비해 37.74% 향상되었습니다. 특히, 실험 결과 신경망과 맥락화된 모델의 결합이 베트남어에서 NLI 과제를 처리하는 데 매우 효과적임을 입증했습니다.



### On the Way to LLM Personalization: Learning to Remember User Conversations (https://arxiv.org/abs/2411.13405)
Comments:
          16 pages, 6 tables, 3 figures

- **What's New**: 이 연구 논문에서는 대화의 개인화를 위한 새로운 접근 방식을 제안합니다. 기존의 개인화 방법들은 주로 사용자의 스타일이나 소소한 정보에 중점을 두었지만, 본 연구는 이전 대화에서의 지식을 LLM에 삽입하는 방법을 탐구하고 있습니다. 연구팀은 PLUM이라는 파이프라인을 제안하여 사용자 대화 데이터를 증강하고 이를 통해 파라미터 효율성이 높은 방식으로 LLM을 미세 조정하는 데 중점을 두고 있습니다.

- **Technical Details**: PLUM은 두 단계의 파이프라인으로 구성되어 있습니다. 첫 번째 단계에서는 사용자 대화 데이터를 증강하며, 두 번째 단계는 사용자 대화의 지식을 LLM에 주입하기 위한 파라미터 효율적인 미세 조정을 진행합니다. 이 과정에서 손실 함수로는 가중치가 있는 크로스 엔트로피 손실을 사용하고, 질문-답변 쌍을 생성하여 LLM의 학습에 활용합니다.

- **Performance Highlights**: 초기 실험에서 PLUM은 100개의 대화를 기준으로 81.5%의 정확도를 달성하며, 기존 RAG 기반 방법의 83.5%와 경쟁력을 보였습니다. 이 연구는 LLM 개인화를 위한 향후 연구의 기초를 마련하고 있으며, 대화 기억을 통한 새로운 개인화 방법론의 가능성을 확인시켜 줍니다.



### Fact-Level Confidence Calibration and Self-Correction (https://arxiv.org/abs/2411.13343)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 LLMs의 자신감(calibration) 문제를 해결하기 위해 사실 수준(fact-level) 보정(framework) 방법을 제안하고 있습니다. 기존의 방법은 전체 응답에 대한 두 개의 스칼라 값을 추정하는 방식으로, 긴 형식의 생성에서는 여러 개의 사실이 포함된 경우 적절하지 않음을 지적합니다. 연구진은 각 사실의 정확도와 관련성(relevance)을 고려하여 보정(calibration) 과정을 보다 세분화하여 진행합니다.

- **Technical Details**: 새로운 사실 수준 보정 프레임워크에서는 각 사실의 정확성과 관련성을 기반으로 자신감을 평가합니다. 이 프레임워크는 모델이 사실 개별적으로 부분적인 자신감과 정확성을 보일 수 있도록 합니다. 또한, Confidence-Guided Fact-level Self-Correction ($\textbf{ConFix}$) 방법이 개발되어 높은 자신감을 가진 사실을 사용하여 낮은 자신감을 가진 사실의 질을 향상시키는 접근 방식을 제시합니다.

- **Performance Highlights**: ConFix 방법은 네 가지 데이터셋과 여섯 가지 모델에 걸쳐 실험된 결과, 허구(hallucination)의 발생을 현저히 줄임으로써 모델의 신뢰성을 증대시킴을 보여주었습니다. 이 방법은 외부 지식 소스를 요구하지 않고도 자체적으로 허구를 완화할 수 있는 가능성을 제시합니다. 전체적으로, 이 연구는 LLMs의 사실 기반 자신감 보정과 사용 가능한 자가 수정 방법을 제시하여, 모델의 견고한 응용을 가능하게 합니다.



### Combining Autoregressive and Autoencoder Language Models for Text Classification (https://arxiv.org/abs/2411.13282)
- **What's New**: 이 논문은 텍스트 분류를 위해 자기회귀(autoregressive)와 오토인코더(autoencoder) 언어 모델을 통합한 새로운 방법인 CAALM-TC(Combining Autoregressive and Autoencoder Language Models for Text Classification)를 제안합니다. CAALM은 입력 텍스트를 기반으로 문맥 정보를 생성하는 자기회귀 모델을 활용하여 원본 텍스트와 결합한 후, 이를 통해 오토인코더 모델에 전달하여 분류를 수행합니다. 이 하이브리드 접근법은 자기회귀 모델의 방대한 문맥적 지식과 오토인코더의 효율적인 분류 능력을 결합하여 텍스트 분류 성능을 향상시킵니다.

- **Technical Details**: CAALM은 작은 데이터셋 및 추상적인 분류 작업에서 특히 우수한 성능을 보이며, 기존 기법들에 비해 상태-of-the-art의 성능을 달성하는 것을 목표로 합니다. 실험은 Mistral NeMo 자가 회귀 언어 모델과 여러 가지 최상위 성능의 BERT 기반 모델에서 수행되었으며, CAALM은 사회 과학 텍스트 분류 작업에서 다른 접근 방식들을 초월할 수 있음을 보여주었습니다. CAALM은 데이터를 새롭게 생성하지 않고도 성능 향상을 달성하여 데이터의 유효성을 보장합니다.

- **Performance Highlights**: 임상 데이터와 전문적인 라벨링이 부족한 경우에도 CAALM은 효과적으로 작업을 수행할 수 있으며, 결과적으로 샘플 크기 요구 사항을 최소화하여 자동화된 콘텐츠 분석을 위한 확장 가능하고 효과적인 솔루션을 제공합니다. 다섯 개의 벤치마크 데이터셋에서의 실험 결과는 CAALM이 기존 방법들보다 일관되게 더 우수한 성능을 보임을 나타냅니다. 이 연구는 소셜 사이언스 연구에서 텍스트 분류의 자동화를 위한 가능성을 더욱 강조합니다.



### Leveraging Prior Experience: An Expandable Auxiliary Knowledge Base for Text-to-SQL (https://arxiv.org/abs/2411.13244)
- **What's New**: 이번 논문에서는 LLMs(대규모 언어 모델)의 학습 방식을 개선하기 위해 LPE-SQL(Leveraging Prior Experience: An Expandable Auxiliary Knowledge Base for Text-to-SQL)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 파라미터 세분화 없이 지속적인 학습(contual learning)을 가능하게 하며, 기존 모델의 성능을 크게 향상시킵니다. LPE-SQL은 올바른 쿼리와 실수를 기록하는 노트북 시스템을 도입하여 과거 경험을 활용하는 방식으로 학습합니다.

- **Technical Details**: LPE-SQL 프레임워크는 네 가지 모듈로 구성되어 있습니다: i) 관련 항목 검색, ii) 효율적인 SQL 생성, iii) 교차 일관성 메커니즘을 통한 최종 결과 생성, 그리고 iv) 성공 및 실패한 작업과 그 과정에서의 반성 생성 팁 기록입니다. 특히 네 번째 모듈이 핵심 기능을 담당하며, 다른 모듈은 기본 방법을 사용하여 기존 SoTA(최첨단) 기술과 쉽게 통합될 수 있도록 설계되었습니다. 실험적으로, LPE-SQL은 소형 Llama-3.1-70B 모델이 대형 Llama-3.1-405B 모델보다 우수한 성능을 발휘하도록 합니다.

- **Performance Highlights**: 시험 결과는 지속적인 학습 접근 방식이 성능 향상에 있어 탁월한 효과를 보임을 보여줍니다. 특히, LPE-SQL 방법을 사용하는 Llama-3.1-70B 모델은 SoTA 기술을 사용하는 더 큰 모델보다 성능이 뛰어난 것으로 나타났습니다. 이 연구는 SQL 쿼리 생성의 정확성을 높이는 동시에, 기존의 데이터 자원을 최적화할 수 있는 가능성을 제시합니다.



### BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework (https://arxiv.org/abs/2411.13237)
- **What's New**: 최근 발표된 논문에서는 Block Inverse Prompting (BIPro)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 제약이 있는 글쓰기 작업, 특히 열린 주제의 중국 전통시 생성에서 뛰어난 결과를 보여줍니다. BIPro는 두 가지 블록 역 프롬프트 방법인 수정(revise)과 재작성(rewrite)을 활용하여 인공지능 텍스트 작성을 인간처럼 모방합니다.

- **Technical Details**: BIPro 프레임워크는 GLM-10B-Chinese라는 그리 강력하지 않은 블록 생성 모델을 기반으로 하며, 이전 및 이후의 문맥을 고려하여 중간 텍스트 생성을 가능하게 합니다. 이 방법은 특히 중국 전통시 생성이라는 도전 과제에서 효과적입니다. 각 문장을 생성한 후 다음 문장을 생성하고 나서 수정하는 과정을 거쳐, 글의 질을 크게 향상시킵니다.

- **Performance Highlights**: BIPro를 사용하여 생성된 시는 GPT-4, GLM-4와 같은 첨단 직접 생성 시스템뿐 아니라 Yusheng, Shisanbai와 같은 도메인 특정 시스템보다 인간 평가에서 우수한 성과를 거두었습니다. 이 논문은 블록 생성 모델의 잠재력을 보여주며, AI 생성 시와 인간 문학작품 간의 격차를 좁혔습니다.



### AIDBench: A benchmark for evaluating the authorship identification capability of large language models (https://arxiv.org/abs/2411.13226)
Comments:
          21 pages, 7 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 익명 텍스트의 저자 식별에 미치는 프라이버시 위험을 다룹니다. 저자 식별을 위한 새로운 벤치마크인 AIDBench를 소개하며, 이메일, 블로그, 리뷰, 연구 논문 등을 포함한 여러 데이터셋을 포함합니다. 또한, Retrieval-Augmented Generation (RAG) 기반의 방법론을 제안하여 LLM의 저자 식별 능력을 향상시킵니다.

- **Technical Details**: AIDBench는 LLM을 이용한 저자 식별을 평가하기 위해 설계된 포괄적인 벤치마크입니다. 이 벤치마크는 저자 식별 작업을 위한 여러 데이터셋과 평가 작업, 지표를 사용하여 구성되어 있습니다. 저자 식별은 두 가지 주요 평가 방법론, 즉 one-to-one 및 one-to-many 형식으로 접근하며, RAG 기법을 통해 정보가 초과될 경우에도 효과적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: AIDBench 실험을 통해 LLM이 저자 식별에서 기대 이상으로 잘 작동함을 확인했으며, 이는 랜덤 추측 이상의 정확도를 보여줍니다. 연구 결과, LLM은 프라이버시 침해의 도구로 작용할 잠재성이 있으며, 이러한 가능성은 무시할 수 없습니다. 코드와 데이터는 수용 후 공개될 예정입니다.



### Hard-Synth: Synthesizing Diverse Hard Samples for ASR using Zero-Shot TTS and LLM (https://arxiv.org/abs/2411.13159)
- **What's New**: 이번 연구에서는 Hard-Synth라는 새로운 데이터 증강 방법을 제안합니다. 이 방법은 대형 언어 모델(LLM)과 제로샷 텍스트-음성 변환(TTS)을 활용하여 기존 ASR(자동 음성 인식) 모델의 어려운 발화 스타일을 복제합니다. 하드 프롬프트 선택 기법을 도입하여 ASR 모델이 인식하기 어려운 발화에 기반한 음성을 합성함으로써 훈련 데이터를 보강합니다.

- **Technical Details**: Hard-Synth 방법은 훈련 세트의 원본 텍스트를 LLM을 이용하여 재작성하고, 취약한 ASR 모델을 사용하여 어려운 발화를 식별합니다. 이후 이러한 발화를 제로샷 TTS를 통해 음성으로 합성하는 과정을 거칩니다. 연구팀은 악센트나 음색과 같은 음향적 특성을 유지하면서, 의미는 동일하지만 단어와 구조가 다른 텍스트를 생성합니다.

- **Performance Highlights**: 실험 결과, Hard-Synth는 Conformer 모델의 성능을 유의미하게 향상시켜 LibriSpeech 데이터셋에서 상대 단어 오류율(WER)을 각각 6.5% 및 4.4% 감소시켰습니다. 단 16.15시간의 합성 데이터로 이러한 개선을 달성하였으며, 이는 실제 데이터의 16%에 해당합니다. 또한, ASR의 성별 편향 및 화자 간 성능 변동 문제를 효과적으로 줄이는 것으로 나타났습니다.



### Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding (https://arxiv.org/abs/2411.13157)
- **What's New**: 최근 대형 언어 모델(LLMs)의 효율적인 추론(inference)은 이들의 규모와 복잡성이 커짐에 따라 중요한 이슈로 대두되고 있습니다. 전통적인 오토리그레시브 디코딩(autoregressive decoding)은 순차적으로 토큰을 생성하는 방식으로 인해 계산비용에서 비효율적입니다. 이 연구에서는 초안(draft)과 확인(verification)이라는 두 단계의 접근 방식을 도입하여 교묘한 디코딩(speculative decoding) 기술을 통해 이러한 병목 현상을 해결하는 방법을 모색합니다.

- **Technical Details**: 교묘한 디코딩은 작은 모델을 사용하여 초기 초안을 생성하고, 보다 큰 모델이 이 초안을 확인하는 방식으로 작동합니다. 여기서는 모델 중심(model-centric)과 초안 중심(draft-centric) 방식으로 교묘한 디코딩 방법을 분류하며, 각 접근법의 핵심 아이디어와 잠재력을 다룹니다. 기술적으로, 초안 생성의 질과 효율성을 개선하는 방법으로 다양한 구현 전략에 대해 이야기하고 있습니다.

- **Performance Highlights**: 성공적인 교묘한 디코딩 방법으로는 Medusa와 EAGLE-2가 언급됩니다. Medusa는 추가 디코딩 헤드를 사용하여 후속 토큰을 병렬로 처리하는 방식으로 성능을 향상시켰습니다. 반면 EAGLE-2는 동적인 초안 트리를 통해 더 나은 샘플링을 가능하게 하여 추론 속도와 정확성을 높이는 데 기여하고 있습니다.



### Song Form-aware Full-Song Text-to-Lyrics Generation with Multi-Level Granularity Syllable Count Contro (https://arxiv.org/abs/2411.13100)
- **What's New**: 이번 논문은 가사 생성에 있어 독특한 도전 과제를 다루고 있으며, 특히 정확한 음절 제어(syllable control)를 실현하는 데 초점을 맞추고 있습니다. 기존의 문장별 접근 방식(line-by-line)은 종종 부자연스러운 어구를 만들어내기 때문에, 보다 세밀한 음절 관리가 필요함을 강조하고 있습니다. 저자들은 다양한 수준의 음절 제어가 가능한 가사 생성 프레임워크를 제안하며, 입력 텍스트와 곡 형식에 따라 조정된 완전한 가사를 생성할 수 있게 됩니다.

- **Technical Details**: 제안된 모델은 단어, 구, 문장 및 단락 수준에서의 유연한 가사 생성을 지원합니다. 생성 과정에서는 특정 곡 형식에 맞춘 구조화된 토큰(token) 시스템을 사용하며, 이를 통해 노래의 각 부분의 전환을 나타냅니다. 예를 들어, <VERSE>, <CHORUS> 같은 곡 형식 토큰과 함께 정의된 음절 수 토큰(<SYL:s𝑠sitalic_s>)을 도입하여 모델이 음절 정밀도를 유지하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 기존의 대형 언어 모델(LLMs)들이 자연어 생성을 위해서는 뛰어난 성능을 보이지만, 음절 수를 정확히 계산하는 데에는 어려움을 겪고 있다는 사실이 드러났습니다. 저자들은 이 모델을 사용하여 가사를 생성하는 측면에서 음절 수와 구조에 따라 LLM을 평가했으며, 그 결과 약 38%~57%의 성공률을 기록했습니다. 제안된 모델의 성능은 기존 모델들에 비해 음절 제어에서 더욱 우수함을 보였으며, 각 조건에서 성공적으로 생성된 가사에 대한 성능 지표가 제시되었습니다.



### Patience Is The Key to Large Language Model Reasoning (https://arxiv.org/abs/2411.13082)
Comments:
          The dataset and model are available at this https URL

- **What's New**: 최근 대형 언어 모델(LLM) 분야에서 Chain of Thought (CoT) 접근 방식의 발전은 복잡한 문제 해결 능력을 크게 향상시켰습니다. 그러나 기존 모델은 사용자 선호에 따라 세부적인 추론을 희생하거나 상세한 사고 능력을 학습하기 위해 방대한 데이터가 필요하여 복잡한 작업을 해결하는 능력이 제한적이었습니다. 이를 해결하기 위해 연구팀은 모델이 새로운 지식이나 기술 없이도 보다 인내심 있는 추론 방식을 채택하도록 유도하는 간단한 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 약 5,000개의 수학 문제를 수집하여, 이를 통해 기본적인 CoT 해결 방안을 생성하고 이를 세분화하여 보다 상세하고 이해하기 쉬운 단계로 다듬는 과정을 거칩니다. 최종적으로, 이러한 개선된 해결 방안을 긍정적인 예로 사용하고 간단한 해결 방안을 부정적인 예로 사용하여 DPO (Diffusion Preference Optimization) 기법을 활용해 모델을 최적화합니다. 이 방법은 주어진 데이터셋에서 모델이 즉각적인 반응보다는 정확한 문제 해결을 할 수 있도록 유도합니다.

- **Performance Highlights**: 최적화된 모델은 GSM8k 벤치마크에서 6.7%의 성능 향상을 달성했습니다. 학생 수준의 수학 문제 해결 능력을 향상시키는 데 주목할 만한 효과가 있으며, MATH 벤치마크에서도 정확도가 0.2% 증가했습니다. 이 방법은 낮은 비용으로 고성능을 달성했으며, 추론 시간이 증가하였음에도 불구하고 보다 정확한 답변을 제공하는 것이 추천됩니다.



### MemoryFormer: Minimize Transformer Computation by Removing Fully-Connected Layers (https://arxiv.org/abs/2411.12992)
Comments:
          NeurIPS2024

- **What's New**: 이번 연구에서는 MemoryFormer라는 새로운 트랜스포머 아키텍처를 제안하여 계산 복잡성을 크게 줄입니다. 기존의 트랜스포머 모델에서 요구되는 거의 모든 계산을 제거하고, 다중 헤드 어텐션(multi-head attention) 연산에 필요한 최소한의 계산만 남깁니다. 이를 위해 완전 연결 층(fully-connected layer)의 선형 프로젝션 대신, 메모리 레이어를 사용하여 기능 변환을 수행합니다.

- **Technical Details**: MemoryFormer는 해시 알고리즘을 사용하여 입력 임베딩과 연관된 벡터의 하위 집합을 동적으로 검색하는 방식으로 작동합니다. 메모리 레이어는 이렇게 검색된 벡터들을 다양한 가중치로 집계하여 출력 임베딩을 형성하는데, 이 과정은 기존의 매트릭스 곱셈 연산을 대체하는 역할을 합니다. 해시 연산으로 인한 계산량은 미미하므로, MemoryFormer는 기존 트랜스포머 블록에 비해 약 19%의 FLOPs 계산만으로 동일한 작업을 수행할 수 있습니다.

- **Performance Highlights**: MemoryFormer는 여러 공인 벤치마크에서 테스트를 실시하여, 기존의 트랜스포머와 비교해 성능 유사성을 입증했습니다. 모델 크기를 늘려도 계산 복잡성이 줄어드는 효과를 보여주며, 차세대 병렬 컴퓨팅 플랫폼의 하드웨어 설계에도 기여할 수 있을 것으로 기대됩니다. 전반적으로, MemoryFormer는 더 적은 계산으로 비슷한 성능을 달성할 수 있음을 입증했습니다.



### Training Bilingual LMs with Data Constraints in the Targeted Languag (https://arxiv.org/abs/2411.12986)
Comments:
          22 pages, 14 figures, 15 tables

- **What's New**: 본 연구는 데이터가 부족한 목표 언어(target language)의 성능을 향상시키기 위해 고품질 보조 언어(auxiliary language)의 데이터를 활용하는 방법을 탐구합니다. 특히 영어와 같이 데이터가 풍부한 보조 언어의 정보를 활용하여 성능 차이를 정량적으로 평가하고, 모델의 스케일링 한계와 데이터 업샘플링 방법을 제안합니다. 기존의 영어 중심 모델들과는 달리, 목표 언어의 성능 향상을 위한 새로운 길을 모색하고 있습니다.

- **Technical Details**: 이 연구에서는 기존의 영어 데이터 선택 파이프라인(data selection pipelines)을 사용하여 데이터가 제한적인 목표 언어에 대한 사전 학습(pretraining) 성능을 증진시키는 방법을 실험합니다. 독일어를 목표 언어로 설정하고, 다양한 스케일의 디코더 전용 트랜스포머 모델을 학습하여 성능 개선을 분석합니다. 모델은 시간적으로 30K와 100K 스텝에서 학습하며, 배치 크기는 1024입니다.

- **Performance Highlights**: 연구 결과, 고품질의 보조 데이터는 목표 언어의 성능 향상에 긍정적인 영향을 미치며, 정보가 풍부한 영어 데이터셋이 다른 언어에서도 유사한 성과를 보일 수 있음을 보여줍니다. 특히, 영어 데이터셋의 품질이 높아질수록 목표 언어에서의 성능도 함께 향상되며, 가까운 언어(cognate languages)에서 효과가 두드러지지만, 영어와 거리가 먼 언어에서는 효과가 제한적임을 확인하였습니다.



### A Flexible Large Language Models Guardrail Development Methodology Applied to Off-Topic Prompt Detection (https://arxiv.org/abs/2411.12946)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 논문은 사용자들이 LLMs(Large Language Models)를 오용할 때 발생하는 문제를 다루고 새로운 방법론을 제시합니다. 기존의 guardrails는 제한적인 예시나 맞춤형 분류기에 의존하여 높은 오류율과 적응성 부족, 실제 데이터의 부족 등의 문제를 안고 있었습니다. 논문에서는 문제 영역을 질적으로 정의하고 이를 통해 다양한 프롬프트를 생성함으로써 합성 데이터셋을 구축하였으며, 이는 기존의 휴리스틱 방법보다 우수한 오용 방지 기능을 가지게 됩니다.

- **Technical Details**: 새롭게 제안된 데이터 없는 guardrail 개발 방법론은 특정 작업에 대한 사용자 프롬프트의 적합성을 분류함으로써 작동합니다. 이 방법론은 LLM을 활용해 다양한 합성 프롬프트를 생성하고, 이를 활용하여 오프-토픽 프롬프트를 탐지하는 분류기를 훈련합니다. 최종적으로 이 방식은 false positive를 줄이고, 다양한 오용 유형에 대한 일반화 가능성을 갖추게 됩니다.

- **Performance Highlights**: 제안된 방식은 모델의 초기 배포를 위해 강력한 기반을 제공하며, 합성 데이터로 훈련된 분류기는 기존의 휴리스틱 방법과 비교하여 오프-토픽 프롬프트 인식 정확도를 크게 향상시킵니다. 또한, 개방형 리소스 형태로 합성 데이터셋 및 오프-토픽 guardrail 모델을 공개하여 향후 LLM 안전성과 규정 준수 연구에 기여하도록 하고 있습니다.



### Signformer is all you need: Towards Edge AI for Sign Languag (https://arxiv.org/abs/2411.12901)
Comments:
          Official Code at: this https URL

- **What's New**: 이번 논문에서는 Signformer라는 새로운 Sign Language Translation (SLT) 모델을 제안합니다. Signformer는 사전 훈련된 모델이나 자원 집약적인 기술에 의존하지 않고, 기존의 gloss-free 방식을 개선하여 Edge AI 환경에서 효율적인 실시간 사용이 가능합니다. 이로써 SLT 분야의 지속 가능성과 실용성을 재정의하고 있습니다.

- **Technical Details**: 이 연구는 기존의 Sign Language Transformer (SL-Transformer) 아키텍처를 기반으로 하여, 새로운 알고리즘 설계를 위해 다양한 수화 언어의 구조적 특성을 철저히 분석합니다. 이를 통해 변형된 컨볼루션, 주의(attention) 메커니즘 및 위치 인코딩을 활용하여 파라미터 수를 극적으로 줄이면서도 성능을 극대화할 수 있었습니다. Signformer 모델은 GLT와 비교해 1807배 더 작은 파라미터 수를 자랑하며, 기존의 여러 SOTA 모델에 비해 뛰어난 성능을 올리고 있습니다.

- **Performance Highlights**: Signformer는 gloss-free SLT 선두주자로 자리매김하면서, 2024년 기준으로 새로운 2위 성적을 기록했습니다. 특히 Signformer-Full 모델은 gloss-free Finest SignLLM보다 더 나은 테스트 세트 성능을 발휘하며, 0.57백만의 파라미터 수를 통해 TOP5에 등재되었습니다. 이는 SLT 분야에서 파라미터와 성능의 균형을 새롭게 정의하는 중요한 이정표가 됩니다.



### AzSLD: Azerbaijani Sign Language Dataset for Fingerspelling, Word, and Sentence Translation with Baseline Softwar (https://arxiv.org/abs/2411.12865)
- **What's New**: 이번 연구에서는 아제르바이잔 수화(Azerbaijani Sign Language) 데이터셋(일명 AzSLD)을 소개합니다. 이 데이터셋은 다양한 연령, 성별 및 수화 스타일을 가진 사용자들로부터 수집된 30,000개의 비디오로 구성되어 있습니다. 각 비디오는 정밀한 수화 라벨과 해당 언어 번역이 주석으로 달려 있으며, 데이터의 질을 보장하기 위해 두 개의 카메라 앵글로 촬영되었습니다.

- **Technical Details**: AzSLD는 손가락 철자(fingerspelling)와 단어, 문장 수준의 수화 데이터셋을 포함합니다. 손가락 철자 데이터셋은 32개의 아제르바이잔 알파벳 문자를 정정하고 10,864개의 이미지와 3,587개의 비디오로 구성되었습니다. 모델의 훈련과 검증을 용이하게 하기 위해 데이터가 안전하고 정확하게 주석 처리되었으며, 비디오 촬영에는 고유한 설정이 사용되었습니다.

- **Performance Highlights**: AzSLD는 연구자와 개발자들에게 손쉬운 접근을 제공하는 유용한 자원입니다. 이 데이터셋은 고품질의 라벨이 달린 데이터를 제공하여 수화 인식 및 번역 시스템의 발전을 지원하며, 데이터 로더와 기술 문서가 공개되어 있어 사용에 대한 편의성을 높입니다. 이 데이터셋은 아제르바이잔 수화 연구 및 기술 개발에 기여할 것으로 기대됩니다.



### Probing the Capacity of Language Model Agents to Operationalize Disparate Experiential Context Despite Distraction (https://arxiv.org/abs/2411.12828)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM) 에이전트의 새로운 연구 결과를 제시하고 있습니다. 특히 OEDD(Operationalize Experience Despite Distraction)라는 새로운 데이터셋을 소개하며, 이는 다양한 경험 정보를 바탕으로 distractor가 있는 상황에서 에이전트가 의사 결정을 내리는 과정을 평가합니다. 이는 LLM의 경험 기반 추론 능력을 향상시키기 위한 기초 자료로 사용될 수 있습니다.

- **Technical Details**: OEDD 데이터셋은 인간 주석자에 의해 검증된 시나리오로 구성되어 있으며, 에이전트는 주어진 입력 프롬프트에서 여러 개의 경험적 정보를 바탕으로 분석해야 합니다. 연구에서는 최첨단 LLM인 GPT-3.5 Turbo, GPT-4o 및 Gemini 1.5 Pro를 평가하고, 최소한의 체인 오브 씽킹(chains of thought) 프롬프트 전략을 적용하였습니다. 평가 과정에서는 1,615개 이상의 토큰을 포함하는 입력 컨텍스트와 두 개의 상반된 환경 전제에서 결정적인 결론을 도출해야 하며, 이후에 주의 분산 요소인 trivial한 정보를 갖는 distractor가 등장합니다.

- **Performance Highlights**: 이 실험 결과, 모든 LLM들은 두 가지 행동 중 더 나은 선택을 할 때 무작위 선택보다 성능이 떨어지는 것으로 나타났습니다. 이로 인해, 복잡한 경험적 정보가 존재하는 상황에서 LLM의 결정 능력을 향상시킬 필요가 확인되었습니다. 연구팀은 코드와 테스트 데이터셋을 공개하여, 다른 연구자들이 이를 활용할 수 있도록 하고 있습니다.



### CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization (https://arxiv.org/abs/2411.12768)
- **What's New**: 최근 연구 결과에 따르면, Large Language Models (LLMs)는 '백도어 공격(backdoor attacks)'에 취약하다는 것이 밝혀졌습니다. 이러한 공격은 적대자가 숨겨진 트리거를 삽입하여 모델의 응답을 조작하는 방식으로 이루어집니다. 본 논문에서는 Internal Consistency Regularization (CROW)라는 새로운 방어 기법을 소개하며, 이는 모델 훈련 중 일관성을 촉진하여 백도어 트리거로 인한 레이어 간 불일치를 해결합니다.

- **Technical Details**: CROW는 적대적 섭동(adversarial perturbations)과 정규화를 통해 내부 일관성(internal consistency)을 강화하여 백도어의 영향을 중화합니다. 이 방식은 클린 데이터(clean data) 세트만을 기반으로 하며, 클린 참조 모델이나 사전 트리거 지식이 필요하지 않아 다양한 LLM 아키텍처에서의 실용성을 높입니다. 또한, CROW는 레이어 간의 일관성을 정의하고 백도어가 이 일관성을 어떻게 교란시키는지를 명확히 하여 새로운 방어 기법의 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, CROW는 Llama-2, CodeLlama, Mistral-7B 등 다양한 모델에서 공격 성공률(ASR)을 일관되게 감소시키는 것으로 나타났습니다. 기존 방어 기법인 파인튜닝(finetuning), 가지치기(pruning), 양자화(quantization)와 비교할 때 성능이 월등히 개선되었으며, 모델의 생성 능력을 유지합니다. CROW는 100개의 클린 샘플로 4분 이내에 훈련을 완료할 수 있어 계산적으로도 효율적입니다.



### Suicide Risk Assessment on Social Media with Semi-Supervised Learning (https://arxiv.org/abs/2411.12767)
Comments:
          Accepted for publication in the 2024 IEEE International Conference on Big Data

- **What's New**: 이 논문에서는 자살 위험 평가를 위한 새로운 반지도 학습 프레임워크를 제안하고 있습니다. 기존 연구의 한계를 극복하기 위해 500개의 라벨이 있는 데이터와 1,500개의 라벨이 없는 데이터를 활용하며, 데이터의 불균형 문제를 해결하기 위해 새로운 pseudo-label 획득 과정을 도입했습니다. 또한, RoBERTa 모델을 기본 구조로 사용하여 성능을 극대화했습니다.

- **Technical Details**: 연구에서 사용되는 데이터는 Reddit의 r/SuicideWatch 서브레딧에서 수집된 500개의 라벨이 있는 사례와 1,500개의 라벨이 없는 사례로 구성됩니다. 데이터셋은 네 가지 자살 위험 수준으로 분류되며, 반지도 학습(SSL) 알고리즘이 적용됩니다. 특히, Stratified Confidence Sampling (SCS) 알고리즘을 사용하여 자살 위험의 낮은 클래스에 대한 pseudo-label 정확성을 보장합니다.

- **Performance Highlights**: 반지도 학습 방법론을 통해, 라벨이 없는 데이터를 활용하여 자살 위험 수준을 식별하는 데 있어 기존의 감독형 접근법보다 향상된 예측 성능을 입증했습니다. 모델의 성능은 다양한 심리적 표현의 정도를 식별하는 데 매우 효과적이며, 자살 예방을 위한 자동화된 시스템 구축에 기여할 것으로 기대됩니다.



### SEFD: Semantic-Enhanced Framework for Detecting LLM-Generated Tex (https://arxiv.org/abs/2411.12764)
- **What's New**: 본 논문은 대용량 언어 모델(LLM)이 생성한 텍스트를 탐지하기 위한 새로운 접근법인 개선된 의미 기반 프레임워크(SEFD)를 소개합니다. 이 프레임워크는 기존 탐지 방법론에 비해 더 robust한 성능을 제공하며, 특히 paraphrasing 기법을 통한 텍스트 변형에 효과적으로 대응합니다. 텍스트의 의미를 충분히 활용하기 위해 검색 기반 메커니즘을 통합한 것이 특징입니다.

- **Technical Details**: SEFD 프레임워크는 초기 탐지 단계, 의미 유사성 계산, 의미가 강화된 탐지의 세 가지 주요 단계를 포함합니다. 초기 탐지를 위해 기존 탐지기를 사용하고, BERT 기반 모델을 통해 검색 풀 내 텍스트와의 의미 유사도를 평가합니다. 마지막으로, 탐지 점수와 유사성 점수를 통합하여 최종 점수를 도출하며, 이 과정에서 검색 풀도 지속적으로 업데이트됩니다.

- **Performance Highlights**: 본 연구에서는 SEFD 프레임워크가 실제 환경에서 흔히 발생할 수 있는 순차 텍스트 시나리오에서의 효율성을 입증합니다. 다양한 LLM 생성 텍스트와 탐지 방법을 사용한 실험을 통해, paraphrasing 상황에서도 탐지 정확성이 크게 향상됨을 보여주었습니다. 이로 인해 데이터의 무결성을 보장하고 신뢰할 수 있는 정보 제공에 기여할 것으로 기대됩니다.



### Playing Language Game with LLMs Leads to Jailbreaking (https://arxiv.org/abs/2411.12762)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)의 안전성을 우회하는 새로운 jailbreak 공격 방법을 제안합니다. 연구자들은 mismatched generalization(불일치 일반화)이라는 개념을 기반으로 한 자연어 게임(natural language games)과 맞춤형 언어 게임(custom language games)이라는 두 가지 혁신적인 방법을 개발했습니다. 이 방법들은 LLM의 안전 메커니즘을 효과적으로 우회할 수 있어, 높은 공격 성공률을 기록했습니다.

- **Technical Details**: 제안된 jailbreak 방법은 자연어를 변형하여 공격자가 모델의 안전 제어를 피할 수 있도록 하는 언어 게임을 활용합니다. 특정 규칙을 적용하여 입력을 변형함으로써, LLM이 해로운 요청을 악의적인 것으로 인식하지 못하게 합니다. 연구에서는 GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet에 대한 실험을 통해 각각 93%, 89%, 83%의 성공률을 기록했습니다.

- **Performance Highlights**: 실험 결과, 제안된 자연어 게임 및 맞춤형 언어 게임이 LLM의 안전 정렬을 효과적으로 우회할 수 있다는 것을 입증했습니다. 또한, Llama-3.1-70B 모델의 미세 조정을 통해 상관된 안전 정렬이 다른 도메인에서 효과적으로 일반화되지 못하는 한계도 발견했습니다. 이는 현재의 안전 정렬 기법이 LLM에 내재된 안전 지식을 효과적으로 일반화하지 못하고 있음을 시사합니다.



### A Novel Approach to Eliminating Hallucinations in Large Language Model-Assisted Causal Discovery (https://arxiv.org/abs/2411.12759)
- **What's New**: 이 논문은 대규모 언어 모델(LLM) 사용 시 발생하는 환각(hallucinations)에 대한 최초의 조사를 제공합니다. 인과 발견(causal discovery)에서 인간 전문가 대신 LLM을 사용하는 증가하는 추세에 따라, 최적의 모델 선택의 중요성이 증가하고 있음을 강조합니다. 기존 LLM의 환각 문제를 다루며, 이를 감소시키기 위한 새로운 방법을 제안하고 있습니다.

- **Technical Details**: 연구에서는 Retrieval Augmented Generation (RAG) 방식을 사용하여 고품질 데이터(quality data)가 있을 때 환각을 줄이는 방법을 제안합니다. 또한, 여러 LLM을 사용하는 새로운 방법론을 도입하여 이들이 논쟁(debate)을 통해 인과 그래프(causal graphs)에서의 엣지를 감시(audit)하는 방식을 소개합니다. 이 방법은 RAG와 유사한 수준으로 환각을 감소시키는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 제안된 방법은 기존의 LLM 사용 시 발생하는 환각을 효과적으로 감소시키며, 인과 발견의 정확도를 높이는 데 기여할 수 있습니다. 연구 결과는 LLM의 선택이 인과 발견의 결과에 중요한 영향을 미친다는 것을 입증하며, 제시된 접근 방식들이 실제 적용 가능성을 보여줍니다.



### An exploration of the effect of quantisation on energy consumption and inference time of StarCoder2 (https://arxiv.org/abs/2411.12758)
- **What's New**: 이 연구는 코드 대형 언어 모델(code Large Language Models, LLM)의 추론(inference) 과정에서 에너지 소비를 줄이기 위한 양자화(quantization) 및 가지치기(pruning) 전략을 조사합니다. StarCoder2를 사용한 결과, 양자화는 처리량(throughput)이 낮아져 더 많은 에너지가 필요하고, 정확도(accuracy)가 떨어질 수 있음을 확인했습니다. 반면, 가지치기는 에너지를 줄일 수 있으나 성능을 저하시키는 한계가 있습니다. 이러한 결과는 LLM 모델 압축의 어려움과 트레이드오프(trade-off)를 강조합니다.

- **Technical Details**: AI의 에너지 소비를 줄이는 것은 중요한 문제로, 특히 추론 단계에서 사용자 인터랙션이 증가함에 따라 그 영향력이 커집니다. 양자화는 모델의 무게를 낮은 비트 포맷으로 변환하여 메모리를 감소시키고, 이는 사용자가 개인 장치에서 대형 모델을 실행할 수 있도록 돕습니다. 이 연구에서는 양자화와 가지치기를 통해 에너지 소비를 줄이는 방법을 탐구하며, 양자화가 성능 손실 없이도 모델을 작게 만들 수 있음에 유의합니다.

- **Performance Highlights**: 연구 결과, 양자화와 가지치기 모두 LLM의 에너지 소비를 감소시킬 수 있지만 각기 다른 성능적 영향을 미칩니다. 특히 양자화의 경우, QLoRA를 통해 양자화된 모델이 ChatGPT와 유사한 성능을 제공할 수 있게 되었습니다. 하지만 가지치기는 성능 저하를 초래할 수 있는 가능성이 높아, 향후 연구에서는 하드웨어 최적화 양자화 방법을 탐색하여 성능 손실을 최소화하는 방향으로 나아갈 필요성이 강조됩니다.



### AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations (https://arxiv.org/abs/2411.13451)
Comments:
          18 pages, 3 figures, an abridged version to appear in NeurIPS 2024 AFM Workshop

- **What's New**: 본 논문에서는 웹 기반 작업을 자동화하는 멀티모달 웹 에이전트를 위한 새로운 AdaptAgent 프레임워크를 제안합니다. 이 프레임워크는 불특정 웹사이트와 도메인에 대해 단 1~2개의 인간 시연을 통해 적응할 수 있도록 설계되었습니다. 실험 결과, 이 접근 방식이 현재 최첨단 모델에 비해 약 3.36%에서 7.21%의 작업 성공률 향상을 가져온다는 것을 보여주었습니다.

- **Technical Details**: AdaptAgent 프레임워크는 소수의 시연을 활용한 학습을 통해 기존의 대규모 사전 학습 및 미세 조정 전략을 보완하는 방법론을 제시합니다. 이 접근 방식은 두 가지 주요 메커니즘인 In-Context Learning (ICL)과 Meta-Learning을 통해 멀티모달 LLM(Multimodal Large Language Model) 기반 모델들이 새로운 웹 환경에 적응할 수 있게 합니다. 연구 결과, 멀티모달 시연이 텍스트 기반 시연보다 더 효과적임을 증명하였으며, 메타 학습에서의 데이터 선택 방법이 에이전트의 일반화 능력에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 두 가지 벤치마크인 Mind2Web과 VisualWebArena에서 수행된 실험을 통해, AdaptAgent 방식이 단 1-2개의 인간 시연을 사용하여 에이전트의 작업 성공률을 크게 향상시킬 수 있음을 입증했습니다. 전반적으로, 이 연구는 대규모 사전 학습과 미세 조정에 의존하지 않고도, 웹 에이전트의 일반화를 개선하는 보완적 방법론을 제시하며, 여러 웹 도메인에서의 작업 수행 가능성을 높이고 있습니다.



### WaterPark: A Robustness Assessment of Language Model Watermarking (https://arxiv.org/abs/2411.13425)
Comments:
          22 pages

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 오용을 줄이기 위한 방안으로 자리 잡고 있는 워터마킹 기법을 체계적으로 분석합니다. 기존 워터마커의 강점과 한계, 특히 공격에 대한 저항력을 집중적으로 평가하는 새로운 플랫폼인 WaterPark를 소개합니다. WaterPark는 10개의 최신 워터마커와 12개의 대표적인 공격을 통합하여 평가하는 데 초점을 맞추고 있으며, 그 설계 선택이 공격 저항력에 미치는 영향을 분석합니다.

- **Technical Details**: 토큰을 생성하는 과정에서 사용되는 LLM의 원리를 이해하기 위해, LLM의 워터마킹 방법을 세 가지 주요 요소인 LLM 자체, 워터마킹 절차(생성기), 탐지 절차(탐지기)로 나눕니다. 응용 프로그램 내에서 특정 통계적 신호를 시각적으로 나타내는 패턴으로서, 워터마커를 통해 LLM이 생성한 텍스트임을 입증할 수 있습니다. 이 연구에서는 대칭 및 비대칭 키를 이용한 워터마킹 기법의 작동 원리에 대해서도 설명합니다.

- **Performance Highlights**: 연구 결과는 여러 흥미로운 발견을 포함하며, 기존 연구의 결론에 도전하는 내용을 담고 있습니다. 예를 들어, 일부 워터마커는 예상보다 더 높은 오류율을 보이며, 여러 번의 패러프레이즈를 사용해야만 여전히 감지될 수 있음을 보여줍니다. 또한, 특정 워터마커는 고강도의 공격에는 저항하지 못함을 발견했습니다. 마지막으로, 격렬한 적대적 환경에서 워터마킹을 운영하기 위한 최상의 전략에 대해서도 논의합니다.



### CAFE A Novel Code switching Dataset for Algerian Dialect French and English (https://arxiv.org/abs/2411.13424)
Comments:
          24 pages, submitted to tallip

- **What's New**: 이번 논문에서는 알제리 방언, 프랑스어, 영어 간의 코드 스위칭(code-switching) 데이터셋인 CAFE를 최초로 소개하고 공개합니다. CAFE 데이터는 자발적인 대화 스타일을 반영한 고유한 음성과 대화 현상을 포착하며, 알제리의 다양한 방언 변화를 다룹니다. 약 37시간의 음성이 포함되어 있으며, 이 중 2시간 36분은 수동으로 주석이 달린 CAFE-small 세트로 제공됩니다.

- **Technical Details**: CAFE 데이터셋은 알제리 방언에서 발생하는 언어적 문제를 해결하기 위해 다양한 사회언어적 맥락에서 수집된 음성 데이터를 포함하고 있습니다. 데이터는 자동 음성 인식(Automatic Speech Recognition, ASR) 모델인 Whisper large-v2,3 및 PromptingWhisper와 함께 벤치마킹되어, 잘 설계된 데이터 처리 파이프라인과 고급 디코딩 기술이 ASR 성능을 향상시키는 방법을 보여줍니다. CAFE 데이터는 특정 이벤트(예: 코드 스위칭 포인트, 중첩된 발화 및 잡음)와 관련된 주석을 포함하고 있습니다.

- **Performance Highlights**: CAFE 데이터셋을 사용하여 성능을 평가한 결과, 혼합 오류율(Mixed Error Rate, MER) 0.310, 문자 오류율(Character Error Rate, CER) 0.329, 단어 오류율(Word Error Rate, WER) 0.538이 도출되었습니다. 이는 고급 음성 인식 모델이 코드 스위칭과 같은 복잡한 언어적 현상을 처리하는 데 있어 향상된 성능을 보여줍니다. 따라서 CAFE 데이터는 ASR 시스템 개발에 중요한 기여를 할 것으로 기대됩니다.



### Executable QR codes with Machine Learning for Industrial Applications (https://arxiv.org/abs/2411.13400)
Comments:
          preprint, 4 pages, 2024

- **What's New**: 본 논문에서는 eQR 코드(Executable QR codes)의 개념을 소개하며, 이 코드가 사용자 인터넷 접근이 불가능한 상황에서도 작동할 수 있는 프로그램을 포함하고 있다는 점에서 기존 QR 코드의 한계를 극복한다고 설명합니다. eQR 코드는 모바일 장치에서 실행되는 프로그램을 직접 바이너리 형식으로 인코딩하여, 복잡한 기계의 운영 및 유지보수를 도와주기 위한 의사 결정 트리 기능을 제공하는 QRtree 프로그래밍 언어를 최초로 사용하고 있습니다. 추가로, 이 논문에서는 산업을 위한 특별히 고안된 QRind 언어를 제안하여, 기계 학습 알고리즘과 예측 유지보수 알고리즘을 통합할 수 있도록 합니다.

- **Technical Details**: QRind는 의사 결정 트리를 구현할 수 있는 QRtree와 유사성을 지니고 있지만, 변수 선언 및 기계 학습 모델 통합을 지원하여 기술의 표현력을 높입니다. 가변 레지스터를 통해 무제한으로 변수를 정의할 수 있으며, 이는 현대 스마트폰의 처리 능력을 활용해 QR 코드에 필요한 정보 이상의 데이터를 처리할 수 있게 해줍니다. 또한, Boolean, int, float, string 및 배열과 같은 다양한 변수 유형이 정의되어 있어, 다양한 산업적 응용 사례에 적합합니다.

- **Performance Highlights**: 논문에서는 QRind 언어가 소형화와 고속 연산을 위한 최적화를 지원한다는 점을 강조하며, 특히 Industry 4.0/5.0 개념을 이끌 수 있는 가능성을 제시합니다. QRind의 코드는 높은 압축률을 유지하면서, 기존 eQR 코드의 미세한 구현 범위를 초월하는 유연성을 제공합니다. 이를 통해 산업 현장 및 다양한 응용 분야에서의 실용성을 확보할 수 있습니다.



### VideoAutoArena: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation (https://arxiv.org/abs/2411.13281)
Comments:
          Project Page: this https URL

- **What's New**: 최신 연구에서는 비디오 분석 능력을 갖춘 대형 다중 모달 모델(LMMs)에 대한 관심이 증가하고 있습니다. 기존 평가 방법들은 사용자 요구를 충분히 반영하지 못하고, 비용이 많이 드는 인적 주석이 필요했습니다. 이를 해결하기 위해 제안된 VideoAutoArena는 사용자 시뮬레이션을 기반으로 한 자동 평가 시스템으로, 질문 생성 방식에서 혁신적인 접근을 제공합니다. 또한, VideoAutoBench는 신속하고 효율적인 평가를 지원하여 실제 사용자 질문 스타일에 더 가깝게 평가할 수 있습니다.

- **Technical Details**: VideoAutoArena는 사용자에 의해 생성된 열린 질문을 통해 모델의 비디오 이해 능력을 엄격하게 평가하도록 설계되었습니다. 이 평가 모델은 ELO 평가 시스템을 통해 다양한 LMM 간의 공정하고 지속적인 비교를 가능하게 합니다. 또한, 점진적으로 질문의 복잡성을 증가시키는 결함 기반 진화 전략을 도입하여 모델이 더 도전적인 비디오 분석 시나리오를 처리할 수 있도록 합니다. 이를 통해 기존의 LMM 평가 방식에서 발생했던 여러 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, VideoAutoArena는 최신 LMM들 간의 비교를 효과적으로 수행하며, 모델의 강점과 개선이 필요한 영역에 대한 통찰력을 제공합니다. 비공식 모델들이 SOTA(SOTA: State Of The Art) 모델인 GPT-4o에 비해 여전히 성능 차이를 보이며, 이 차이는 비디오 길이가 늘어날수록 더욱 두드러집니다. VideoAutoBench의 결과는 VideoAutoArena의 평가 결과와 밀접하게 일치하는 것으로 나타나, 두 가지 벤치마크가 서로 보완적인 역할을 하고 있음을 보여줍니다.



### Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning (https://arxiv.org/abs/2411.13045)
Comments:
          Submitted to WWW 2025

- **What's New**: 이번 논문에서는 전자상거래 검색 시스템에서 사용자 경험과 만족도를 향상시키기 위해, 효과적인 query-item relevance(쿼리-아이템 관련성) 모델링의 중요성을 강조합니다. 최근에 등장한 Large Language Model(대규모 언어 모델) 접근법은 기존 신경망 기반의 전문 관련성 학습 방법에 비해 뛰어난 성능과 긴 꼬리(generalization) 능력을 보여주고 있습니다. 하지만 이 모델들은 온라인 배포에 어려움이 있으며, LLM의 복잡한 내부 지식을 추출하고 적용하는 데 한계가 있습니다.

- **Technical Details**: 이를 해결하기 위해 우리는 Explainable LLM-driven Multi-dimensional Distillation(설명 가능한 대규모 언어 모델 기반 다차원 증류) 프레임워크를 제안합니다. 이 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) ELLM-rele(설명 가능한 LLM)로서 관련성 모델링을 Chain-of-Thought(사고의 흐름) 추론으로 분해하여 해석 가능성과 성능을 향상시킵니다. 2) MKD(다차원 지식 증류) 아키텍처는 ELLM-rele의 지식을 현재 배포 가능한 모델에 전달하며, 이는 관련성 점수 분포와 CoT 추론 측면에서 이루어집니다.

- **Performance Highlights**: 우리는 이 프레임워크가 Taobao 검색 광고 환경에서의 방대한 오프라인 평가와 온라인 실험을 통해 전자상거래 관련성 학습 성능과 사용자 경험을 크게 개선함을 보였습니다. MKD는 학생 모델의 의미적 상호작용과 긴 꼬리 일반화 능력을 모두 향상시키는 데 기여하였습니다. 이러한 결과는 LLM의 해석 가능성과 모델 성능 향상이라는 두 가지 주요 목표를 달성함으로써, 향후 전자상거래 시스템에서 더욱 효과적인 적용이 가능함을 시사합니다.



### Breaking the Cycle of Recurring Failures: Applying Generative AI to Root Cause Analysis in Legacy Banking Systems (https://arxiv.org/abs/2411.13017)
- **What's New**: 전통적인 은행들은 유산 시스템(legacy systems)으로 인해 디지털 혁신에서 큰 도전에 직면하고 있습니다. 이 연구는 GenAI 에이전트와 "Five Whys" 기법을 통합하여 근본 원인(root cause) 분석을 개선하는 새로운 방법론을 제안합니다. 초기 분석 결과, 기존에 외부 요인으로 여겨졌던 약 70%의 사건들이 내부 코드 결함 때문이라는 사실을 밝혀냈습니다.

- **Technical Details**: 본 연구는 GenAI를 사용하여 사고 후 분석(post-incident analysis) 프로세스를 자동화하고 강화하는 방법을 제시합니다. 이는 'Five Whys' 기법을 보완하여 발생한 문제의 기술적 배경을 깊이 있게 분석할 수 있게 해줍니다. 지식 그래프(knowledge graph)를 활용하여 소프트웨어 개발 생명주기(SDLC) 전반에서 수집된 정보를 바탕으로 보다 정확한 근본 원인 분석을 가능하게 합니다.

- **Performance Highlights**: 이러한 접근법을 통해 한 글로벌 금융 회사에서 주요 사고의 발생을 연간 45%, 변경 실패율을 45.5%, 그리고 배포 리드 타임을 46.3% 감소시키는 성과를 거두었습니다. 이러한 성과는 다양한 개발 단계(phases)에서 GenAI 에이전트를 적용함으로써 가능했으며, 이는 소프트웨어 개발 프로세스 개선에도 기여할 것입니다.



### LLMSteer: Improving Long-Context LLM Inference by Steering Attention on Reused Contexts (https://arxiv.org/abs/2411.13009)
- **What's New**: LLMSteer는 fine-tuning 없이 LLM의 생성 품질을 향상시키는 혁신적인 프레임워크입니다. 이 접근법은 쿼리 독립적인 attention steering을 통해 모델이 선택된 토큰에 대한 주의를 집중하도록 조정합니다. LLMSteer는 기존 방법들과 비교하여 성능 격차를 65.9% 좁히고 런타임 지연을 최대 4.8배 감소시키는 것으로 나타났습니다.

- **Technical Details**: LLMSteer는 두 번의 다른 prefix prompts를 사용하여 LLM이 문맥을 여러 번 읽도록 유도합니다. 이를 통해 주목할 가치가 있는 토큰을 선정하고 이들의 attention score를 재조정하여 모델이 문맥을 효과적으로 이해할 수 있도록 합니다. 이러한 방식은 기존의 KV 캐시를 재사용할 수 있게 하여 LLM의 효율성과 품질을 동시에 개선합니다.

- **Performance Highlights**: LLMSteer는 LLaMA-3.1-8b-Instruct에서 구현되었으며, F1 점수를 72.9에서 82.0으로 증가시키는 성과를 거두었습니다. 또한 기존의 최고 성능인 attention steering 방법과 비교했을 때, 최대 4.8배 더 빠른 처리 속도를 제공합니다. 이 연구는 fine-tuning 없이 높은 모델 생성 품질을 달성하는 최초의 시도로 큰 주목을 받고 있습니다.



### MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Collaborative Learning (https://arxiv.org/abs/2411.12977)
- **What's New**: 최근의 복합체(embodied) 에이전트들은 오픈된 환경에서 스스로 학습하는 데 뛰어난 능력을 보여주고 있습니다. 하지만 대규모 언어 모델(LLM)과 결합될 경우, 이들 에이전트는 종종 기본적인 작업에서도 어려움을 겪습니다. 이를 해결하기 위해 제안된 \

- **Technical Details**: \collabvoyager는 인간의 문화적 학습 이론에 영향을 받아 개발된 새로운 프레임워크입니다. 이 프레임워크는 세 가지 주요 혁신으로 구성되어 있습니다: (1) 관점 이해 이론(theory of mind)을 통한 인지 연결, (2) 에이전트 간 자연어(Natural Language) 통신, (3) 작업 및 환경 지식의 의미 기억(semantic memory) 및 협력 경험의 에피소딕 기억(episodic memory). 이러한 기능들은 에이전트가 자신과 타인의 정신 상태를 이해하고 추론하도록 돕습니다.

- **Performance Highlights**: 실험 결과, \



### Loss-to-Loss Prediction: Scaling Laws for All Datasets (https://arxiv.org/abs/2411.12925)
- **What's New**: 이번 연구는 다른 데이터 분포에서의 손실 예측을 위한 새로운 방법론인 loss-to-loss prediction을 제시합니다. 기존의 scaling laws는 단일 데이터 분포의 훈련 손실 예측에 유용하지만, 이 연구는 다양한 pre-training 데이터셋 간 그리고 모델 간의 손실 관계를 탐구합니다. 또한, 연구 결과는 모델이 훈련된 데이터셋 간의 손실 변환을 통해 scaling laws를 효과적으로 활용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 논문에서는 세 가지 유형의 loss-to-loss 관계를 살펴봅니다. 첫 번째는 train-to-train 관계로, 서로 다른 데이터셋에서 훈련된 모델 간의 손실을 비교합니다. 두 번째는 train-to-test 관계로, 한 데이터셋에서 훈련된 모델의 성능을 다른 데이터셋에서 평가하는 방식입니다. 마지막으로 test-to-test 관계는 서로 다른 데이터셋에서 훈련된 모델의 테스트 손실을 비교합니다. 이 모든 관계에서 shifted power law가 관찰되었습니다.

- **Performance Highlights**: 연구 결과는 다양한 데이터셋과 다운스트림 작업에서 예측 가능한 손실 간의 관계를 입증했습니다. 특히, 적은 수의 샘플로부터 새로운 데이터셋에 대한 만약의 성능을 예측할 수 있는 전이 성능이 향상되었음을 보여줍니다. 또한, 여러 쌍의 데이터셋을 사용하면 새로운 데이터셋에서의 성능 예측이 더 정확해질 수 있음을 발견했습니다.



### Selective Attention: Enhancing Transformer through Principled Context Contro (https://arxiv.org/abs/2411.12892)
- **What's New**: 이번 연구에서는 기존의 self-attention 메커니즘이 쿼리에 대해 균일하게 처리하는 방식이 문맥의 희소성과 적합성을 제어하는 능력을 저해한다고 주장합니다. 따라서, Selective Self-Attention (SSA) 레이어를 도입하여 소프트맥스 비선형성을 열 조절 방식을 통해 보강합니다. 이 방법을 통해 쿼리 임베딩과 컨텍스트 윈도우에서의 위치에 따라 문맥의 희소성을 조절할 수 있으며, 결과적으로 주의력 희석을 완화하고 전반적인 모델 성능을 향상시킵니다.

- **Technical Details**: Selective Self-Attention (SSA) 레이어는 쿼리 및 값 임베딩에 온도 조절(Temperature Scaling, TS)을 적용하며, 이를 통해 문맥의 희소성과 의미적 유사성을 분리합니다. SSA는 쿼리 임베딩에 대해 계산된 온도를 바탕으로 소프트맥스를 적용하고, 하이퍼파라미터를 조정을 통해 원활한 학습이 가능합니다. 이론적으로는, SSA가 더 적은 파라미터로도 목표 주의 맵을 표현할 수 있게 돕는다는 점을 증명했습니다.

- **Performance Highlights**: SSA를 적용한 모델은 Wikitext, Lambada, Piqa 등의 언어 모델링 벤치마크에서 눈에 띄고 일관된 정확도 개선을 이루었습니다. 이 접근 방식은 GPT-2, Pythia, Llama 등 다양한 모델에서 테스트되었으며, fine-tuning과 pre-training 과정 모두에서 긍정적인 결과를 보였습니다. 추가로, SSA는 transformer의 검색 능력을 극대화하며, 실험 결과가 이를 확인해줍니다.



### ProSec: Fortifying Code LLMs with Proactive Security Alignmen (https://arxiv.org/abs/2411.12882)
Comments:
          The first two authors contributed equally to this work

- **What's New**: 이 논문에서는 코드 생성에 특화된 대형 언어 모델(LLM)의 보안 문제를 해결하기 위한 새로운 접근 방식인 ProSec을 제안합니다. ProSec은 안전한 코드 작성 관행에 맞추어 LLM을 정렬하는 능동적인 보안 정렬 방법론으로, 일반적인 보안 지침을 넘어 취약성을 체계적으로 드러내고 해결합니다. 이 모델은 Common Weakness Enumerations (CWE)를 활용하여 오류 유발 코딩 시나리오를 합성함으로써 LLM을 훈련시킵니다.

- **Technical Details**: ProSec은 실시간 코드 리포지토리에서 취약한 코드 샘플의 희소성을 해결하기 위해, CWE를 기반으로 하는 코딩 시나리오를 합성하여 인 Instruction-tuning 데이터셋을 확장합니다. 모델은 유사 코드 스니펫을 생성하면서도 보안 경고를 이해하고, 피드백을 통해 취약성을 해결하는 방법을 학습합니다. 이 접근 방식은 25배 더 많은 취약 코드 샘플을 생성하고, 종전 데이터셋보다 7배 더 큰 보안 정렬 데이터셋을 생성합니다.

- **Performance Highlights**: ProSec으로 학습한 모델은 이전 SafeCoder 데이터셋으로 학습한 모델보다 29.2%에서 35.5% 더 높은 보안률을 보였습니다. 또한 ProSec의 활용은 모델의 유용성에 미치는 부정적 영향이 2% 포인트 이하로 제한되는 것을 확인했습니다. 이 연구는 코드 LLM의 보안 생성 능력을 높이는 데 기여하며, 여러 모델, 언어, 취약성 유형을 아우르는 전반적인 성능 향상을 증명합니다.



### SCOUT: A Situated and Multi-Modal Human-Robot Dialogue Corpus (https://arxiv.org/abs/2411.12844)
Comments:
          14 pages, 7 figures

- **What's New**: 본 논문에서는 SCOUT(Situated Corpus Of Understanding Transactions)라는 다중 모달리티의 인간-로봇 대화 데이터를 소개합니다. 이 데이터셋은 비접촉식 로봇과의 협력 탐색 작업에서 인간이 로봇에 구두 지시를 내리는 내용을 포함하고 있으며, 총 89,056개의 발화와 310,095개의 단어로 구성되어 있습니다. SCOUT는 실험 중 수집된 다양한 데이터 스트림과 이미지, 지도 데이터를 함께 결합하여 로봇이 환경을 탐색하는 과정을 연구할 수 있도록 설계되었습니다.

- **Technical Details**: SCOUT는 5,785개의 이미지와 30개의 LIDAR 지도가 포함된 데이터셋으로, 각 발화는 말하는 사람의 의도와 의미를 파악하기 위한 Abstract Meaning Representation(AMR) 및 Dialogue-AMR 주석이 추가되어 있습니다. 또한, Transactional Units(TUs)와 발화 간의 관계를 추적하는 주석이 있어 대화 구조의 패턴을 드러낼 수 있습니다. 이러한 방식으로 수집된 데이터는 로봇의 자율적 네비게이션 및 대화 시스템의 개발에 활용될 수 있습니다.

- **Performance Highlights**: SCOUT는 인간-로봇 상호작용(HRI) 및 대화 연구 커뮤니티에 즉각적인 혜택을 제공할 것으로 기대됩니다. 이 데이터셋은 비단순한 문의 및 요청을 포함하여, 로봇과의 실제 대화에서 나타나는 다양한 전략과 패턴을 탐구할 수 있는 기회를 제공합니다. SCOUT 데이터는 오픈소스로 공개되어 있으며, 자율적이고 상황에 맞는 인간-로봇 대화를 연구하는 데 기여할 것입니다.



### Reward Modeling with Ordinal Feedback: Wisdom of the Crowd (https://arxiv.org/abs/2411.12843)
- **What's New**: 본 논문에서는 인간의 선호도를 반영하여 보상 모델을 학습하는 새로운 프레임워크를 제안합니다. 기존의 이진 피드백을 넘어, 다양한 수준의 순서형 피드백(ordinal feedback)을 활용하여 데이터로부터 중요한 정보를 더 효과적으로 추출할 수 있습니다. 특히 이 연구는 선호 간의 미세한 차이를 반영하는 방법을 제시합니다.

- **Technical Details**: 제안된 프레임워크에서는 인간의 선호 피드백을 특정 응답이 다른 응답보다 더 낫다는 확률과 연결짓습니다. 우리는 'wisdom of the crowd'라는 사회학적 개념을 기반으로 한 마진 편향 제로(marginal unbiasedness) 조건을 도입하여 이론적 근거를 마련합니다. 또한, 우리는 이 순서형 피드백 모델의 통계적 이점을 검증하며, Rademacher 복잡성을 줄일 수 있는 방법을 제시합니다.

- **Performance Highlights**: 수치 실험의 결과, 미세한 피드백을 포함한 경우 보상 모델 학습이 개선된다는 것을 확인하였습니다. 4개의 서로 다른 순서형 피드백 시스템을 설정하여 이론적 발견을 검증하였고, 특히 특정 비율의 동점(tied) 샘플을 포함하는 것이 RM 학습을 촉진하는 것으로 나타났습니다. 이러한 결과는 보상 모델의 성능을 크게 향상시킬 가능성을 보여줍니다.



### Human-Robot Dialogue Annotation for Multi-Modal Common Ground (https://arxiv.org/abs/2411.12829)
Comments:
          52 pages, 14 figures

- **What's New**: 본 논문에서는 코로나19와 같은 재난 상황에서 로봇과 인간의 대화 데이터를 활용하여 의미의 차원을 접근 가능하게 하기 위한 기호적 표현 개발을 다룹니다. 특히, 이 과정을 통해 로봇은 사람과의 공통된 이해를 구축할 수 있게 됩니다. 대화 시스템에서 프로포지셔널 의미(propositional semantics)와 발화의 의도(illocutionary force)를 포착하는 새로운 방법론을 제시하며, 다양한 발화 패턴을 분석할 수 있도록 다중 발화 층(multi-floor dialogue structure) 주석 체계를 개발했습니다.

- **Technical Details**: 논문에서 기술한 방식은 대화의 프로포지셔널 의미를 포착하기 위한 Dialogue-AMR 주석을 적용합니다. 이 주석은 Abstract Meaning Representation의 확장으로, 인간과 로봇 간의 대화에서 각 발화 간의 연관성을 분석하는 데 초점을 맞추고 있습니다. 또한, 비주얼 모달리티(visual modalities)가 대화의 상황 정보를 제공하는 방식에 대한 주석과 분석을 진행합니다.

- **Performance Highlights**: 결론적으로, 본 연구는 로봇이 인간과의 양방향 대화 및 탐색을 자율적으로 수행할 수 있도록 하는 다양한 사용 사례, 아키텍처 및 시스템을 제안합니다. 로봇이 재난 구조와 같은 복잡한 상황에서 인간과 효율적으로 협력하기 위해 필요한 여러 의미 차원에 접근할 수 있도록 하는 방법론을 제시합니다. 이로 인해 로봇이 더 나은 팀워크를 이루고 효율적인 의사소통을 할 수 있는 기반을 마련합니다.



### Revisiting Fake News Detection: Towards Temporality-aware Evaluation by Leveraging Engagement Earliness (https://arxiv.org/abs/2411.12775)
Comments:
          WSDM 2025

- **What's New**: 사회적 그래프 기반의 가짜 뉴스 탐지에서 모델이 과거 정보만을 사용하여 훈련되는 현실적인 평가 방식을 제안합니다. 기존 접근 방식들은 미래 데이터에 접근이 가능하여 발생하는 정보 유출로 인해 평가가 비현실적임을 지적하고, 이에 따라 ‘DAWN’이라는 새로운 방법을 개발하였습니다. 이 방법은 사회적 네트워크에서의 사용자 참여 초기 시점을 활용하여 가짜 뉴스 탐지 성능을 향상시킵니다.

- **Technical Details**: DAWN는 그래프 구조적 학습(Graph Structure Learning, GSL) 프레임워크를 사용하고, 사용자와 트윗의 참여 초기 시점에 기반한 특성 표현을 통해 노이즈 엣지의 가중치를 낮추는 방식으로 작동합니다. 이를 통해 더 정확한 구별 능력을 발휘할 수 있도록 설계되었습니다. 또한, 본 연구에서는 참여의 빠름(earliness)과 뉴스의 진위(label consistency)와의 관계를 심층적으로 분석하여, 정보의 진실성을 연결하는 중요한 지표를 제시하였습니다.

- **Performance Highlights**: DAWN은 두 가지 실제 가짜 뉴스 데이터셋에서 기존 방법들에 비해 높은 성능을 기록하였으며, 특히 ‘temporality-aware’ 훈련과 평가 설정 하에서도 튼튼한 성능을 보였습니다. 실험 결과, DAWN은 가짜 뉴스 탐지 분야에서 향상된 성능을 나타내어 강인성을 증명하였습니다. 실험을 통해 사용자 참여 초기정보를 활용하는 것이 가짜 뉴스 탐지에서 얼마나 효과적인지 확인하였습니다.



### A Library Perspective on Supervised Text Processing in Digital Libraries: An Investigation in the Biomedical Domain (https://arxiv.org/abs/2411.12752)
Comments:
          JCD2024 Full Paper, 12 pages, 6 figures

- **What's New**: 이번 연구는 디지털 도서관에서 관계 추출(relation extraction)과 텍스트 분류(text classification) 작업을 통해 훈련 데이터를 생성하고 이를 통해 도서관의 콘텐츠를 강화하는 방법을 논의합니다. 본 연구는 최신 모델을 사용하는 것만큼, 비용과 품질 간의 균형(trade-off)을 중시하며 접근합니다. 또한, ChatGPT와 같은 대형 언어 모델(large language models)들을 사용하여 효율적인 데이터 생성 방법도 제안합니다.

- **Technical Details**: 연구에서는 여러 생물 의학(biomedical) 벤치마크를 사용하여 텍스트에서 명명된 엔티티(named entities)를 추출하고 이들의 관계를 식별하는 방법을 분석합니다. 이를 위해, 일반화(generalizability)를 위한 네 가지 벤치마크에서 실험을 진행하며, 완전한 파이프라인 구현을 위한 시스템 설계를 중요시합니다. 마지막으로, 약한 감독(weak supervision)과 대형 언어 모델을 사용하여 훈련 데이터를 어떻게 주석(labeling)할 수 있는지를 다룹니다.

- **Performance Highlights**: 본 연구에서 BERT 기반 모델이 다른 모델에 비해 높은 성능을 보여주며, 단일 모델이 여러 작업을 동시에 처리할 수 있는 방법에 대한 논의가 포함됩니다. 연구팀은 다양한 비용 및 품질 요소를 고려하여, 기존 모델과 최신 언어 모델 사용 간의 균형을 찾아내고자 합니다. 결국, 디지털 도서관에서의 데이터 처리 작업이 비용 효율적이면서도 높은 품질을 유지할 수 있는 방안을 제안합니다.



### Neon: News Entity-Interaction Extraction for Enhanced Question Answering (https://arxiv.org/abs/2411.12449)
- **What's New**: 이 논문에서는 NEON 프레임워크를 제안하여, 급변하는 정보 환경에서 최신 정보를 캡처하고 이를 기반으로 LLM의 출력을 보강하는 방법을 다루고 있습니다. 이 프레임워크는 뉴스 기사에서 나온 최신 엔터티 상호작용을 추출하여 시계열 기반의 지식 그래프를 생성합니다. 이러한 접근은 정보 검색의 정확성을 높이기 위해, 정보 추출(openIE) 스타일의 튜플을 LLM과 통합합니다.

- **Technical Details**: NEON은 뉴스 스트림에서 엔터티 간의 상호작용을 모델링하는 openIE 스타일의 지식 그래프입니다. 이 그래프는 엔터티를 노드로, 상호작용을 엣지로 구성되며, 시간 프레임을 포함하여 각 엔터티 간의 이전 및 현재의 상호작용에 대한 타임스탬프를 기록합니다. 최적화된 색인 기법을 통해 효과적인 시계열 정보 검색을 가능하게 하여, 사용자의 임시 질문에 대한 정확하고 관련 있는 응답을 생성합니다.

- **Performance Highlights**: 실험을 통해 NEON을 LLM 프롬프트에 통합하는 것이 응답의 시계열적 관련성과 품질을 향상시킨다는 것을 보여주었습니다. 3,000개의 실제 쿼리를 분석한 결과, NEON 튜플을 포함한 질문이 더 높은 품질의 답변을 생성했으며, 자동 평가 및 인간 평가 모두에서 유의미한 개선이 있었습니다. 이러한 결과는 NEON이 정보 중심 질문에 대한 효율적인 해결책임을 입증합니다.



### SRA-MCTS: Self-driven Reasoning Augmentation with Monte Carlo Tree Search for Enhanced Code Generation (https://arxiv.org/abs/2411.11053)
- **What's New**: 이 논문에서는 복잡한 문제를 해결하기 위한 새로운 데이터 생성 프로세스인 SRA-MCTS(Reasoning-Augmented Monte Carlo Tree Search)를 제안하고 있습니다. 이 방법은 모델이 자율적으로 고급 중간 추론 경로를 생성하도록 세심하게 안내합니다. 결과적으로, 이 프로세스는 지속적인 개선을 위한 긍정적인 피드백 루프를 생성하여 자연어 추론 경로를 실행 가능한 코드로 변환합니다.

- **Technical Details**: SRA-MCTS는 세 가지 주요 단계로 구성된 데이터 생성 파이프라인을 제안합니다: 계획 생성, 계획을 코드로 변환, 모델 학습입니다. 이 과정의 핵심은 Monte Carlo Tree Search(MCTS)의 가지 선택 메커니즘을 활용하여 문제의 최적 해결책을 선택하도록 하는 것입니다. 이를 통해 다양한 솔루션을 생성하고 각 단계에서 올바른 프로세스를 선택하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, SRA-MCTS를 통해 생성된 데이터로 미세 조정된 모델은 기존 모델 구조 및 CoT를 기반으로 한 미세 조정 모델보다 우수한 성능을 보였습니다. 특히, 작은 모델에서도 자가 개선의 잠재력이 크다는 것을 발견했습니다. CoT가 성능 저하를 보이는 경우에도 SRA-MCTS의 접근 방식은 더욱 개선된 다양성 메트릭을 보여주었습니다.



New uploads on arXiv(cs.IR)

### Unleashing the Power of Large Language Models for Group POI Recommendations (https://arxiv.org/abs/2411.13415)
- **What's New**: 이번 연구에서 제안하는 LLMGPR( Large Language Model 기반 Group Point-of-Interest 추천 시스템)은 기존의 Single ID 기반 특징을 포기하고, POI와 관련된 풍부한 의미 정보를 활용하여 그룹 사용자들의 다양한 선호를 만족하는 향상된 추천을 제공합니다. 이 시스템은 POI 토큰과 원래의 단어 토큰을 결합하여 POI에 대한 의미 강화된 표현을 초기화하고, 그룹 결정을 위한 복잡한 요인을 모델링합니다.

- **Technical Details**: LLMGPR는 Quantized Low-Rank Adaptation(QLORA) 기반의 새로운 시퀀스 어댑터를 도입하여, POI 토큰과 고려된 위치 인코딩 및 시공간적 차이를 조합하여 시퀀스 표현을 학습합니다. 이 시스템은 특정 시퀀스 유형에 따라 그룹이나 사용자 표현을 학습할 수 있으며, 개인 구성원의 표현을 집계하는 아키텍처를 포함하여 데이터 희소성을 완화합니다.

- **Performance Highlights**: 실험 결과, LLMGPR는 기존의 방법보다 뛰어난 성능을 보였으며, 그룹 수준의 체크인 데이터 희소성 문제를 효과적으로 해결하여 더 정확한 POI 추천을 제공합니다. 이 연구는 그룹 POI 추천의 맥락에서 사전 훈련된 대형 언어 모델을 통합한 최초의 사례로 기록되며, POI 추천 분야의 연구에 중요한 기여를 할 것으로 기대됩니다.



### Scaling Laws for Online Advertisement Retrieva (https://arxiv.org/abs/2411.13322)
Comments:
          10 pages, 8 figures

- **What's New**: 본 연구에서는 온라인 광고 검색 시스템에서의 스케일링 법칙(scaling law)을 식별하기 위한 경량화 혁신(paradigm)을 제안합니다. 이러한 접근 방식은 최소한의 실험 비용으로 온라인 수익(online revenue)과 기계 비용(machine cost)의 관계를 이해하는 데 중점을 둡니다. 특히, R/R*라는 새로운 오프라인 메트릭(offline metric)을 도입하여 검색 모델의 온라인 수익과 높은 선형 상관관계를 보이고 있습니다. 이는 광고 시스템 최적화의 가능성을 보여줍니다.

- **Technical Details**: 연구진은 R/R* 메트릭과 FLOPs(부동 소수점 연산 수)를 사용하여 오프라인 실험을 수행하고 MLP 모델의 스케일링 행동을 검증했습니다. 결과적으로, 이 스케일링 행동은 파손된 신경 스케일링 법칙(broken neural scaling law)을 따르는 것으로 나타났습니다. 또한, 기계 비용을 추정하기 위한 간단한 시뮬레이션 알고리즘을 제안하여 기계 비용과 온라인 수익 간의 관계를 설정하였습니다. 이를 통해 실험 비용을 크게 줄일 수 있었습니다.

- **Performance Highlights**: 이 연구의 주요 기여는 R/R* 메트릭이 온라인 광고 시스템의 수익을 예측하는 데 있어 뛰어난 오프라인 대리 메트릭 surrogate metric임을 입증한 것입니다. 이 메트릭을 활용함으로써 ROI 제약에 따른 모델 설계 및 다양한 시나리오에서 자원을 효율적으로 할당하였고, 각각 0.85%와 2.8%의 온라인 수익 개선을 달성했습니다. 이러한 결과는 광고 시스템 최적화에 있어 스케일링 법칙의 잠재력을 뒷받침합니다.



### On the Statistical Significance with Relevance Assessments of Large Language Models (https://arxiv.org/abs/2411.13212)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 생성한 레이블이 인간의 판단과 동일한 쌍별 통계적 유의성을 유지할 수 있는지를 분석합니다. LLM이 제공하는 자동 relevance 판단이 정보 검색(IR) 평가에서 유용함을 증명하고자 하며, 이는 통계적 테스트 결과를 평가하는 새로운 기틀을 제공합니다.

- **Technical Details**: 연구는 LLM이 제공한 판단(QLLM)과 인간 전문가의 판단(QH) 간의 유의성 결정을 비교합니다. 각 시스템 쌍에 대해 평균 정밀도(Average Precision)를 사용하여 p-value를 계산한 후, 두 서로 다른 평가에 대한 유의성 결정을 분석합니다. 이 과정에서 유의한 결과를 나타낸 양성(True Positive), 거짓 양성(False Positive) 등의 결과를 측정합니다.

- **Performance Highlights**: 연구 결과, LLM 판단이 대부분의 유의미한 차이를 탐지하여 받아들일 만한 수준의 거짓 긍정을 유지함을 보여주었습니다. 그러나 몇몇 시스템은 LLM 레이블 아래에서 공정하게 평가되지 않는다는 점도 발견되었습니다. 이러한 결과는 LLM의 유용성을 지지하면서도 도전 과제를 함께 제시하고 있어, 향후 연구에 대한 중요한 기초 자료를 제공할 것으로 기대됩니다.



### Writing Style Matters: An Examination of Bias and Fairness in Information Retrieval Systems (https://arxiv.org/abs/2411.13173)
Comments:
          In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining (WSDM 25)

- **What's New**: 본 논문에서는 정보 검색 시스템(Information Retrieval, IR) 내에서 최신 보편적 텍스트 임베딩 모델의 문서 및 쿼리 작성 스타일에 따른 잠재적 편향을 탐색합니다. 특히, 비공식적이고 감정적으로 표현된 스타일은 대부분의 임베딩 모델에서 적게 선호됨을 밝혔습니다. 이 연구는 다양한 작성 스타일이 정보 검색 시스템에서의 편향과 공정성에 미치는 영향을 집중적으로 분석합니다.

- **Technical Details**: 정보 검색 시스템은 쿼리에 대한 관련 데이터를 검색하는 것이 핵심 목표이며, 이 과정에서 문서의 임베딩 유사도 점수가 사용됩니다. 최근 LLM(대형 언어 모델)의 부상은 IR 시스템 내에서의 편향 문제를 더욱 복잡하게 만들었으며, 편향 및 공정성의 문제는 전반적인 시스템의 신뢰성을 저하시킬 수 있습니다. 이 논문은 이러한 편향을 분석하여 공정성을 측정하는 새로운 방법을 제안합니다.

- **Performance Highlights**: 논문에서는 텍스트 임베딩 모델들이 서로 다른 작성 스타일에 대해 어떻게 편향되어 있는지를 분석하였으며, LLM의 답변 스타일에 대한 편향 또한 식별하였습니다. 이 연구는 IR 시스템의 공정하고 강력한 모델 개발에 중요한 통찰력을 제공합니다. 다양한 최신 텍스트 임베딩 모델을 비교하며, 쿼리 스타일의 영향 역시 고려하였습니다.



### DMQR-RAG: Diverse Multi-Query Rewriting for RAG (https://arxiv.org/abs/2411.13154)
- **What's New**: 본 논문에서는 DMQR-RAG라는 다중 쿼리 리라이트(Rewrite) 프레임워크를 소개하여 문서 검색 및 최종 응답 개선을 위해 설계되었습니다. 기존의 쿼리 리라이트 방법들이 단일 쿼리만 생성하는 문제를 해결하기 위해, 다양한 정보량을 가진 여러 개의 쿼리를 통해 문서의 다양성을 증가시킴으로써 RAG의 성능을 개선하는 데 중점을 두었습니다. 또한, 리라이트 전략 선택 방법을 통해 리라이트 수를 최소화하면서 최적의 성능을 달성할 수 있는 방안을 제안합니다.

- **Technical Details**: DMQR-RAG 프레임워크는 정보의 양에 따라 작동하는 네 가지 리라이트 전략을 기반으로 다양성을 증대시키고, 크로스-어텐션(Attention) 임베딩 모델을 사용하여 검색된 문서들을 재순위(Rerank)합니다. 각 리라이트 쿼리는 서로 다른 문서를 검색할 수 있는 잠재력을 가지고 있으며, 재작성 전략 선택 방법을 통해 최적의 리라이트 전략을 동적으로 식별합니다. 이러한 접근은 특정 쿼리에 적합한 전략을 선택하여 효과적인 문서 검색과 응답 생성의 성능을 향상시킵니다.

- **Performance Highlights**: 다중 쿼리 리라이트 방식은 단일 쿼리 리라이트 방식에 비해 일반적으로 더 나은 성능을 보여줍니다. 연구 결과, 정보 기반의 다중 쿼리 접근 방식이 기존의 RAG-Fusion 방법을 초월하는 경우가 많았습니다. 본 연구에서 제안된 방법들은 둘 다 학술 및 산업 환경에서 철저히 실험을 통해 검증되어 효과성이 입증되었습니다.



### Branches, Assemble! Multi-Branch Cooperation Network for Large-Scale Click-Through Rate Prediction at Taobao (https://arxiv.org/abs/2411.13057)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 MBCnet이라는 새로운 다중 분기 협력 네트워크를 소개합니다. 이 네트워크는 다양한 피쳐 상호작용 모델링 능력을 갖춘 여러 개의 브랜치 네트워크가 서로 협력하여 복잡한 피쳐 관계를 더욱 잘 캡처할 수 있도록 설계되었습니다. MBCnet은 전문가 기반 피쳐 그룹화 및 크로싱(Expert-based Feature Grouping and Crossing, EFGC) 브랜치와 로우 랭크 크로스 네트워크(low rank Cross Net) 및 딥 브랜쳐로 구성됩니다. 이러한 새로운 협력 구조가 기존의 단일 접근 방식의 한계를 극복하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: MBCnet은 세 개의 주요 브랜치로 구성되며, 각 브랜치는 고유한 기능과 모델링 능력을 가지고 있습니다. EFGC 브랜치는 특정 피쳐 필드를 그룹화하고 도메인 전문가의 지식을 이용해 해당 그룹 간의 피쳐 상호작용을 수행하여 모델의 기억 능력을 증대시킵니다. 또한, 기존의 로우 랭크 크로스 네트워크와 딥 브랜치는 명시적 및 암시적 피쳐 크로싱을 통해 모델의 일반화 능력을 강화하게 됩니다. 다중 브랜치 협력 체계에서는 잘 학습된 브랜치가 그렇지 않은 브랜치를 지원하도록 하는 '브랜치 공동 교육(branch co-teaching)'과 적절한 차별화를 유지하는 원칙이 적용됩니다.

- **Performance Highlights**: MBCnet은 대규모 산업 데이터셋에서 진행한 실험을 통해 Click-Through Rate (CTR)에서 0.09 포인트 상승을 기록하였으며, 거래 수는 1.49% 증가하고 GMV는 1.62% 상승하였습니다. 이러한 성과는 MBCnet의 협력적 학습이 다양한 피쳐 상호작용을 효과적으로 발견하는 데 기여했음을 보여줍니다. 또한, Taobao 앱의 image2product 검색에 MBCnet이 배포되어 현장에서의 꼭 필요한 성과를 입증했습니다.



### On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspectiv (https://arxiv.org/abs/2411.13052)
- **What's New**: 이 논문에서는 Content-based Recommender Systems (CRSs)의 임베딩 테이블이 저장 용량에서 중요한 병목 현상을 초래하는 문제를 해결하기 위해 Shapley Value-guided Embedding Reduction (Shaver)라는 새로운 접근 방식을 제안합니다. 기존의 임베딩 가지치기 방법은 각 목표 매개변수 예산을 위해 재학습이 필요하여 높은 계산 비용을 초래하는데, 본 연구는 이러한 문제에 혁신적인 해결책을 제공합니다.

- **Technical Details**: Shaver는 협력적 게임 관점에서 문제를 바라보고 각 임베딩 매개변수의 기여도를 Shapley values로 정량화하여 기여 기반 매개변수 가지치기를 촉진합니다. 또한 Shapley values의 계산 비용을 줄이기 위해 효율적이고 편향되지 않은 방법을 제안하며, 가지치기 단계에서는 전통적인 zero-out 처리를 보완하기 위한 field-aware codebook을 도입하여 정보 손실을 최소화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 Shaver는 다양한 매개변수 예산에서 경량 추천 모델과 경쟁력 있는 성능을 보여주었습니다. 이 연구는 federated learning 및 스트리밍 환경과 같은 다양한 저장 요구 사항을 가진 실제 응용 프로그램에서의 효율성을 제공하여 주목할 만한 기여를 하고 있습니다.



### Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning (https://arxiv.org/abs/2411.13045)
Comments:
          Submitted to WWW 2025

- **What's New**: 이번 논문에서는 전자상거래 검색 시스템에서 사용자 경험과 만족도를 향상시키기 위해, 효과적인 query-item relevance(쿼리-아이템 관련성) 모델링의 중요성을 강조합니다. 최근에 등장한 Large Language Model(대규모 언어 모델) 접근법은 기존 신경망 기반의 전문 관련성 학습 방법에 비해 뛰어난 성능과 긴 꼬리(generalization) 능력을 보여주고 있습니다. 하지만 이 모델들은 온라인 배포에 어려움이 있으며, LLM의 복잡한 내부 지식을 추출하고 적용하는 데 한계가 있습니다.

- **Technical Details**: 이를 해결하기 위해 우리는 Explainable LLM-driven Multi-dimensional Distillation(설명 가능한 대규모 언어 모델 기반 다차원 증류) 프레임워크를 제안합니다. 이 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) ELLM-rele(설명 가능한 LLM)로서 관련성 모델링을 Chain-of-Thought(사고의 흐름) 추론으로 분해하여 해석 가능성과 성능을 향상시킵니다. 2) MKD(다차원 지식 증류) 아키텍처는 ELLM-rele의 지식을 현재 배포 가능한 모델에 전달하며, 이는 관련성 점수 분포와 CoT 추론 측면에서 이루어집니다.

- **Performance Highlights**: 우리는 이 프레임워크가 Taobao 검색 광고 환경에서의 방대한 오프라인 평가와 온라인 실험을 통해 전자상거래 관련성 학습 성능과 사용자 경험을 크게 개선함을 보였습니다. MKD는 학생 모델의 의미적 상호작용과 긴 꼬리 일반화 능력을 모두 향상시키는 데 기여하였습니다. 이러한 결과는 LLM의 해석 가능성과 모델 성능 향상이라는 두 가지 주요 목표를 달성함으로써, 향후 전자상거래 시스템에서 더욱 효과적인 적용이 가능함을 시사합니다.



### Data Watermarking for Sequential Recommender Systems (https://arxiv.org/abs/2411.12989)
- **What's New**: 최근 대규모 기초 모델의 발전으로 데이터가 고성능 AI 시스템 구축에 있어 필수 요소로 떠오르고 있습니다. 정보 보호와 무단 사용 방지에 대한 요구가 증가함에 따라, 본 연구에서는 Sequential Recommender Systems를 위한 데이터 워터마킹(data watermarking) 문제를 탐구합니다. 우리는 데이터 세트의 소유권을 보호하는 Dataset Watermarking과 개별 사용자의 데이터를 보호하는 User Watermarking이라는 두 가지 도전 과제를 제시하고, 이를 해결하기 위한 DWRS 방법론을 소개합니다.

- **Technical Details**: 제안된 DWRS(Data Watermarking for Recommender Systems) 방법론은 대량의 사용자-아이템 상호작용 데이터를 사용하여 워터마크를 삽입합니다. 구체적으로, 자주 사용되지 않는 임의의 아이템들을 선택하여 워터마크 시퀀스를 생성하고, 이를 정상 사용자 상호작용 데이터에 삽입합니다. DWRS는 두 가지 변형으로 나뉘며, DWRS-D는 전체 데이터세트를 워터마킹하는 데 사용되고, DWRS-U는 특정 사용자의 상호작용 데이터에 삽입됩니다.

- **Performance Highlights**: DWRS에 대한 광범위한 실험 결과, 제안된 방법이 데이터 저작권 보호에 있어 높은 유효성을 발휘하면서 모델의 유용성을 유지하는 것으로 나타났습니다. 다섯 개의 대표적인 Sequential Recommender System과 세 개의 벤치마크 데이터셋에서 검증된 DWRS는 데이터 소유자가 소유권을 주장하고 무단 사용을 피할 수 있도록 합니다. 이러한 결과는 데이터 워터마킹이 권리를 보호하고 데이터 안전성을 보장하는데 중요한 역할을 할 수 있음을 보여줍니다.



### A Comparative Study of Text Retrieval Models on DaReCzech (https://arxiv.org/abs/2411.12921)
- **What's New**: 이 논문은 체코어 데이터셋 DaReCzech에서 7개의 상용 문서 검색 모델의 성능을 포괄적으로 평가한 결과를 제시합니다. 모델들은 Splade, Plaid, Plaid-X, SimCSE, Contriever, OpenAI ADA, Gemma2로 구성되며, 체코어 검색 접근 방식의 품질을 추정하는 것이 주요 목적입니다. 실험 결과, Gemma2가 가장 높은 정밀도와 재현율을 보였으며, SPLADE와 PLAID 모델은 효율성과 성능의 균형을 잘 이루었습니다.

- **Technical Details**: 제안된 연구에서는 다양한 정보 검색(IR) 모델들로부터 인덱스 크기, 검색 속도 및 메모리 사용량 등을 분석합니다. 각각의 모델을 체코 문서와 쿼리에 대해 평가하며, 베이스라인 모델인 BM25를 사용하여 다른 모델의 효과를 평가합니다. 모델들이 체코어 데이터셋과 영어 번역에서 어떻게 성능을 발휘하는지 비교 분석하였고, 특히 지수의 크기와 검색 속도가 큰 데이터셋에서의 스케일링에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험을 통해 저자들은 각 모델 간의 성능 차이를 명확히 식별했습니다. Gemma2는 정밀도와 재현율 모두에서 뛰어난 결과를 보였으나, Contriever는 저조한 성능을 나타냈습니다. SPLADE와 PLAID 모델은 효율성과 성능을 모두 고려할 수 있는 좋은 선택으로 밝혀졌습니다. 이번 연구는 체코어의 정보 검색 방법론의 심층 비교를 제공하는 첫 번째 사례로서, 향후 연구에 중요한 이정표가 될 것입니다.



### Advancing Large Language Models for Spatiotemporal and Semantic Association Mining of Similar Environmental Events (https://arxiv.org/abs/2411.12880)
- **What's New**: 이번 논문은 최신 검색 도구에서 필수적인 두 가지 작업인 Retrieval(검색)과 Recommendation(추천)을 개선하기 위한 새로운 Retrieval-Reranking(재정렬) 프레임워크를 소개합니다. 이 프레임워크는 Large Language Models (LLMs)을 활용하여 뉴스 기사와 웹 포스트에 설명된 비정상적인 기후 및 환경 사건을 더 효과적으로 검색하고 추천할 수 있도록 설계되었습니다. 전통적인 수작업 큐레이션 방법의 고비용과 비확장성 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방안은 고급 자연어 처리 (Natural Language Processing) 기법을 사용하여 시공간(spatiotemporal) 사건을 분석하는 최적화된 솔루션을 탐구합니다. 또한, 공간적 근접성(spatial proximity), 시간적 연관성(temporal association), 의미 유사성(semantic similarity), 카테고리 기반 유사성(category-instructed similarity)을 포함한 다각적 기준을 통합한 Geo-Time Re-ranking (GT-R) 전략을 제안합니다. 이 모델은 4000개 Local Environmental Observer (LEO) Network 사건 데이터셋에 적용되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 다수의 최신 Dense Retrieval 모델들 사이에서 비슷한 사건 추천에 있어 최고의 성능을 달성했습니다. 이 검색 및 추천 파이프라인은 지리적(geospatial) 및 시간적(temporal) 데이터와 관련된 다양한 검색 작업에도 적용될 수 있습니다. 우리는 관련 사건을 연결함으로써 대중이 기후 변화와 그로 인한 다양한 커뮤니티에 미치는 영향을 더 잘 이해할 수 있도록 지원할 수 있기를 희망합니다.



### PatentEdits: Framing Patent Novelty as Textual Entailmen (https://arxiv.org/abs/2411.13477)
- **What's New**: 본 연구에서는 창작권 확보를 위해 필요한 특허 수정 예측 문제를 새로운 학습 가능한 작업으로 설정하고, 이를 위한 PatentEdits 데이터셋을 소개합니다. 이 데이터셋은 105,000개의 수정 예시를 포함하고 있으며, 기존 문서 수정 예측 연구와 달리 주목할 만한 이전 문헌과의 관계를 고려합니다. 특히, 대형 언어 모델(LLM)을 활용하여 인용된 참조와 초안 문장 간의 텍스트 포함 관계를 평가하는 방식을 제안합니다.

- **Technical Details**: PatentEdits 데이터셋은 2007년부터 2014년까지의 미국 공인 발명 특허 105,000건으로 구성되어 있습니다. 데이터셋에 포함된 초안, 인용된 참조 및 최종 특허 텍스트를 정렬하여 어떤 문장이 Kept, Edited 또는 Deleted 되는지를 자동적으로 라벨링하는 알고리즘을 설계했습니다. 향상된 클레임 예측을 위해 LLM을 활용하여 인용된 문장을 긍정 예로, 초안 문장을 앵커로, 최종 문장을 부정 예로 설정하고, triplet loss로 파인튜닝합니다.

- **Performance Highlights**: 실험 결과, 인용된 참조를 포함시키고 초안 특허 클레임과 인용된 특허 클레임 간의 포함 관계에 집중할 경우, 특허 클레임의 새로움을 평가하는 데 효과적임을 보여주었습니다. 특히, BLEU-4 및 METEOR와 같은 다양한 유사도 평가 방법을 활용하여 초안 및 최종 문장 간의 문서 매칭 품질을 검증하였습니다. 이러한 접근 방식은 특허 출원자와 미국 특허 심사원이 필요한 수정을 예측하는 데 실질적인 기여를 할 수 있을 것으로 기대됩니다.



### Epidemiology-informed Network for Robust Rumor Detection (https://arxiv.org/abs/2411.12949)
- **What's New**: 이 논문에서는 전염병학 지식( Epidemiology-informed )을 통합하여 소셜 미디어에서의 루머 탐지 성능을 향상시키는 새로운 네트워크 모델인 Epidemiology-informed Network (EIN)을 제안합니다. 기존의 그래프 기반 방법들이 데이터 품질에 민감한 문제를 해결하기 위해 이 접근 방식은 대규모 언어 모델을 활용해 사용자 입장을 자동으로 주석을 추가하는 방안을 보여줍니다. 이 모델은 다양한 깊이의 전파 트리( propagation tree )에서 향상된 강건성을 나타냅니다.

- **Technical Details**: EIN 모델은 정보 전파 패턴을 포착하기 위해 그래프 신경망( Graph Neural Networks )을 활용하며, 사용자의 소스 정보에 대한 입장을 주석하여 데이터 기반 설계의 한계를 극복합니다. 일반적인 사례에서, 얕은 전파 트리는 제한된 상호작용만을 가지고 있어 충분한 전파 패턴을 포착하기 어렵기 때문에 주목받지 못하는 뉴스나 초기 탐지를 처리하는 데 한계를 가집니다. EIN은 이러한 문제를 해결하고 퍼포먼스를 향상시킵니다.

- **Performance Highlights**: EIN은 실제 데이터셋에서 최신 방법들과 비교해 명확한 성능 향상을 보였으며, 다양한 트리 깊이에서도 안정적인 결과를 나타냈습니다. 실험 결과는 EIN이 기본적인 전파 트리 구조의 고유한 장점을 잘 활용하고 있음을 보여주며, 노이즈가 많은 사용자 반응 환경에서도 성능 저하가 없음을 입증합니다. 따라서 이 연구는 루머 탐지 분야에서 데이터 기반 접근 방식의 한계를 극복할 수 있는 가능성을 제시합니다.



### SEFD: Semantic-Enhanced Framework for Detecting LLM-Generated Tex (https://arxiv.org/abs/2411.12764)
- **What's New**: 본 논문은 대용량 언어 모델(LLM)이 생성한 텍스트를 탐지하기 위한 새로운 접근법인 개선된 의미 기반 프레임워크(SEFD)를 소개합니다. 이 프레임워크는 기존 탐지 방법론에 비해 더 robust한 성능을 제공하며, 특히 paraphrasing 기법을 통한 텍스트 변형에 효과적으로 대응합니다. 텍스트의 의미를 충분히 활용하기 위해 검색 기반 메커니즘을 통합한 것이 특징입니다.

- **Technical Details**: SEFD 프레임워크는 초기 탐지 단계, 의미 유사성 계산, 의미가 강화된 탐지의 세 가지 주요 단계를 포함합니다. 초기 탐지를 위해 기존 탐지기를 사용하고, BERT 기반 모델을 통해 검색 풀 내 텍스트와의 의미 유사도를 평가합니다. 마지막으로, 탐지 점수와 유사성 점수를 통합하여 최종 점수를 도출하며, 이 과정에서 검색 풀도 지속적으로 업데이트됩니다.

- **Performance Highlights**: 본 연구에서는 SEFD 프레임워크가 실제 환경에서 흔히 발생할 수 있는 순차 텍스트 시나리오에서의 효율성을 입증합니다. 다양한 LLM 생성 텍스트와 탐지 방법을 사용한 실험을 통해, paraphrasing 상황에서도 탐지 정확성이 크게 향상됨을 보여주었습니다. 이로 인해 데이터의 무결성을 보장하고 신뢰할 수 있는 정보 제공에 기여할 것으로 기대됩니다.



### FedCL-Ensemble Learning: A Framework of Federated Continual Learning with Ensemble Transfer Learning Enhanced for Alzheimer's MRI Classifications while Preserving Privacy (https://arxiv.org/abs/2411.12756)
Comments:
          6 pages, 4 figures

- **What's New**: 이 연구는 알츠하이머 질병(Alzheimer's disease)의 분류(classification)를 위한 새로운 접근법을 제안합니다. 고급 딥 러닝 기술(deep learning techniques)을 활용하고, 안전한 데이터 처리 방법을 결합하여 향상된 성능을 보이는 모델을 개발했습니다. 특히, 전이 학습 모델(transfer learning models)을 사용하여 의료 이미지 데이터에서 고급 특징을 추출하는 데 중점을 두었습니다.

- **Technical Details**: 주요 기술로는 ResNet, ImageNet, VNet과 같은 전이 학습 모델을 사용하여 알츠하이머 관련 미세한 패턴을 감지할 수 있습니다. 이 모델은 데이터 소스의 다양성에 대해 강력한 특징을 추출할 수 있도록 조정되었습니다. 또한, 페더레이션 학습(federated learning) 접근법을 통합하여 데이터 개인 정보 보호를 보장하면서 분산된 모델의 이점을 최대한 활용합니다.

- **Performance Highlights**: 실험 결과는 알츠하이머 분류의 정확성을 향상시키는 데 기여하며, 안전하고 협력적인 건강 관리 데이터 분석을 위한 프레임워크도 제공합니다. 추가적으로, 데이터 전송 시 기밀성과 무결성을 보장하기 위해 암호 기반 encryption 메커니즘도 적용되었습니다. 이러한 결과는 예측 성능을 개선하고 환자 데이터를 공유하지 않으면서도 모델의 강력한 학습을 가능하게 합니다.



### Neon: News Entity-Interaction Extraction for Enhanced Question Answering (https://arxiv.org/abs/2411.12449)
- **What's New**: 이 논문에서는 NEON 프레임워크를 제안하여, 급변하는 정보 환경에서 최신 정보를 캡처하고 이를 기반으로 LLM의 출력을 보강하는 방법을 다루고 있습니다. 이 프레임워크는 뉴스 기사에서 나온 최신 엔터티 상호작용을 추출하여 시계열 기반의 지식 그래프를 생성합니다. 이러한 접근은 정보 검색의 정확성을 높이기 위해, 정보 추출(openIE) 스타일의 튜플을 LLM과 통합합니다.

- **Technical Details**: NEON은 뉴스 스트림에서 엔터티 간의 상호작용을 모델링하는 openIE 스타일의 지식 그래프입니다. 이 그래프는 엔터티를 노드로, 상호작용을 엣지로 구성되며, 시간 프레임을 포함하여 각 엔터티 간의 이전 및 현재의 상호작용에 대한 타임스탬프를 기록합니다. 최적화된 색인 기법을 통해 효과적인 시계열 정보 검색을 가능하게 하여, 사용자의 임시 질문에 대한 정확하고 관련 있는 응답을 생성합니다.

- **Performance Highlights**: 실험을 통해 NEON을 LLM 프롬프트에 통합하는 것이 응답의 시계열적 관련성과 품질을 향상시킨다는 것을 보여주었습니다. 3,000개의 실제 쿼리를 분석한 결과, NEON 튜플을 포함한 질문이 더 높은 품질의 답변을 생성했으며, 자동 평가 및 인간 평가 모두에서 유의미한 개선이 있었습니다. 이러한 결과는 NEON이 정보 중심 질문에 대한 효율적인 해결책임을 입증합니다.



New uploads on arXiv(cs.CV)

### REDUCIO! Generating 1024$\times$1024 Video within 16 Seconds using Extremely Compressed Motion Latents (https://arxiv.org/abs/2411.13552)
Comments:
          Code available at this https URL

- **What's New**: 이 논문에서 제시된 Reducio 방식은 비디오 생성을 위해 아직도 높은 비용과 계산 자원의 장벽을 줄이는 혁신적인 접근을 보여줍니다. 기존의 2D VAE보다 64배 더 작은 latents를 사용하여 비디오를 압축하는 방법을 제안합니다. 추가적으로, Reducio-DiT라는 새로운 모델은 고품질 비디오를 보다 효율적으로 생성할 수 있도록 설계되었습니다.

- **Technical Details**: 논문에서는 이미지 조건부 VAE(Image-conditioned VAE)를 설계하여 비디오를 매우 압축된 motion latent 공간에 인코딩합니다. 이 구조는 4096×4096×4096으로 down-sampling하며, 3D convolution은 비디오의 중복 정보를 활용하여 reconstruct 성능을 유지합니다. Reducio 모델은 데이터 축소를 통해 고해상도(1K) 비디오 생성이 가능하게 합니다.

- **Performance Highlights**: Reducerio-DiT는 UCF-101 데이터셋에서 318.5의 FVD 점수를 기록하였으며, Lavie 모델보다 16.6배 더 빠른 속도로 1024×1024 해상도의 16프레임 비디오를 15.5초 만에 생성할 수 있습니다. 이 모델은 낮은 GPU 자원에서도 우수한 성능을 보여주며, 비디오 LDMs의 효율성을 현저하게 증가시킵니다.



### Find Any Part in 3D (https://arxiv.org/abs/2411.13550)
Comments:
          Project website: this https URL

- **What's New**: 본 연구는 3D 객체에서 임의의 텍스트 쿼리에 기반하여 어떤 부품이든 분할하는 개방형 세계(open-world) 부품 분할 방법을 제안합니다. 기존 방법은 객체 카테고리와 부품 어휘에서 제한적이었지만, 최신 AI 발전을 통해 2D에서 효과적인 개방형 인식 기능을 달성했습니다. 우리는 'Find3D'라는 모델을 통해 인터넷에서 수집한 대규모 3D 자산으로 훈련한 일반 카테고리 포인트 임베딩 모델을 소개합니다.

- **Technical Details**: 'Find3D'는 데이터 엔진과 대조적 훈련 방법을 결합하여 3D 서브젝트에 대한 쿼리 가능한 의미적 특성을 예측합니다. 이 모델은 46.2M 파라미터로, 27K 레이블이 지정된 객체를 활용하여 3D 포인트 클라우드에서 임의의 객체의 모든 부품을 찾을 수 있도록 설계되었습니다. 핵심 기술은 CLIP와 유사한 모델을 기반으로 하여 2D 비전과 언어 모델을 활용해 3D 객체를 자동으로 라벨링하는 데이터 엔진입니다.

- **Performance Highlights**: 우리의 모델은 기존 방법보다 mIoU에서 최대 3배의 향상을 달성했으며, 추론 속도는 6배에서 300배까지 향상되었습니다. 이러한 성능 개선은 다양한 데이터세트를 통한 일반화의 강점을 보여줍니다. 또한, 개방형 3D 부품 분할 연구를 장려하기 위해 다양한 객체와 부품을 평가할 수 있는 벤치마크를 출시하였습니다.



### Generating 3D-Consistent Videos from Unposed Internet Photos (https://arxiv.org/abs/2411.13549)
- **What's New**: 이 논문에서는 비구조적인 인터넷 사진에서 비디오를 생성하는 문제를 다룹니다. 기존의 비디오 모델은 이러한 작업에서 한계가 있으며, 본 논문은 스스로 감독 학습(self-supervised learning) 방법을 설계하고 3D 인식 능력을 갖춘 비디오 모델을 구현합니다. 이 모델은 2D 데이터(예: 동영상 및 멀티뷰 인터넷 사진)만으로도 3D 학습을 확장할 수 있음을 제시합니다.

- **Technical Details**: 논문에서는 두 가지 주요 목표를 설정합니다. 첫 번째는 멀티뷰 인페이팅(multiview inpainting)으로, 이는 3D 주석 없이 3D 프라이어를 학습하게 합니다. 두 번째는 뷰 인터폴레이션(view interpolation)으로, 이는 주어진 시작 및 종료 프레임 사이의 중간 프레임을 생성할 수 있게 합니다. 이 두 가지 목표는 모델이 정확하고 일관된 카메라 경로를 생성할 수 있도록 합니다.

- **Performance Highlights**: 저자들은 제안한 방법이 기존 최첨단 비디오 생성 모델보다 뛰어난 성능을 보인다고 주장합니다. 사용자 연구를 통해 모델의 기하학적 및 외관 일관성을 검증하였으며, 생성된 비디오는 3D 모델로 변환될 수 있어 다양한 응용이 가능함을 보여줍니다. 결과적으로 이 모델은 카메라 제어를 요구하는 작업에 유용한 결과를 제공합니다.



### HF-Diff: High-Frequency Perceptual Loss and Distribution Matching for One-Step Diffusion-Based Image Super-Resolution (https://arxiv.org/abs/2411.13548)
Comments:
          8 pages

- **What's New**: 최근의 확산 기반 (diffusion-based) 단일 단계 슈퍼 해상도 (super-resolution) 방법들이 SinSR에 비해 더 나은 성능을 보이지만, 계산 복잡성이 있다는 점을 고려하여, 본 연구에서는 고주파 세부 정보( high-frequency detail features)를 보존하면서 성능을 개선하는 방안을 제시합니다. 이를 위해, ImageNet 데이터셋에서 사전 훈련된 가역 신경망 (invertible neural network; INN)을 활용하여 고주파 지각 손실 (perceptual loss)을 도입했습니다. 슈퍼 해상도 이미지와 정답 이미지(ground truth; GT) 간의 고주파 특징을 보존하는 것과 함께 Jenson-Shannon 발산(Jensen-Shannon divergence)을 활용하여 분포를 일치시키는 접근법을 적용했습니다.

- **Technical Details**: 본 연구에서는 단일 단계 확산 기반 슈퍼 해상도 알고리즘에서 고주파 보존 손실을 도입하여 세밀한 세부 정보를 보존하도록 설계되었습니다. INN을 활용하여 ImageNet-1K 데이터셋에서 훈련을 진행하였으며, 이를 통해 생성된 SR 이미지와 해당 GT 이미지 간의 고주파 지각 손실을 계산했습니다. 이 외에도 DINO-v2 임베딩 공간에서 GT 이미지와 SR 이미지의 분포를 최소화하는 Jenson-Shannon 발산을 이용해 분포를 정렬하는 방안을 마련했습니다.

- **Performance Highlights**: 본 슈퍼 해상도 알고리즘은 RealSet65, RealSR, ImageNet, DIV2K-Val, DrealSR 데이터셋에서 최첨단 CLIPIQA 점수를 획득하였으며, DIV2K-Val와 RealSR 데이터셋에서 OSEDiff 알고리즘보다 더 뛰어난 성과를 보였습니다. 본 연구의 방법론은 최근의 다른 SR 접근법들에 비해 질적으로도 개선된 시각적 표현을 제공합니다. 이러한 성과들은 고주파 보존 손실과 분포 정렬이 SR 이미지 품질에 긍정적인 영향을 미친다는 것을 뒷받침합니다.



### Pushing the Limits of Sparsity: A Bag of Tricks for Extreme Pruning (https://arxiv.org/abs/2411.13545)
Comments:
          10 pages, 5 figures, 3 tables

- **What's New**: 이 연구에서는 Extreme Adaptive Sparse Training (EAST)라는 새로운 방법을 제안하여, 99.90%, 99.95%, 및 99.99%의 극단적인 희소성에서도 성능 저하 없이 네트워크를 지속적으로 학습할 수 있도록 돕습니다. 이 방법은 동적 ReLU( Dynamic ReLU) 가동, 가중치 공유, 사이클 희소성(Cyclic Sparsity) 전략을 통합하여 네트워크 성능을 최적화합니다. 이 방법은 기존의 Sparse Training (DST) 프레임워크에 적용하여, 성능 감소를 지연시켜주는 유연한 구조를 제공합니다.

- **Technical Details**: EAST는 세 가지 주요 전략을 결합합니다. 첫 번째는 동적 ReLU(DyReLU)로, 초기에는 보다 풍부한 요청 파라미터 탐색을 허용한 다음 표준 ReLU로 점진적으로 대체됩니다. 두 번째는 가중치 공유로, 잔차(layer) 내에서 파라미터를 재사용하면서도 학습 가능한 파라미터 수를 유지합니다. 세 번째는 사이클 희소성으로, 학습 과정에서 희소성 수준과 패턴이 동적으로 진화하여 파라미터 탐색을 더욱 촉진합니다.

- **Performance Highlights**: EAST는 CIFAR-10, CIFAR-100 및 ImageNet 데이터셋에서 ResNet-34 및 ResNet-50을 사용하여 실험을 수행하여, 기존의 모든 방법보다 높은 성능을 기록했습니다. 또, DyReLU는 초기 설정에서 ReLU의 대체로 사용되었음에도 불구하고 극단적인 희소성에서 높은 성과를 가져오는데 기여했습니다. 이 연구는 희소성과 성능 간의 절충이 불가피하더라도, 극단적 희소성에서도 유의미한 효용을 보여주는 중요한 기여를 합니다.



### DIS-Mine: Instance Segmentation for Disaster-Awareness in Poor-Light Condition in Underground Mines (https://arxiv.org/abs/2411.13544)
- **What's New**: 이 논문에서는 DIS-Mine이라는 새로운 인스턴스 세분화(instance segmentation) 방법을제안합니다. 이 방법은 지하 채굴 작업에서 폭발 및 구조 손상과 같은 재해 영향을 받은 지역을 식별하는 데 초점을 맞출 수 있도록 설계되었습니다. DIS-Mine은 저조도 또는 시야가 좁은 조건에서도 객체를 감지할 수 있어, 구조대원들이 구조 작업을 수행하는 데 도움을 줍니다.

- **Technical Details**: DIS-Mine은 이미지 품질을 향상시키는 컴포넌트(image brightness improvement)와 SAM(Segment Anything Model)과 통합된 인스턴스 세분화 기능 등의 네 가지 핵심 혁신 사항을 포함합니다. Mask R-CNN 기반의 세분화와 특징 매칭을 통한 마스크 정렬(mask alignment with feature matching) 기능도 포함되며, 이를 통해 세분화 결과를 개선합니다. 연구팀은 학습을 위해 저조도 조건에서 수집된 이미지 데이터셋인 ImageMine을 마련했습니다.

- **Performance Highlights**: DIS-Mine은 다양한 데이터셋에서 기존의 최첨단 인스턴스 세분화 방법보다도 뛰어난 성능을 보였으며, F1 score는 86.0%, mIoU는 72.0%에 달합니다. 이는 최소 15배의 개선을 나타내며 객체 감지에서는 최대 80%의 높은 정확도를 자랑합니다. 이러한 결과는 DIS-Mine이 극한의 저조도 환경에서도 효과적으로 작동함을 보여줍니다.



### Identity Preserving 3D Head Stylization with Multiview Score Distillation (https://arxiv.org/abs/2411.13536)
Comments:
this https URL

- **What's New**: 본 논문에서는 PanoHead 모델을 활용하여 360도 시점에서 이미지를 합성하여 3D 헤드 스타일화 (3D head stylization) 문제를 다루고 있습니다. 기존의 3D 스타일화 방법들은 주로 근전방 (near-frontal) 뷰에서 합성되며, 원본 이미지의 고유한 정체성 (identity) 유지에 어려움이 있는 반면, 이번 연구는 이러한 한계를 극복하고자 합니다. 제안된 프레임워크는 negative log-likelihood distillation (LD)을 통해 정체성 보존과 스타일화 품질을 향상시킵니다.

- **Technical Details**: 연구에서 제안하는 방법은 PanoHead의 사전 훈련된 매개 변수를 미세 조정하여 다양한 도메인에서 이미지를 생성합니다. LD 기법을 3D 인식 이미지 생성기 (3D-aware image generators)에 적용하며, SDS와의 차이점을 설명하고, 교차 포즈 의존성 (cross-pose dependencies) 및 그리드 노이즈 제거 (grid denoising)를 추가하여 스타일화 품질을 향상시키는 방법을 제시합니다. 성능을 높이기 위해 점수 텐서에 대한 순위 감소 (rank reduction)를 사용하며, 이는 스타일 기법에도 긍정적인 영향을 미칩니다.

- **Performance Highlights**: 제안한 방법은 관련 있는 헤드 스타일화 방법들에 비해 질적 및 양적 차원이 상당한 개선을 나타냅니다. LD를 사용하여 더욱 선명하고 ID 보존이 우수한 결과를 얻으며, 다각적 그리드 및 미러 점수 기울기를 통합하여 스타일화 품질을 더욱 개선합니다. 이 연구는 3D 헤드 스타일화의 발전뿐 아니라, GAN을 통한 효과적인 증류 (distillation) 과정에 대한 중요한 통찰을 제공합니다.



### Entropy Bootstrapping for Weakly Supervised Nuclei Detection (https://arxiv.org/abs/2411.13528)
Comments:
          Submitted for CVPR 2025

- **What's New**: 본 논문은 Histopathology 영역에서 세포(instance) 분할을 위한 새로운 약한 감독 학습(weakly supervised learning) 접근 방식을 제안합니다. 이 방법은 단일 포인트 레이블(label)만을 사용하여 세포 픽셀의 기본 분포를 추정하며, 이로부터 완전한 세포 마스크를 유추해 Mask-RCNN(Instance Segmentation Model)을 활용하여 결과를 도출합니다. 연구 결과, 95%의 픽셀 레이블 축소에도 불구하고, 제안한 방법이 충분히 좋은 성능을 발휘한다고 보고하였습니다.

- **Technical Details**: 연구에 사용된 데이터는 PanNuke 데이터세트로, 이는 다양한 조직 유형에 대한 암 세포 핵(instance) 분할 및 분류 데이터셋입니다. 논문에서는 Bayesian Segmentation Network를 사용하여 점 레이블 지정보다 더 정확한 세분화(segmentation) 결과를 생성하기 위한 엔트로피 부트스트랩(entropy bootstrap) 단계를 수행합니다. 이 네트워크는 각 픽셀이 특정 클래스에 속할 확률을 추정하며, 제안된 방법은 약한 레이블을 사용하여 세포 검출(task)을 수행하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 접근법은 임상 이미징 시나리오를 더 잘 반영하고, 기존의 세포 분할 데이터셋보다 복잡한 임상 환경을 나타냅니다. 결과적으로, 제안된 방법은 높은 효율성을 통해 최소한의 레이블로도 좋은 성능을 보여줍니다. 특히, 약한 감독 방식으로 얻은 초기 세분화 결과가 Ground Truth와 매우 잘 일치하며, 기존의 고급 방법들과 비교할 때 간단한 구조임에도 불구하고 우수한 검출 성과를 거두었습니다.



### Geometric Algebra Planes: Convex Implicit Neural Volumes (https://arxiv.org/abs/2411.13525)
Comments:
          Code is available at this https URL

- **What's New**: GA-Planes는 볼륨 모델링의 새로운 패러다임을 제시합니다. 이 모델은 컨벡스 최적화(convex optimization)를 통해 훈련할 수 있는 최초의 임플리시트 신경 볼륨 모델입니다. 주요 특징은 각기 다른 해상도의 구성 요소를 포함할 수 있다는 점으로, 많은 기존 모델을 일반화하고 새로운 적용 가능성을 제공합니다. 특히, GA-Planes는 최적화의 안정성을 보장할 수 있는 방법을 제공합니다.

- **Technical Details**: GA-Planes는 기하대수적 기초 요소에서 영감을 받아 설계된 혼합 모델입니다. 이 모델은 선(line), 평면(plane), 볼륨(volume) 특징을 결합하여, 측정값에 적합하도록 하는 목표 함수가 컨벡스일 때 최적화가 가능합니다. GA-Planes는 2D에서 저랭크 및 저해상도 매트릭스 근사와 동등하다는 것을 입증하였으며, 기존의 희소 근사보다 자연 이미지를 더 잘 적합할 수 있음을 보여주었습니다. 또한 3D 작업에서는 데이터 표현성과 최적화 가능성의 경쟁력을 입증합니다.

- **Performance Highlights**: GA-Planes는 3D 복사 필드 재구성, 3D 분할, 비디오 분할과 같은 세 가지 볼륨 적합 작업에서 뛰어난 성능을 보입니다. 이 모델은 메모리 사용량, 표현력, 최적화 가능성 면에서 기존의 방법들과 비교하여 우수한 성능을 보여줍니다. 특히, GA-Planes는 훈련이 보장된 최적화 속도까지 확장할 수 있어 다양한 작업에서 효율적으로 활용될 수 있습니다.



### VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models (https://arxiv.org/abs/2411.13503)
Comments:
          Leaderboard: this https URL Code: this https URL Project page: this https URL extension of arXiv:2311.17982. arXiv admin note: substantial text overlap with arXiv:2311.17982

- **What's New**: VBench라는 비디오 생성 평가 기준이 새롭게 발표되었습니다. 이 시스템은 비디오 생성 품질을 세분화된 계층적 치수로 나누어 평가할 수 있도록 설계되었습니다. 특히, VBench는 인간의 인식과 일치하는 평가 기준을 제공하며, 이를 통해 각 모델의 강점과 약점을 세밀하게 파악할 수 있습니다. 또한, VBench++는 다양한 비디오 생성 모델의 평가를 지원하는 복합적인 프레임워크로 발전했습니다.

- **Technical Details**: VBench는 16가지 비디오 생성 치수를 포함하고 있어, 주체 정체성 일관성, 움직임의 매끄러움, 시간적 깜빡임 및 공간적 관계 등의 다양한 요소를 평가합니다. 비디오 생성 모델의 특성에 맞춘 평가 치수를 통해, 생성된 비디오의 안정성과 조건 일관성을 평가할 수 있도록 개발되었습니다. 이와 함께, 각 평가 치수에 대해 100개의 텍스트 프롬프트를 활용하여 텍스트-비디오(T2V) 생성 모델을 테스트하는 방식을 채택하였습니다.

- **Performance Highlights**: VBench는 다차원적 접근 방식을 통해 비디오 생성 모델의 강점과 약점을 상세하게 분석할 수 있는 통찰을 제공합니다. 평가 방법이 사람의 선호도와 높은 상관관계를 보여주며, 실험과 주석을 통해 인간 인식과 잘 맞도록 조정되었습니다. 또한, VBench는 비디오와 이미지 생성 모델 간의 차이를 조사하여, 다양한 콘텐츠 카테고리에 대한 세분화된 평가를 가능하게 합니다.



### Quantum-Brain: Quantum-Inspired Neural Network Approach to Vision-Brain Understanding (https://arxiv.org/abs/2411.13378)
- **What's New**: 이 논문에서는 뇌 신호와 이미지 자극 간의 연결성을 학습하기 위한 Quantum-Brain 접근법을 제안합니다. 이 방법은 전통적인 딥러닝 방법의 한계를 극복하며, 양자 컴퓨팅의 개념을 도입하여 뇌 신호의 연결성을 효과적으로 처리합니다. 새로운 Quantum-Inspired Voxel-Controlling 모듈을 통해 뇌의 voxel 간의 상호작용을 학습하고, Phase-Shifting 모듈로 신호 값을 보정하여 강건성을 높입니다.

- **Technical Details**: 이 연구의 핵심 기술은 양자 이론을 기반으로 한 네트워크 구조입니다. Quantum-Inspired Voxel-Controlling 모듈은 Hilbert 공간에서 뇌 신호 간의 상관관계를 학습하도록 설계되었습니다. Phase-Shifting 모듈은 fMRI voxel 값을 조정하여 연결성 추출의 정확성을 향상시키며, Measurement-like Projection 모듈은 추출된 정보를 기능 공간으로 변환합니다.

- **Performance Highlights**: 제안한 방법은 Natural Scene Dataset 벤치마크에서 95.1% 및 95.6%의 Top-1 정확도를 달성하여 이미지 및 뇌 검색 작업에서 뛰어난 성능을 나타냅니다. 또한, fMRI-to-image 재구성 작업에서 95.3의 Inception score를 기록하여 기존 방법들과 비교했을 때 월등한 성과를 입증했습니다. 이 연구는 비전-뇌 문제를 해결하기 위한 새로운 패러다임을 제시합니다.



### Learning based Ge'ez character handwritten recognition (https://arxiv.org/abs/2411.13350)
- **What's New**: 이번 연구는 Ge'ez 문자 인식 분야에서 영상 데이터 활용의 부족함을 해결하기 위해 최첨단 Ge'ez 필기체 인식 시스템을 개발했습니다. Convolutional Neural Networks (CNNs)와 Long Short-Term Memory (LSTM) 네트워크를 통합한 이 시스템은 두 단계 인식을 도입하여 뛰어난 성능을 거두었습니다. 실험 결과, 제안한 모델은 Ge'ez 필기 인식에서 여덟 가지의 최첨단 방법과 인간 성과를 초월한 새로운 최고점을 달성했습니다.

- **Technical Details**: 제안된 OCR 모델 아키텍처는 CNN과 LSTM의 조합을 활용하여 Ge'ez 텍스트를 인식합니다. 이 모델은 여러 개의 잔차 블록으로 구성된 컨볼루션 레이어부터 시작하여 특징을 추출하며, 이어서 두 개의 양방향 LSTM 레이어에서 시퀀스 종속성을 캡처합니다. Connectionist Temporal Classification (CTC) 손실을 통해 비가공된 입력 시퀀스에서 학습을 가능하게 하여, 필기체 인식 작업에 있어 더욱 강건한 접근 방식을 제공합니다.

- **Performance Highlights**: 모델은 26.95의 Character Error Rate (CER) 및 26.50의 Normalized Edit Distance (NED)를 기록하며 Ge'ez 광학 문자 인식에서 최첨단 성능을 보였습니다. 이 연구는 Ge'ez 문화유산의 보존과 접근성을 크게 향상시키며, 역사적 문서의 디지털화, 교육 도구 및 문화 보전과 같은 분야에 긍정적인 영향력을 미칠 것입니다.



### WHALES: A Multi-agent Scheduling Dataset for Enhanced Cooperation in Autonomous Driving (https://arxiv.org/abs/2411.13340)
- **What's New**: 이 논문은 Wireless enHanced Autonomous vehicles with Large number of Engaged agentS (WHALES)라는 데이터셋을 소개하고 있습니다. 이 데이터셋은 CARLA 시뮬레이터를 통해 생성되었으며, 평균 8.4개의 에이전트가 포함된 드라이빙 시퀀스를 특징으로 하여 자율주행 데이터셋 중 가장 많은 수의 에이전트와 시점을 제공합니다. WHALES는 에이전트의 행동을 기록하여 다중 작업에서 협력 가능성을 증가시킵니다.

- **Technical Details**: WHALES 데이터셋은 70K RGB 이미지, 17K LiDAR 프레임, 2.01M의 3D 바운딩 박스 주석을 포함하여 다중 에이전트 협업 연구를 지원합니다. 이 데이터셋은 다양한 도로 시나리오를 포함하며, V2V 및 V2I 인식을 지원하는 대규모 스케줄링 데이터셋으로 설계되었습니다. 실험에서는 단독 및 협력 3D 인식, 에이전트 스케줄링 작업을 포함합니다.

- **Performance Highlights**: WHALES 데이터셋은 기존 협력 인식 데이터셋보다 높은 성능 평가를 가능하게 합니다. 평균 8.4개의 시점을 통해 다양한 다중 에이전트 스케줄링 알고리즘의 성능을 평가할 수 있습니다. 데이터를 통해 협력 인식 향상에 기여할 수 있는 새로운 연구를 촉진하는 것이 목표입니다.



### Teaching VLMs to Localize Specific Objects from In-context Examples (https://arxiv.org/abs/2411.13317)
- **What's New**: 본 연구는 Vision-Language Models (VLMs)의 새로운 접근 방식을 소개합니다. 기존의 VLMs가 사람의 인지 능력 중 하나인 맥락 기반 객체 로컬라이징(Localization)을 효과적으로 학습하지 못했음을 지적하며, 이를 해결하기 위한 few-shot personalized localization 작업에 초점을 맞추었습니다. 새로운 데이터 중심(data-centric) 해결책과 정규화 기법을 채택하여 VLMs의 성능 향상을 이루었습니다.

- **Technical Details**: 연구에서는 비디오 객체 추적 데이터셋에서 선별된 데이터를 활용하여 VLMs를 미세 조정하는 방법을 제시합니다. 이를 통해 여러 프레임에서 동일한 객체를 추적하는 방법으로 맥락 인식을 증진하고, 객체 레이블을 pseudo-names로 교체하여 모델이 시각적 맥락에 의존하도록 유도하는 정규화 기법을 도입하였습니다. 이러한 접근 방식은 모델을 더욱 효과적인 few-shot localization으로 이끌어 주며, 일반화 능력도 유지됩니다.

- **Performance Highlights**: 제안된 방법은 personalized localization을 위해 특별히 개발된 새로운 평가 벤치마크에서 모델의 성능을 크게 향상시킵니다. VLMs가 단순히 제공된 few-shot 예시를 복사하는 경향을 줄이며, 객체 인식 맥락에서 보다 구조적인 응답(output)을 생성할 수 있도록 합니다. 전체적으로 본 연구는 VLMs의 context-driven vision-language 응용 프로그램의 미래 연구 방향을 설정하는 중요한 기초가 됩니다.



### A Resource Efficient Fusion Network for Object Detection in Bird's-Eye View using Camera and Raw Radar Data (https://arxiv.org/abs/2411.13311)
Comments:
          IEEE Intelligent Transportation Systems Conference (ITSC) 2024

- **What's New**: 이 연구는 자동차 자율주행 시스템에서 카메라와 레이더 데이터를 융합하여 객체 인식을 향상시키는 새로운 아키텍처를 제안합니다. 기존의 레이더 신호 처리를 피하고 원시 Range-Doppler (RD) 스펙트럼을 직접 활용합니다. 특히, 카메라 이미지를 Bird's-Eye View (BEV) 극좌표 도메인으로 변환하고, 카메라 인코더-디코더 아키텍처를 통해 필수 특징을 추출합니다.

- **Technical Details**: 제안된 방법은 카메라 데이터와 레이더 데이터를 각각 독립적으로 처리하여 최적의 융합을 추구합니다. 카메라 이미지는 RA와 유사한 표현으로 변환되며, RD 스펙트럼으로부터 복원된 Range-Azimuth (RA) 특징과 융합됩니다. 이 방법은 다양한 감지 작업에 대한 정밀한 정보 표현을 목표로 하며, 데이터 처리를 위한 새로운 이미지 처리 파이프라인을 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 융합 전략이 기존 방법들과 비교하여 정확도에서 경쟁력을 가지며, 계산 복잡성 측면에서 우수한 성능을 보임을 확인했습니다. RADIal 데이터셋을 활용하여 평가한 결과, 리소스 소모를 최소화하면서도 뛰어난 인식 성능을 달성했습니다.



### Can Reasons Help Improve Pedestrian Intent Estimation? A Cross-Modal Approach (https://arxiv.org/abs/2411.13302)
- **What's New**: 이번 연구에서는 보행자의 의도를 예측하는 데 있어 직관적인 이유(Why)를 탐색하는 새로운 접근 방식을 소개합니다. 특히, 우리는 보행자의 의도를 이해하기 위해 의도의 "이유"를 예측하는 것이 매우 유용하다는 것을 보여줍니다. 이를 위한 새로운 다중 라벨 텍스트 설명 데이터를 포함하는 PIE++ 데이터셋을 제안하고, 이 데이터셋은 보행자 의도에 대한 다각적인 정보를 제공합니다.

- **Technical Details**: 본 연구에서는 의도 예측과 그 이유를 동시에 추론하기 위해 MINDREAD라는 새로운 다중 작업 학습 프레임워크를 제안합니다. 이 프레임워크는 시각적(visual) 및 언어적(textual) 모듈을 활용하는 Cross-modal representation learning을 적용하여 보행자 의도를 예측하는 데 도움을 줍니다. 추가적으로, 텍스트 설명의 의미적 상관 관계를 포착하는 새로운 모듈을 도입하여, 이를 시공간(spatio-temporal) 특성과 결합하여 더 나은 의도 예측을 달성합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, MINDREAD를 사용한 PIE++ 데이터셋에서 의도 예측의 정확도는 5.6%, F1-score는 7% 향상되었습니다. 또한, JAAD 데이터셋에서도 4.4%의 정확도 향상을 달성했습니다. 이러한 결과는 우리의 방법론의 효과성을 입증하며, 보행자 의도 예측의 새로운 가능성을 제시합니다.



### DATAP-SfM: Dynamic-Aware Tracking Any Point for Robust Structure from Motion in the Wild (https://arxiv.org/abs/2411.13291)
- **What's New**: 이 논문에서는 동적 객체가 포함된 비디오에서 부드러운 카메라 궤적을 추정하고 밀집 포인트 클라우드를 생성하기 위한 새로운 파이프라인인 Dynamic-Aware Tracking Any Point (DATAP) 방법을 제안합니다. 기존의 방법들은 인접한 프레임 간의 optical flow를 기반으로 카메라의 위치를 추정하는 과정에서 누적 오류가 발생하는 문제를 안고 있었습니다. DATAP은 비디오 시퀀스 전반에 걸쳐 일관된 깊이 정보와 포인트 추적을 활용하여 이러한 문제를 해결하고, 모든 카메라 포즈를 동시에 추정할 수 있게 합니다.

- **Technical Details**: DATAP은 슬라이딩 윈도우 방식으로 작동하는 트랜스포머 네트워크입니다. 이 모델은 각 포인트의 추적과 가시성을 추정하고 동적 모션 레이블을 예측하기 위해 다층 지각층(multi-layer perceptual layer)을 통합합니다. 슬라이딩 윈도우 내에서는 각 질의 포인트의 경로를 초기화하고, 이전 윈도우의 예측을 바탕으로 반복적으로 이 경로를 정제하는 방식으로 동작합니다. 또한, 비디오 깊이 정보를 이용하여 2D 포인트의 모션 세분화를 개선하여 스케일 모호성을 해결합니다.

- **Performance Highlights**: DATAP를 통합한 결과, 동적 장면에서 평균적인 카메라 로컬라이제이션 정확도를 획기적으로 향상시켰습니다. Sintel 및 TUM RGBD 동적 데이터셋에서 실시한 실험 결과, 제안한 방법이 기존의 최첨단(SOTA) 방법들을 능가하는 성능을 보였음을 확인했습니다. 또한, 실제 환경에서의 캐주얼 비디오에 대해 실험한 결과, 복잡한 도전적인 시나리오에서도 강력한 로컬라이제이션 성능을 입증하였습니다.



### Unbiased Scene Graph Generation by Type-Aware Message Passing on Heterogeneous and Dual Graphs (https://arxiv.org/abs/2411.13287)
- **What's New**: 이번 논문에서는 TA-HDG를 통해 비편향적 장면 그래프 생성(Unbiased Scene Graph Generation)의 성능을 향상시킵니다. 이 모델은 객체 간 상호작용과 관계 간 상호작용을 동시에 모델링하여, 긴 꼬리 문제를 해결합니다. 이를 통해 기존의 SGG 기법에서 발생하는 예측 성능의 편향성을 줄입니다.

- **Technical Details**: TA-HDG는 Heterogeneous and Dual Graph Construction(HDGC)와 Type-Aware Message Passing(TAMP)를 사용합니다. HDGC는 객체와 관계 간의 상호작용을 모델링하며, TAMP는 intra-type(내부 타입) 및 inter-type(외부 타입) 컨텍스트를 캡처하여 복잡한 상호작용을 이해하도록 돕습니다. 이 두 가지 방법을 통해 관계의 정확한 예측 가능성을 크게 향상시킵니다.

- **Performance Highlights**: TA-HDG는 Visual Genome 및 Open Images 데이터세트에서 경쟁력 있는 성능을 보여줍니다. 실험 결과, R@K 및 mR@K의 지표 모두 향상되었으며, 이는 TA-HDG가 긴 꼬리 클래스를 정확하게 예측하고 머리 클래스의 경쟁력 있는 성능을 유지할 수 있음을 증명합니다.



### DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition (https://arxiv.org/abs/2411.13284)
- **What's New**: 새로운 연구에서는 WiFi 기반 인식의 도메인 간 일반화를 위한 Domain-Adversarial Test-Time Adaptation (DATTA)라는 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 도메인-적대적 학습(DAT), 테스트 시간 적응(TTA) 및 가중치 리셋을 결합하여 이전에 본 적 없는 타겟 도메인에 적응하고 파국적 망각(catastrophic forgetting)을 방지합니다. 또한, DATTA는 속도 최적화를 위해 경량화되고 유연한 아키텍처에 통합됩니다.

- **Technical Details**: DATTA 프레임워크는 WiFi 기반의 인식 시스템에서 도메인 불변 특성을 학습하며, 수신 데이터의 도메인 전이에 적응하는 것을 목표로 합니다. 이 시스템은 WiFlexFormer 기반의 중앙 특징 추출기를 통해 실시간으로 인식할 수 있도록 설계되었습니다. 또한, 실제 측정한 WiFi 채널 상태 정보(CSI)의 특성을 이용한 특수한 보강 모듈이 통합되어 교차 도메인 일반화를 강화합니다.

- **Performance Highlights**: DATTA 방법론은 공개된 데이터셋을 사용한 포괄적인 평가에서 검증되었으며, 인간 활동 인식을 위한 실시간 애플리케이션에 적합함을 입증했습니다. 비디오 기반의 TTA 변형과 비교하여, DATTA는 8.1% 향상된 F1-스코어를 달성했습니다. 이 연구의 PyTorch 구현은 공개되어 있어, 후속 연구와 재현 가능성을 용이하게 합니다.



### VideoAutoArena: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation (https://arxiv.org/abs/2411.13281)
Comments:
          Project Page: this https URL

- **What's New**: 최신 연구에서는 비디오 분석 능력을 갖춘 대형 다중 모달 모델(LMMs)에 대한 관심이 증가하고 있습니다. 기존 평가 방법들은 사용자 요구를 충분히 반영하지 못하고, 비용이 많이 드는 인적 주석이 필요했습니다. 이를 해결하기 위해 제안된 VideoAutoArena는 사용자 시뮬레이션을 기반으로 한 자동 평가 시스템으로, 질문 생성 방식에서 혁신적인 접근을 제공합니다. 또한, VideoAutoBench는 신속하고 효율적인 평가를 지원하여 실제 사용자 질문 스타일에 더 가깝게 평가할 수 있습니다.

- **Technical Details**: VideoAutoArena는 사용자에 의해 생성된 열린 질문을 통해 모델의 비디오 이해 능력을 엄격하게 평가하도록 설계되었습니다. 이 평가 모델은 ELO 평가 시스템을 통해 다양한 LMM 간의 공정하고 지속적인 비교를 가능하게 합니다. 또한, 점진적으로 질문의 복잡성을 증가시키는 결함 기반 진화 전략을 도입하여 모델이 더 도전적인 비디오 분석 시나리오를 처리할 수 있도록 합니다. 이를 통해 기존의 LMM 평가 방식에서 발생했던 여러 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, VideoAutoArena는 최신 LMM들 간의 비교를 효과적으로 수행하며, 모델의 강점과 개선이 필요한 영역에 대한 통찰력을 제공합니다. 비공식 모델들이 SOTA(SOTA: State Of The Art) 모델인 GPT-4o에 비해 여전히 성능 차이를 보이며, 이 차이는 비디오 길이가 늘어날수록 더욱 두드러집니다. VideoAutoBench의 결과는 VideoAutoArena의 평가 결과와 밀접하게 일치하는 것으로 나타나, 두 가지 벤치마크가 서로 보완적인 역할을 하고 있음을 보여줍니다.



### Paying more attention to local contrast: improving infrared small target detection performance via prior knowledg (https://arxiv.org/abs/2411.13260)
Comments:
          16 pages, 8 figures

- **What's New**: 본 논문에서는 Local Contrast Attention Enhanced infrared small target detection Network (LCAE-Net)을 제안하여 이전의 데이터 기반 방법과 전문가 지식을 결합했습니다. LCAE-Net은 U자형(neural network model) 모델로, Local Contrast Enhancement (LCE) 모듈과 Channel Attention Enhancement (CAE) 모듈을 포함하고 있습니다. 이 접근법은 주어진 작은 데이터 세트에서도 효율적으로 학습할 수 있도록 설계되었습니다.

- **Technical Details**: LCAE-Net의 LCE 모듈은 사전 지식을 활용하여 Local Contrast Attention (LCA)을 생성함으로써 배경을 억제하고 잠재적인 타겟의 위치 정보를 강조합니다. CAE 모듈은 다채널(feature map) 정보의 융합을 수행하여 다운샘플링 진행 중 정보를 효과적으로 활용하도록 돕습니다. 이 모델은 파라미터 수와 Floating-Point Operations (FLOPs) 수에서 기존 방법들보다 더 적은 양을 요구하여 엣지 디바이스에서의 배포가 용이합니다.

- **Performance Highlights**: LCAE-Net은 NUDT-SIRST, NUAA-SIRST, IRSTD-1K의 세 개의 공공 데이터 세트에서 최첨단 방법들보다 우수한 성능을 나타냈습니다. 특히, 이 모델은 최대 70 fps의 감지 속도를 기록하여 실시간 응용에서도 유용성(utility)을 보입니다. 또한, LCAE-Net은 성능과 계산 효율 간의 균형을 잘 맞추었다는 점에서 엣지 디바이스에서의 효과적인 활용 가능성을 시사합니다.



### BelHouse3D: A Benchmark Dataset for Assessing Occlusion Robustness in 3D Point Cloud Semantic Segmentation (https://arxiv.org/abs/2411.13251)
Comments:
          20 pages, 6 figures, 3 tables, accepted at ECCV 2024 Workshops

- **What's New**: BelHouse3D 데이터셋은 실세계 조건에 밀접하게 일치한 합성 포인트 클라우드 데이터셋으로, 실내 장면의 의미적 세분화를 위해 설계되었습니다. 벨기에의 32개 주택에서 얻은 실제 참고 자료를 바탕으로 구축되어 더 높은 공정성(fairness)과 신뢰성을 제공합니다. 이전의 데이터셋들과는 달리, 본 데이터셋은 occlusion(가림 현상)을 시뮬레이션한 테스트 세트를 포함하여 out-of-distribution (OOD) 상황을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 기존의 3D 벤치마킹 데이터셋은 종종 훈련과 테스트 데이터가 동일하게 분포(IID)된 것이라는 암묵적인 가정을 가지고 평가되었습니다. 하지만 실제 포인트 클라우드 데이터에서는 occlusion이 불가피하게 발생하여 장면의 형태와 구조가 변화하므로, OOD 샘플링에 따른 성능 저하가 우려됩니다. BelHouse3D는 이러한 문제를 해결하기 위해 수작업 3D 모델링과 샘플링 기법을 사용하여 데이터셋을 생성하며, 이는 라벨링 정확성을 향상시키기 위함입니다.

- **Performance Highlights**: BelHouse3D와 OOD 환경에서의 평가를 통해 인기 있는 포인트 기반 의미적 세분화 기법들의 성능을 검증하였습니다. 이러한 OOD 설정은 모델들이 실제 환경에서 어떻게 일반화되는지를 이해하는 데 도움을 주며, 더 견고하고 신뢰할 수 있는 모델 설계를 용이하게 합니다. 연구자들은 이 데이터셋과 OOD 환경을 통해 실내 장면에 대한 3D 포인트 클라우드 의미적 세분화 연구가 발전하길 기대하고 있습니다.



### XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation (https://arxiv.org/abs/2411.13243)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문에서는 XMask3D라는 새로운 방법론을 제안하고 있습니다. XMask3D는 3D 특성과 2D-텍스트 임베딩 공간 간에 보다 정밀한 mask-level 정렬을 실현하는 cross-modal mask reasoning 프레임워크입니다. 이를 통해 기존 방법들보다 정교한 세분화 경계를 다루게 되어, 개방된 어휘(open vocabulary)의 3D 의미론적 분할 문제를 해결하는 데 중요한 진전을 이룰 수 있습니다.

- **Technical Details**: XMask3D는 3D 지점 클라우드(segmentation model)와 텍스트 기반의 mask generator로 구성되어 있습니다. 3D 브랜치는 기하학적 특성을 추출하는 데 탁월하지만, 새로운 카테고리 외삽에는 한계가 있습니다. 반면, 2D 브랜치는 vision-language feature space와 정렬된 mask를 생성하는 역할을 하며, 세 가지 주요 기법인 3D-to-2D Mask Generation, 2D-to-3D Mask Regularization, 3D-2D Mask Feature Fusion을 포함하여 두 가지 브랜치 간의 상호작용을 강화하는 방식으로 설계되었습니다.

- **Performance Highlights**: XMask3D는 다양한 벤치마크 데이터 세트인 ScanNet20, ScanNet200 및 S3DIS에서 경쟁적인 성능을 보여주었습니다. 이 방법은 다양한 자료에 대한 ablation studies와 직관적인 시각화를 통해 제안된 각 mask-level 기법의 기여도를 명확히 드러내고 있습니다. 전체적으로 XMask3D는 3D 개방어휘 의미론적 분할에서 뛰어난 성능을 입증하며, 각 기법이 함께 시너지를 이루는 방식으로 혁신을 나타냅니다.



### ViSTa Dataset: Do vision-language models understand sequential tasks? (https://arxiv.org/abs/2411.13211)
- **What's New**: 이 연구에서는 시각-언어 모델(Vision-Language Models, VLMs)을 강화 학습에서 보상 모델로 활용하는 가능성을 탐구합니다. 기존에는 특정 최종 결과를 중심으로 한 목표 지향 작업에서만 사용되었지만, 본 연구에서는 VLMs가 최종 상태에 의해 평가할 수 없는 연속 작업을 감독하는 데도 활용될 수 있음을 주장합니다. 이를 위해 'ViSTa'라는 새로운 데이터셋을 도입하여 다양한 복잡도의 작업을 평가합니다. ViSTa는 가상 환경, 마인크래프트, 실제 환경에서 수행된 4,000개 이상의 비디오와 단계별 설명으로 구성되어 있습니다.

- **Technical Details**: ViSTa는 기본 단일 액션 작업을 더 복잡한 순차 작업으로 조합하는 새로운 계층 구조를 가지고 있습니다. 이 계층적 구조를 통해 VLMs가 다양한 복잡도의 작업을 어떻게 이해하고 판단할 수 있는지를 세밀하게 평가할 수 있습니다. 연구에서는 최신 VLM인 CLIP, ViCLIP, GPT-4o를 사용하여 평가를 진행했고, 제안된 데이터셋을 통해 이러한 모델들이 순차 작업을 이해하는 데 한계가 있음을 발견했습니다. 특히, GPT-4o는 비록 우수한 객체 인식 능력을 보였지만, 복잡한 연속 작업에서는 비트 이상의 성능을 보이지 못했습니다.

- **Performance Highlights**: 비교한 모델들 모두 기본적인 객체 인식 작업에서는 우수한 성능을 보였지만, 순차 작업에 대한 이해에서는 실패를 보였습니다. 이 연구는 모델이 특정 복잡도 이상의 작업을 감독할 준비가 되지 않았음을 강조합니다. 이러한 결과는 VLMs의 적용 가능성을 재조명하며, 저자들은 ViSTa를 통해 이러한 모델들이 순차 작업에서 겪는 문제를 체계적으로 분석할 수 있도록 하는 것을 목표로 삼고 있습니다. VLM의 한계를 극복하고, 보다 효과적인 감독 방식의 개발이 필요함을 시사합니다.



### VADet: Multi-frame LiDAR 3D Object Detection using Variable Aggregation (https://arxiv.org/abs/2411.13186)
Comments:
          Accepted by WACV 2025

- **What's New**: 이번 연구에서는 Variable Aggregation Detection (VADet)이라는 새로운 방법을 제안합니다. 기존의 고정 집계(fixed aggregation) 기법은 여러 프레임을 결합하는 동안 성능이 저하되는 경향이 있었으나, VADet은 물체의 관측 속성에 따라 동적으로 프레임 수를 조절하여 문제를 해결합니다. 이 방식은 아키텍처에 독립적이며, 다양한 속도의 물체에 대해 최적의 집계 성능을 발휘합니다.

- **Technical Details**: VADet의 핵심은 η(eta) 함수로, 각 물체를 집계할 최적의 프레임 수와 매핑합니다. 이 함수는 무작위 집계 훈련(random aggregation training, RAT)을 통해 효율적으로 생성되며, 다양한 프레임 수 조합의 영향을 연구합니다. RAT를 사용하면 오브젝트에 따라 효과적인 집계 프레임 수를 조정할 수 있어, 속도와 점 밀도(point density)의 요소를 모두 고려할 수 있습니다.

- **Performance Highlights**: VADet은 세 가지 인기 있는 단일 단계 탐지기(single-stage detectors)에 적용되어 Waymo 데이터셋에서 최첨단 성능을 달성했습니다. VADet은 고정 집계 방식보다 우수한 성능을 지속적으로 발휘하며, 복잡한 기법에 비해 안정적인 성능 향상을 보여주었습니다. 이러한 결과들은 VADet이 기존 모델의 성능을 보완하고, 실시간 애플리케이션에서의 사용 가능성을 높이는데 기여할 수 있음을 시사합니다.



### Click; Single Object Tracking; Video Object Segmentation; Real-time Interaction (https://arxiv.org/abs/2411.13183)
- **What's New**: 이번 논문에서는 전통적인 단일 객체 추적(SOT)의 초기화 방법에 대한 재검토를 통해 실시간 상호작용 시나리오에 적합한 ClickTrack이라는 새로운 패러다임을 제안합니다. ClickTrack은 클릭 상호작용을 사용해 객체의 경계 상자를 초기화하며, 이를 통해 효율적이고 직관적인 추적 효율성을 목표로 합니다. 이를 보완하기 위해 Guided Click Refiner(GCR)라는 모델을 개발하여 클릭과 선택적 텍스트 정보를 결합해 경계 상자의 불확실성을 줄입니다.

- **Technical Details**: GCR은 클릭 상호작용을 통해 객체의 정확한 위치 정보를 제공하며, 항목의 카테고리 정보를 추가하여 추적할 영역을 명확히 할 수 있도록 설계되었습니다. 이 구조는 시각적 특성과 안내적 특성이 결합되어 경계 상자를 생성하는 방식으로 작동합니다. ClickTrack 방법은 단순히 클릭을 사용하여 초기화를 수행하므로, 복잡한 자연어 설명 없이도 상황을 쉽게 전달할 수 있습니다.

- **Performance Highlights**: LaSOT와 GOT-10k 벤치마크에서 GCR이 결합된 추적기의 성능이 입증되었으며, 이는 실시간 상호작용 시나리오에서도 안정적인 성능을 기록했습니다. GCR과 SAM을 통합한 실험 결과, GCR 구조가 단일 점 입력으로 인한 모호성 문제를 완화하는 데 도움을 주어 실시간 상호작용 시나리오에서의 SAM 성능을 향상시키는 것으로 나타났습니다. GCR은 단일 포인트 위치 입력에 대해서도 강력한 견고성을 보이며, 실시간 상호작용 요구 사항을 충족하는 처리 속도를 자랑합니다.



### Cross-Camera Distracted Driver Classification through Feature Disentanglement and Contrastive Learning (https://arxiv.org/abs/2411.13181)
- **What's New**: 이 논문에서는 운전자의 주의 분산을 효과적으로 분류하는 새로운 모델인 Driver Behavior Monitoring Network (DBMNet)을 소개합니다. 이 모델은 카메라 위치 변화에 강한 내성을 가지며, 경량화된 CNN 아키텍처를 사용해 상용 장치에 배치 가능하도록 설계되었습니다. 특히, 특징 분리(disentanglement) 모듈을 도입하여 운전 행동의 중요한 특징을 강화하고, 다양한 카메라 뷰에 대해 일관된 성능을 보장합니다.

- **Technical Details**: DBMNet은 RGB 이미지를 인코딩하는 경량화된 CNN을 기반으로 하며, 행동 관련 특징과 뷰 관련 특징을 분리하는 기능을 갖춘 모듈을 채택합니다. 또한, triplet loss를 사용하여 서로 다른 카메라의 같은 행동을 더 가까이 묶고, 다른 행동은 separación하여 운전 행동 분류의 정확성을 높입니다. 이 구조는 학습 과정에서 모델이 다양한 카메라 뷰에 대해 강건한 특징 표현을 학습하도록 유도합니다.

- **Performance Highlights**: DBMNet은 100-Driver 데이터셋의 주간 및 야간 부분에서 기존 기술 대비 Top-1 정확성이 평균 9% 향상됨을 입증하였습니다. 또한, AUCDD-V1, EZZ2021 및 SFD와 같은 세 가지 벤치마크 데이터셋에서 크로스 데이터셋 및 크로스 카메라 실험을 통해 뛰어난 일반화 능력을 보여주었습니다. 이러한 성과는 다양한 운전 상황에서 분산 운전자의 행동을 효과적으로 분류할 수 있는 가능성을 열어줍니다.



### AGLP: A Graph Learning Perspective for Semi-supervised Domain Adaptation (https://arxiv.org/abs/2411.13152)
Comments:
          8page

- **What's New**: 이 논문은 반지도 방식 도메인 적응(Semi-Supervised Domain Adaptation, SSDA)에 대한 새로운 접근 방식을 제안하며, 데이터 구조와 도메인 레이블을 모형화하는 그래프 학습 관점을 도입합니다. 기존의 SSDA 방법들은 도메인 레이블과 클래스 레이블의 정보는 사용하였으나, 데이터의 구조적 정보를 간과했습니다. 이 연구는 이러한 문제를 해결하기 위해 그래프 컨볼루션 네트워크(Graph Convolutional Network, GCN)를 인스턴스 그래프에 적용하여 구조적 정보를 전파하도록 설계되었습니다.

- **Technical Details**: 제안된 모델(AGLP: Adversarial Graph Learning Perspective)은 클래스 중심 정렬(class centroid alignment) 과정 중 서로 다른 클래스의 중심을 이동시키는 방법을 사용하여 도메인 불변적이고 의미론적 표현을 효과적으로 학습할 수 있도록 합니다. 이 네트워크는 표준 CNN으로 추출된 샘플의 CNN 피쳐를 기반으로 밀접하게 연결된 인스턴스 그래프를 구성하고, 이 그래프를 통해 데이터 구조와 도메인 레이블을 동시에 모델링합니다. 네트워크의 설계된 가중치를 통해 구조적 정보를 학습하여 도메인간 간극을 줄이는 효율성을 높입니다.

- **Performance Highlights**: 여러 표준 벤치마크에서 수행된 실험 결과, 제안된 AGLP 알고리즘이 기존의 최첨단 SSDA 방법과 비교하여 성능이 우수함을 보여주었습니다. 특히, 이 모델은 도메인 간 불일치를 줄이며, 더 나은 일반화 능력을 제공하여 모델의 적응력을 향상시킵니다. 이를 통해 SSDA 문제를 해결하는 데 중요한 기여를 할 수 있음을 입증하였습니다.



### RAW-Diffusion: RGB-Guided Diffusion Models for High-Fidelity RAW Image Generation (https://arxiv.org/abs/2411.13150)
Comments:
          Accepted at WACV 2025

- **What's New**: 이 논문에서는 새로운 Diffusion 기반 방법을 통해 RAW 이미지 생성을 RGB 이미지 가이드를 통해 수행하는 혁신적인 접근법을 제안합니다. 기존의 RGB 이미지는 정보 손실이 많고 처리 비용이 높지만, RAW 이미지는 풍부한 데이터를 제공하여 인식 성능을 높입니다. 이 방식은 고해상도 RAW 이미지를 생성하며, 카메라 특정 RAW 데이터셋을 만들어 내는 데 필요한 시간을 대폭 줄여줍니다.

- **Technical Details**: 제안된 방법은 다양한 해상도에서 RGB 이미지에서 추출한 특징을 활용하여 RAW 이미지를 고충실도로 생성하는 과정에서 RGB 가이드를 포함합니다. RGB 가이드 모듈을 사용하여 핵심 특징을 추출하고, 이를 역 확산 과정에 통합함으로써 높은 품질의 RAW 이미지를 생성합니다. 이 과정은 특히 낮은 조명 조건에서도 가장 효과적인 이미지 인식을 가능하게 하며, 한정된 수의 훈련 샘플로도 우수한 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, RAW-Diffusion 모델은 네 개의 DSLR 데이터셋에서 최신 기술을 초월하는 성능을 입증했습니다. 데이터 효율성 측면에서도 25개 샘플만으로도 놀라운 성과를 낼 수 있었으며, BDD100K-RAW 및 Cityscapes-RAW 데이터셋을 생성하여 RAW 이미지에서의 객체 감지 성능도 향상시켰습니다. 이러한 접근법은 향후 다양한 컴퓨터 비전 작업에서 RAW 이미지의 활용성을 높이는데 기여할 것으로 기대됩니다.



### YCB-LUMA: YCB Object Dataset with Luminance Keying for Object Localization (https://arxiv.org/abs/2411.13149)
- **What's New**: 이 논문은 고품질의 훈련 데이터를 생성하는 데 도움이 되는 새로운 방법인 luminance keying을 제안합니다. YCB-V 세트의 나머지 객체들을 추가로 기록함으로써, 이 접근법의 유용성을 더욱 입증하고 있습니다. 이로 인해 여러 가지 투명 물체, 다양한 색상의 물체 및 변형 가능한 물체가 포함되었으며, 이는 새로운 2D 객체 탐지 및 분할 알고리즘 테스트에도 활용될 수 있습니다.

- **Technical Details**: Deep Neural Networks (DNNs)는 최상의 성능을 위해 많은 양의 주석이 달린 데이터 세트를 요구합니다. 기존의 데이터 수집 방법인 수동 주석 및 렌더링은 시간 소모가 크고 오류가 발생하기 쉬운 반면, luminance keying 방법은 자동 객체 마스킹과 밝기 임계값을 활용하여 노동 집약적인 수동 주석의 필요를 제거합니다. 이 연구에서는 추가 객체들을 기록하고 데이터를 자동으로 처리할 수 있는 코드도 제공하여 데이터의 다양성을 증가시켰습니다.

- **Performance Highlights**: 우리는 YCB-V 객체 데이터 세트를 luminance keying 방식으로 확장하여 2D 객체 탐지기와 분할 알고리즘의 성능 평가에 기여하고자 합니다. 제공된 데이터는 https://huggingface.co/datasets/tpoellabauer/YCB-LUMA와 https://github.com/tpoellabauer/ycb-luma에서 확인할 수 있으며, YCB-V 데이터 세트도 사용할 수 있습니다. 이러한 향상된 데이터 세트는 다양한 객체들로 구성되어 있으며, 알고리즘 성능 검증의 의미를 더욱 높이고 있습니다.



### GraphCL: Graph-based Clustering for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2411.13147)
Comments:
          9page

- **What's New**: 이 논문에서는 반지도학습(semi-supervised learning, SSL)을 기반으로 한 새로운 손상된 의학 이미징 분할 기술인 GraphCL(그래프 기반 군집화 클러스터링)을 제안합니다. 기존의 방법들과 달리, 데이터 구조 정보를 그래프 모델로 통합하여 반지도 의학 이미지 분할에서의 성능을 개선했습니다. 구체적으로, 이 연구는 반지도 의학 이미지 분할(SSMIS)에 대한 데이터 구조 정보 모델링의 첫 번째 사례로 간주됩니다.

- **Technical Details**: GraphCL의 핵심 개념은 CNN(Convolutional Neural Network)에서 추출된 기능을 기반으로 샘플의 구조적 유사성을 반영한 밀집 인스턴스 그래프(dense instance graph)를 구축하는 것입니다. 각 노드는 CNN의 피쳐와 연결되어 있으며, 이 인스턴스 그래프에서 그래프 신경망(GCN, Graph Convolutional Network)을 적용해 구조 정보를 전달합니다. k-클러스터(k-clustering)의 수를 지정할 필요가 없는 클러스터링 전략을 도입하여 더 높은 세분화 정확성을 얻습니다.

- **Performance Highlights**: 세 가지 표준 벤치마크에서의 광범위한 실험 결과는 GraphCL 알고리즘이 최신의 반지도 의학 이미지 분할 기법들보다 우수한 성능을 보임을 보여줍니다. 이전의 방법들에서는 반지도 학습에서 그래프 정보의 중요성이 간과되었으나, GraphCL은 이 문제를 해결하여 클러스터 특성을 효과적으로 통합합니다. 이 알고리즘은 의학적 데이터 세트의 레이블이 부실한 상황에서도 뛰어난 결과를 입증했습니다.



### Globally Correlation-Aware Hard Negative Generation (https://arxiv.org/abs/2411.13145)
Comments:
          Accepted by IJCV'24

- **What's New**: 본 논문은 전 세계 샘플 관계를 고려하여 하드 네거티브 생성(hard negative generation, HNG)의 효율을 향상시키고자 하는 GCA-HNG(Globally Correlation-Aware Hard Negative Generation) 프레임워크를 제안합니다. 기존 HNG 기법이 로컬(local) 샘플들의 상관관계에만 집중한 것과 달리, GCA-HNG는 전 세계적으로 샘플 간 상관관계를 학습하여 더 정보가 풍부한 네거티브를 생성할 수 있게 합니다. 이 프레임워크는 구조화된 그래프를 사용하여 샘플 간의 관계를 모델링하며, 이를 통해 전역적인 상관관계를 학습합니다.

- **Technical Details**: GCA-HNG 프레임워크는 두 가지 주요 모듈, 즉, GCL(Globally Correlation Learning) 모듈과 CACAI(Correlation-Aware Channel-Adaptive Interpolation) 모듈로 구성됩니다. GCL 모듈은 샘플 간 코릴레이션을 효과적으로 추출하기 위해 반복적인 그래프 메시지 전파(消息传播) 메커니즘을 도입하며, 각 노드는 특정 샘플을, 각 엣지는 상관관계를 나타냅니다. CACAI 모듈은 학습된 전역 상관관계를 사용하여 HNG를 위한 채널 적응형(interpolation) 벡터를 생성하며, 각 채널에 대해 개별적인 계수를 결정하여 다양한 네거티브 샘플을 생성합니다.

- **Performance Highlights**: 다양한 백본 네트워크(backbone networks)와 메트릭 손실(metric losses)의 조합을 사용하여 네 가지 이미지 검색 벤치마크 데이터세트에서 GCA-HNG의 성능을 광범위하게 실험했습니다. 그 결과, GCA-HNG는 기존의 고급 메트릭 학습 방법보다 우수한 성능을 보여주었습니다. 논문에서 제안한 기법은 정보가 풍부한 네거티브 표현을 생성하여 메트릭 모델 최적화에 직접 적용할 수 있으며, 이는 모델 적응력(feature adaptability)을 크게 향상시킵니다.



### TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models (https://arxiv.org/abs/2411.13136)
- **What's New**: 이 논문에서는 CLIP의 시각적 적대 공격에 대한 추론 강건성을 향상시키기 위해 Test-Time Adversarial Prompt Tuning(TAPT)이라는 새로운 방어 방법을 제안합니다. TAPT는 테스트 샘플마다 방어적인 bimodal(텍스트 및 시각) 프롬프트를 최적화하여 추론 과정을 개선합니다. 이 방법은 비지도 방식으로 다중 시각적 엔트로피를 최소화하고 적대적-클린 분포를 정렬하여 작동합니다.

- **Technical Details**: TAPT는 테스트 단계에서 작동하며, 특정 작업에 대한 훈련 세트나 주석이 필요하지 않습니다. 시스템은 두 개의 비지도 손실, 즉 다중 뷰 엔트로피와 적대적-클린 임베딩 정렬을 최소화하여 방어 프롬프트를 학습합니다. 이를 통해 TAPT는 테스트 샘플이 제공될 때마다 적대적인 이미지 임베딩에 맞춰 텍스트 임베딩을 정렬하는 강력한 프롬프트를 동적으로 식별합니다.

- **Performance Highlights**: TAPT는 11개의 벤치마크 데이터 세트에서의 실험을 통해 기존 APT 방법들을 능가하여, AutoAttack에 대한 제로샷 적대적 강건성을 최소 48.9% 향상시킴을 보였습니다. 이 방법은 CLIP의 성능을 유지하면서도 청정 샘플에 대한 성능저하를 최소화하였습니다. 또한 TAPT는 여러 기존 방법과 비교했을 때 평균 36.6%의 강건성 향상을 기록했습니다.



### Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images (https://arxiv.org/abs/2411.13127)
Comments:
          13 pages, 9 figures

- **What's New**: 본 논문에서는 Cloud-Adapter라는 새로운 구름 세분화(cloud segmentation) 방법을 제안합니다. 이 방법은 미리 훈련된 VFM(vision foundation model)을 동결(frozen)한 상태에서, 추가 훈련 없이 효율적으로 구름 세분화의 정확도와 견고성을 향상시킵니다. 또, 경량의 공간 인식 모듈(spatial perception module)을 사용하여 고밀도의 공간 표현을 추출하고, 이를 기반으로 도메인을 조정(adapting)하는 모듈을 통해 구름 세분화를 수행합니다.

- **Technical Details**: Cloud-Adapter는 ConvNet(convolutional neural network)을 활용하여 다중 규모(multi-scale)의 특징을 추출하고, 이를 집계하여 맥락(contextual) 입력으로 사용합니다. 이 방식은 VFM 내 동결된 변환기(transformer) 층을 조절하여 구름 세분화 작업을 효과적으로 수행합니다. 실험 결과, Cloud-Adapter는 훈련 가능한 매개변수의 0.6%만 사용하여도 상당한 성능 향상을 이룹니다.

- **Performance Highlights**: Cloud-Adapter는 여러 위성 소스, 센서 유형, 및 토지 피복 시나리오의 다양한 구름 세분화 데이터셋에서 SOTA(state-of-the-art) 성능을 달성합니다. 특히, 이 방법은 여러 데이터셋의 단지 0.6%의 추가 파라미터로도 우수한 성능을 보여주어, 구름 세분화 작업에서의 견고성을 강조합니다. 본 연구의 소스 코드와 미리 훈련된 모델은 제공되어 후속 연구를 지원합니다.



### Virtual Staining of Label-Free Tissue in Imaging Mass Spectrometry (https://arxiv.org/abs/2411.13120)
Comments:
          33 Pages, 6 Figures

- **What's New**: 이번 연구에서는 영상 질량 분석(Imaging Mass Spectrometry, IMS)의 새로운 가상 조직 염색 방법을 제안합니다. 이 방법은 IMS의 10배 더 큰 픽셀 크기를 사용하면서도, 실험에서 얻은 이미지를 기존의 조직 염색 이미지와 유사하게 재현할 수 있음을 보여줍니다. 이를 통해 IMS의 화학적 특이성과 민감도를 유지하면서도 세포 형태학적 대비를 디지털적으로 강화하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 확산 모델(diffusion model)을 이용하여 라벨 없는 인체 조직에서의 질량 분석 이미지를 향상시킵니다. 무작위 테스트(Blind testing)를 통해 다양한 신장 조직 샘플에서 가상 염색 이미지가 기존의 조직 염색 이미지와 매우 유사하다는 결과를 얻었습니다. 이 과정에서 최적화된 노이즈 샘플링 기법을 활용하여 생성된 이미지의 변동성을 감소시켜 믿을 수 있고 재현 가능한 가상 염색을 가능하게 합니다.

- **Performance Highlights**: 가상 염색 방법은 생명 과학 분야에서 IMS의 적용 가능성을 크게 확대할 것으로 기대됩니다. 연구 결과는 새로운 질량 분석 기반 생물 의학 연구 분야에서의 잠재적인 혁신을 시사하며, IMS 데이터로부터 세포와 조직의 특정 특징을 정밀하게 식별할 수 있는 새로운 가능성을 제공합니다.



### DriveMLLM: A Benchmark for Spatial Understanding with Multimodal Large Language Models in Autonomous Driving (https://arxiv.org/abs/2411.13112)
Comments:
          Code will be available at \url{this https URL}

- **What's New**: 본 논문에서는 자율주행을 위한 공간적 이해 능력을 평가하기 위해 DriveMLLM이라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 2,734개의 전방 카메라 이미지로 구성되어 있으며, 절대 및 상대적 공간 추론 작업을 포함합니다. 또한 자연어 질문이 다양하게 포함되어 있어 MLLMs의 성능을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: DriveMLLM은 nuScenes 데이터셋을 기반으로 하여 구성되었으며, 880개의 이미지를 포함하여 다양한 자연어 질문을 추가했습니다. 이 벤치마크는 객체 간의 공간 이해를 중점적으로 다루며, MLLM 기반의 새로운 평가 메트릭스를 제안합니다. DriveMLLM은 현재의 모델들이 복잡한 공간 관계를 이해하는 데 한계가 있음을 보여줍니다.

- **Performance Highlights**: DriveMLLM을 통해 여러 최신 MLLM 모델의 성능을 평가한 결과, 많은 모델들이 자율주행 공간 이해에서 비효율적임을 확인했습니다. 이 연구는 현재 MLLM 기반 공간 추론 방법의 발전의 필요성을 강조하며, DriveMLLM이 자율주행 연구에 중요한 기여를 할 수 있음을 나타냅니다.



### Superpixel Cost Volume Excitation for Stereo Matching (https://arxiv.org/abs/2411.13105)
Comments:
          13 pages, 7 figures

- **What's New**: 이번 연구에서는 스테레오 매칭(stereo matching)에서 슈퍼픽셀(Neighbors pixels) 소프트 제약(superpixel soft constraints)를 도입하여 예측된 불일치(disparity maps)의 경계에서의 부정확성을 줄이는 데 중점을 두었습니다. 슈퍼픽셀로부터 얻은 인사이트에 기반하여, 네트워크가 각 슈퍼픽셀 내에서 일관된 확률 분포를 생성하도록 유도함으로써, 전반적인 예측 불일치 정밀도 및 일관성을 향상시키고자 하였습니다. 실험 평가를 통해 제안하는 접근 방식이 경쟁적인 성능을 회복시킬 수 있음을 입증했습니다.

- **Technical Details**: 연구에서는 피쳐 추출(feature extraction), 비용 볼륨(cost volume) 구축, 비용 집계(cost aggregation), 및 불일치 회귀(disparity regression)로 구성된 전통적인 비용 볼륨 기반 아키텍처를 채택했습니다. 최적의 매칭을 선택하기 위해 비용 집계 단계에서 지역적 모호성을 해결하는 데 한계를 드러낸 점을 반영하여, 각 슈퍼픽셀에서의 관계 우선사항(pixel relationship prior)을 도입하였습니다. 또한, Laplace 분포를 활용하여 슈퍼픽셀 수준에서의 정답 모델링을 수행하고, 비용 집계 시 발생하는 다중 피크 문제를 완화하는 크로스 엔트로피 손실(cross-entropy loss)을 적용하였습니다.

- **Performance Highlights**: 연구의 결과, 제안한 슈퍼픽셀 기반 접근 방식은 확률 볼륨 내에서 더 일관된 집합을 형성하며, 동시에 계산 리소스를 절약하면서 정확한 확률 표현을 생성했습니다. 이를 통해 스테레오 매칭의 전반적인 정밀도 및 성능이 향상되었음을 입증하였고, 실험 결과를 통해 압축된 장면 구조 정보(scene structural information)를 효과적으로 보존하는 데 기여함을 보여주었습니다.



### Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension (https://arxiv.org/abs/2411.13093)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 Video Retrieval-Augmented Generation (Video-RAG)라는 새로운 접근 방식을 제안하고 있습니다. Video-RAG는 기존의 Large Video-Language Models (LVLMs)의 한계를 극복하기 위해 시각적으로 정렬된 보조 텍스트를 활용하여 비디오 이해를 개선하는 훈련 없는(cost-effective) 파이프라인입니다. 이러한 방식으로 긴 비디오에서도 효과적인 정보 처리가 가능해지며, 기존 LVLM에 비해 효율성과 성능이 크게 향상됩니다.

- **Technical Details**: Video-RAG는 주어진 쿼리에서 보조 텍스트를 추출하는 Retrieval(검색) 요청을 분해한 후, 관련 정보를 여러 개 생성하고 이를 통합하여 LVLM에 입력하는 방식으로 작동합니다. 이 과정에서 Optical Character Recognition (OCR), Automatic Speech Recognition (ASR), Object Detection과 같은 오픈 소스 외부 도구를 사용하여 비디오 데이터에서 관련 정보를 추출합니다. 이러한 방법은 비디오의 시각적 맥락과 일치하여 정보 손실을 최소화합니다.

- **Performance Highlights**: Video-RAG를 통해 다양한 LVLM에 대한 성능 평가를 진행한 결과, Video-MME, MLVU, LongVideoBench 등의 벤치마크에서 평균 8.0%의 성능 향상을 기록했습니다. 특히 72B 모델을 사용할 경우, Gemini-1.5-Pro와 같은 상용 모델에 비해 우수한 성과를 보여주었습니다. 또한, 이 시스템은 추가적인 GPU 메모리와 짧은 추론 시간만으로도 구현할 수 있어 자원의 효율성을 극대화합니다.



### ESARM: 3D Emotional Speech-to-Animation via Reward Model from Automatically-Ranked Demonstrations (https://arxiv.org/abs/2411.13089)
Comments:
          Accepted by the 26th IEEE International Conference on High Performance Computing and Communications (HPCC2024)

- **What's New**: 본 논문은 기존 모델의 한계를 극복하기 위해 새로운 3D 음성-애니메이션(STA) 생성 프레임워크를 제안합니다. 현재의 STA 모델은 감정 표현이 부족하고 다양성이 결여된 애니메이션을 생성하는 경향이 있으며, 이는 인간의 기대에 부합하지 않습니다. 따라서, 본 연구에서는 여러 전략으로 감정을 추출하고 조정함으로써 더욱 감정적으로 풍부한 애니메이션을 생성할 수 있도록 합니다.

- **Technical Details**: 연구에서 제안하는 ESARM은 강화 학습(Reinforcement Learning, RL)을 활용하여 감정 표현이 풍부한 3D 애니메이션 생성 작업을 정의합니다. 나라면, 애니메이션 생성 작업을 Markov 결정 프로세스(Markov Decision Process, MDP)로 형식화하고, 인간의 선호와 일치하는 애니메이션 생성에 대한 보상을 제공합니다. 이 임무는 음성과 FLAME 얼굴 특징을 통해 초기 애니메이션 모델을 훈련시키고, 보상 모델을 훈련하기 위해 노이즈가 추가된 애니메이션을 자동으로 순위를 매기는 방법을 포함합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 프레임워크가 인간의 선호와 잘 일치하며, 감정적으로 풍부하고 다양한 3D 애니메이션을 생성하는데 효과적임을 입증했습니다. 연구 결과는 기존 모델에 비해 생성된 애니메이션의 품질이 현저하게 개선되었음을 보여줍니다. 이는 VR, 게임 및 영화 제작 등의 다양한 분야에서 실제적용 가능성을 의미합니다.



### Practical Compact Deep Compressed Sensing (https://arxiv.org/abs/2411.13081)
Comments:
          Accepted by IEEE T-PAMI

- **What's New**: 이 논문은 압축 센싱(compressed sensing, CS)를 위한 새로운 네트워크인 PCNet을 제안합니다. PCNet은 고유한 협력 샘플링 연산자를 포함하고 있으며, 이는 심층 조건 필터링 단계와 이중 분기 빠른 샘플링 단계로 구성됩니다. 이러한 접근 방식은 임의의 샘플링 속도로 학습된 후 유연성, 해석 가능성, 강력한 복구 성능을 제공합니다.

- **Technical Details**: PCNet은 적응형 로컬 필터링과 이산 코사인 변환(discrete cosine transform), 스크램블된 블록 대각선 가우시안 행렬을 통해 언더샘플링된 측정치를 생성하는 독창적인 구조를 특징으로 합니다. 또한, 향상된 근접 경량 하강 알고리즘으로 구성된 네트워크를 통해 이미지 재구성을 수행합니다. 이는 실제 CS 시스템에서 배포 가능한 방안을 위한 단일 픽셀 CS 이미징 시스템을 위한 추출 계획도 제시합니다.

- **Performance Highlights**: PCNet의 성능은 자연 이미지 CS, 정량화된 CS, 자가 감독 CS에 대한 광범위한 실험을 통해 기존의 최첨단 방법들에 비해 우수한 복구 정확도와 일반화 능력을 보여줍니다. 특히 고해상도 이미지에서 뛰어난 성능을 발휘하며, CS 비율(γ)과 높은 품질의 이미지 재구성이 가능합니다.



### Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving (https://arxiv.org/abs/2411.13076)
- **What's New**: 이 논문에서는 HoP(Hints of Prompt) 프레임워크를 제안하여 자율주행(VQA) 작업에 필요한 세 가지 주요 향상을 도입했습니다. 첫째, Affinity hint는 인스턴스 수준 구조를 강조하여 토큰 간 연결을 강화하고, 둘째, Semantic hint는 차량 및 교통 표지와 같은 도메인 특정 사례와 관련된 고수준 정보를 통합합니다. 셋째, Question hint는 시각적 특징을 질문 맥락에 맞추어 조정하여 질문과 관련된 영역에 초점을 맞춥니다. 이러한 향상들은 Hint Fusion 모듈을 통해 융합되어 자율주행 VQA 작업에서 시각적 표현을 풍부하게 하고 다중 양식(multimodal) 추론을 강화합니다.

- **Technical Details**: HoP 프레임워크는 세 가지 유형의 힌트인 Affinity, Semantic, Question 힌트를 포함하며, 이를 통해 시각적 표현을 정교하게 조정합니다. Affinity hint는 인스턴스 수준 구조를 캡처하며, Semantic hint는 도메인 특정 정보를 제공하여 중요한 인스턴스에 집중하도록 돕습니다. Question hint는 질문 맥락에 따라 시각적 특성을 상관시켜 질문과 관련된 영역에 집중하게 만듭니다. 이러한 힌트는 Hint Fusion 모듈을 통해 통합되어, 자율주행 시나리오에서 VQA의 답변 생성 과정을 개선합니다.

- **Performance Highlights**: 실험 결과, HoP 프레임워크는 LingoQA 벤치마크에서 67.8의 Lingo-Judge 점수를 기록하여 새로운 최첨단 결과를 달성했습니다. 세 가지 힌트 유형은 상호 보완적이며, 시각적 표현 능력을 집단적으로 향상시키는 것으로 확인되었습니다. 또한 다양한 형태의 Semantic 및 Affinity 힌트의 결과는 토큰 간 affinity를 나타낼 때 성능이 향상된다는 것을 보여주었습니다. 이러한 성과는 DRAMA 및 BDD-X 데이터셋에 대해서도 동일하게 나타났습니다.



### Automatic marker-free registration based on similar tetrahedras for single-tree point clouds (https://arxiv.org/abs/2411.13069)
Comments:
          remote sensing; terrestrial lidar; multi-scan cloud registration

- **What's New**: 본 논문에서는 비표식(marker-free) 자동 등록 방법을 제안하여 벌채 포인트 클라우드(terrestrial laser scanning data)를 더 효과적으로 처리할 수 있도록 하였습니다. 이 방법은 유사한 테트라헤드라(tetrahedra)를 기반으로 하여, 기존의 단일 스캔 데이터의 한계점을 극복하고 있습니다.

- **Technical Details**: 제안된 방법은 동일한 나무에 대한 두 개의 스캔으로부터 생성된 나무 스켈레톤(tree skeletons)을 기반으로 하여, 키 포인트 세트(key point sets)를 구축합니다. 이후, 유사성 원칙에 따라 테트라헤드라를 필터링하고 매칭하여, 매칭된 테트라헤드라의 정점(vertices)을 사용해 포인트 쌍(matching point pairs)을 생성합니다. coarse registration 후에는 ICP(Iterative Closest Point) 방법을 적용하여 더욱 정밀한(precise) 등록 파라미터를 도출합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 등록 정확도에서 기존의 ICP 및 NDT 방법보다 크게 우수한 성능을 보였습니다. 속도면에서도 ICP보다 최대 593배, NDT보다 113배 빠른 성능을 기록하여, 단일 나무 포인트 클라우드 등록에 있어 뛰어난 내구성과 효율성을 보여주었습니다. 이 연구는 실질적인 등록 시나리오에서의 적용 가능성이 매우 높음을 시사합니다.



### Towards Unbiased and Robust Spatio-Temporal Scene Graph Generation and Anticipation (https://arxiv.org/abs/2411.13059)
Comments:
          Under review

- **What's New**: 이 논문에서는 Spatio-Temporal Scene Graphs (STSGs)의 생성을 위한 새로운 훈련 프레임워크인 ImparTail을 제안합니다. ImparTail은 커리큘럼 학습(curriculum learning)과 손실 마스킹(loss masking)을 활용하여 긴 꼬리 분포(long-tailed distribution)로 인한 편향(bias)을 줄이는 방법론으로, 희귀한 관계 클래스(tail classes)에 더욱 집중함으로써 전체 관계 유형을 균형 있게 캡처하는 것을 목표로 합니다. 더욱이, 본 연구는 강인한 STSG 모델의 성능을 평가하기 위한 새로운 과제인 Robust Spatio-Temporal Scene Graph Generation과 Robust Scene Graph Anticipation을 도입합니다.

- **Technical Details**: ImparTail 프레임워크는 훈련 과정에서 주요 클래스(head classes)의 지배력을 점진적으로 줄이면서 희귀 클래스에 초점을 맞춰 학습을 진행합니다. 이 과정은 손실 마스킹 기법을 통해 보완되며, 희귀 클래스의 영향을 증대시킵니다. 따라서 실제 데이터에서 발생할 수 있는 분포 변화(distribution shifts)에 대한 저항력을 높이기 위한 평가 작업도 포함되어 있습니다.

- **Performance Highlights**: Action Genome 데이터셋을 사용한 실험 결과, ImparTail은 높은 평균 재현율(mean recall)을 기록하며, 특히 분포 변화가 발생하는 상황에서도 안정적인 성능을 보여주는 것으로 나타났습니다. 기존 방법들과 비교했을 때, ImparTail은 STSG 모델의 편향 없는 성과와 강인성을 크게 향상시키는 것으로 밝혀졌습니다.



### Efficient Masked AutoEncoder for Video Object Counting and A Large-Scale Benchmark (https://arxiv.org/abs/2411.13056)
- **What's New**: 이 논문에서는 비디오 객체 카운팅(video object counting)에서의 전경-배경(dynamic imbalance)에 대한 새로운 접근 방식을 제안합니다. 새로운 
Density-Embedded Efficient Masked Autoencoder Counting (E-MAC) 프레임워크를 도입하여 전경 객체의 희소성(sparsity)으로 인한 문제를 해결하려고 합니다. 더불어, migratory bird 보호를 위한 새로운 비디오 조류 카운팅 데이터셋인 \textit{DroneBird}를 처음으로 제안합니다.

- **Technical Details**: 제안된 E-MAC 프레임워크는 광학 흐름(optical flow) 기반의 시간적 협업 융합(temporal collaborative fusion)을 활용하여, 다수의 프레임에서의 밀도 잔차(multi-frame density residuals)를 유도하여 동적 변화를 효과적으로 캡처합니다. 또한, 밀도 맵(density map)을 보조 모달리티로 사용하여 멀티모달(self-representation learning) 학습을 통해 밀도 맵을 회귀(regress)합니다. 이는 전경 영역에 집중하는데 있어 중복된 배경 정보를 처리하기 위한 효율적인 공간 적응 마스킹(efficient spatial adaptive masking) 기법을 포함합니다.

- **Performance Highlights**: 세 가지 군집 데이터셋과 제안된 \textit{DroneBird} 데이터셋에 대한 실험을 통해, 제안된 방법(E-MAC)이 기존 방법들에 비해 우수한 성능을 보여줍니다. 특히, 여러 프레임의 정보를 활용함으로써 현재 프레임의 카운팅 정확도가 대폭 향상됩니다. 이를 통해 비디오 객체 카운팅의 다양한 문제를 효과적으로 해결할 수 있음을 입증했습니다.



### MEGL: Multimodal Explanation-Guided Learning (https://arxiv.org/abs/2411.13053)
- **What's New**: 이번 연구에서는 Multimodal Explanation-Guided Learning (MEGL) 프레임워크를 제안하여 이미지 분류 작업의 해석 가능성을 높이고 성능을 개선하고자 합니다. 기존의 XAI 방법들이 시각적 혹은 텍스트 기반의 단일 모달 해석에 의존했던 반면, MEGL은 두 가지 모달의 보완적인 특성을 통합하는 접근 방식을 사용합니다. 이를 통해 시각적 및 텍스트 기반 설명의 일관성과 완전성을 높여 AI 모델의 '블랙 박스'(black box) 특성을 해결하고자 합니다.

- **Technical Details**: MEGL에서 제안하는 Saliency-Driven Textual Grounding (SDTG) 방법은 시각적 설명에서 얻은 공간 정보를 텍스트 설명에 통합합니다. 이는 입력 이미지를 처리하여 생성된 Saliency Map을 사용하여 텍스트에서 공간적으로 관련된 통찰을 효과적으로 반영하도록 돕습니다. 또한, Visual Explanation Distribution Consistency loss를 도입하여 시각적 설명의 일관성을 강화하고, 누락된 주석이 있는 경우에도 안정적이고 맥락적으로 타당한 시각적 설명을 보장합니다.

- **Performance Highlights**: Object-ME와 Action-ME라는 두 개의 새로운 데이터셋을 기반으로 실험을 수행한 결과, MEGL은 기존의 이미지 분류 및 해석 가능성 증진 방법보다 우수한 성능을 보였습니다. MEGL은 예측 정확성, 시각적 설명 가능성 및 텍스트 설명 가능성 모두에서 뛰어난 성능을 기록하며, 다양한 훈련 조건에서 유의미하고 일관된 다중모달 설명을 효과적으로 이용할 수 있음을 입증하였습니다.



### Attentive Contextual Attention for Cloud Remova (https://arxiv.org/abs/2411.13042)
Comments:
          13 pages, 7 figures

- **What's New**: 이 연구에서는 Attentive Contextual Attention(AC-Attention)이라는 새로운 주의 메커니즘을 도입하여 구름 제거 과정에서 중요한 장거리 패턴을 효과적으로 탐지합니다. AC-Attention은 데이터 기반의 선택 점수를 동적으로 학습하여 불필요한 노이즈와 관련 없는 특성을 효과적으로 필터링합니다. 이러한 방법은 기존의 주의 메커니즘보다 더 나은 성능을 발휘하여, 구름 제거 결과의 시각적인 품질을 크게 향상시킵니다.

- **Technical Details**: AC-Attention 메커니즘은 두 개의 학습 가능한 모듈을 사용하여 유용한 특성과 정보의 관계를 강화하고, 유해하거나 부정확한 특성 연결을 제거하는 방식으로 작동합니다. 기존의 DSen2-CR 클라우드 제거 네트워크에 AC-Attention 모듈을 통합하여 ACA-CRNet이라는 18층 네트워크 모델을 새롭게 설계하였습니다. 이는 잔여 접합 네트워크와 AC-Attention 블록을 모두 포함하고 있습니다.

- **Performance Highlights**: 다양한 데이터셋을 통해 ACA-CRNet의 성능을 평가한 결과, 기존의 구름 제거 방법과 비교하여 뛰어난 이미지 재구성 품질을 확인했습니다. 아블레이션 연구를 통해 AC-Attention을 여러 기존 방법 및 널리 사용되는 네트워크 아키텍처에 통합하여도 효과성과 적응성이 입증되었습니다. 이는 구름 제거 과정에서 중요한 장거리 의존성을 효과적으로 캡쳐할 수 있음을 보여줍니다.



### RobustFormer: Noise-Robust Pre-training for images and videos (https://arxiv.org/abs/2411.13040)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 RobustFormer라는 새로운 방법을 제안합니다. 이 방법은 Discrete Wavelet Transform (DWT)을 기반으로 하여 이미지와 비디오 데이터의 노이즈에 강한 사전 학습(pre-training)을 가능하게 합니다. 기존 DWT 기반 방법들은 masked autoencoder (MAE) 사전 학습의 이점을 활용하지 못했지만, RobustFormer는 계산 효율성을 높이고 attention 메커니즘을 단순화하여 이러한 한계를 극복합니다.

- **Technical Details**: RobustFormer는 비디오 입력에 호환 가능한 DWT 기반의 masked autoencoder 구조를 처음으로 구현하였습니다. 이 방법은 고립된 저주파 정보에 집중하여 노이즈 및 왜곡에 대한 강건성을 향상시키며, 그림 2와 같이 3D-DWT를 활용하여 시공간(spatio-temporal) 분석을 수행합니다. 또한, iDWT 단계를 삭제하여 계산 비용을 크게 줄임으로써 보다 효율적으로 작동합니다.

- **Performance Highlights**: RobustFormer는 벤치마크 데이터셋인 UCF101 및 HMDB51에서 다양한 노이즈 유형에 대한 포괄적인 평가를 통해 최첨단 성능을 달성했습니다. 실험 결과, MAE 기반의 사전 학습을 통해 iDWT 단계를 우회할 수 있으며, 이미지 및 비디오 작업 모두에서 뛰어난 결과를 보여주었습니다. 본 연구는 다양한 실제 상황에서의 비디오 분류 작업에서 RobustFormer의 유용성을 입증하고 있습니다.



### Unsupervised Homography Estimation on Multimodal Image Pair via Alternating Optimization (https://arxiv.org/abs/2411.13036)
Comments:
          This paper is accepted to the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이번 논문에서는 AltO라는 새로운 비지도 학습 프레임워크를 제안하여 다중 모달 이미지 쌍에서 호모그래피(homography)를 추정합니다. 기존의 비지도 학습 방법들이 동일한 카메라로 촬영된 이미지 쌍에서만 효과적인 반면, AltO는 서로 다른 도메인에서 온 이미지 쌍을 처리할 수 있도록 설계되었습니다. 이 프레임워크는 Expectation-Maximization(EM)과 유사한 두 단계의 교차 최적화 과정을 통해 기하학적 격차와 모달리티 격차를 각각 다룹니다.

- **Technical Details**: AltO는 Geometry loss와 Modality loss를 사용하여 각 기하학적 및 모달리티 격차를 감소시키는 방법을 채택합니다. Geometry loss는 피쳐 수준에서 이미지를 유사하게 만들어 기하학적 격차를 줄이며, Modality loss는 서로 다른 모달리티를 가진 이미지 쌍을 동일한 피쳐 공간으로 매핑하는 역할을 합니다. 이러한 교차 최적화 과정을 통해 ground-truth 데이터 없이도 효과성을 확보할 수 있습니다.

- **Performance Highlights**: AltO는 기존의 비지도 학습 방법들과 비교했을 때 정량적 및 정성적으로 뛰어난 성능을 보여주었으며, 특히 다중 모달 조건에서도 높은 정확성을 유지합니다. 이 방법은 호모그래피 추정기 아키텍처와의 호환성이 뛰어나며, 추가적인 정렬 네트워크와의 결합을 통해 성능을 더욱 향상시킬 수 있습니다.



### X as Supervision: Contending with Depth Ambiguity in Unsupervised Monocular 3D Pose Estimation (https://arxiv.org/abs/2411.13026)
- **What's New**: 최근 비지도 학습 기법들은 제한된 주석 3D 데이터에 대한 의존성을 줄이려는 노력을 기울였으나, 대부분 2D 공간에서만 수립되어 깊이 모호성을 간과하고 있습니다. 이를 극복하기 위해, 우리는 새로운 비지도 학습 프레임워크를 제안합니다. 이 프레임워크는 다중 가설 탐지기와 여러 맞춤형 전제 작업이 결합되어 있으며, 3D 구조의 경험적 분포에 맞춰 해결 공간을 조정합니다.

- **Technical Details**: 우리의 접근법은 새로운 비지도 3D 포즈 추정 프레임워크로, SMPL 모델의 3D 인간 프리셰드를 활용하여 제약 조건을 부여합니다. 다중 가설 탐지기가 단일 히트맵 내에서 여러 가설을 인코딩하여 다중 해결 구조를 관리하며, GCN 기반의 판별기를 통해 예측 분포를 규제합니다. 또한, 합성 이미지를 렌더링하여 탐지기의 제약을 보강합니다.

- **Performance Highlights**: 우리의 방법론은 Human3.6M 및 MPI-INF-3DHP와 같은 다양한 인간 데이터셋에서 최신 비지도 3D 포즈 추정 방법들과 비교하여 우수한 성능을 발휘합니다. 또한, 자연 현장에서 수집된 데이터에 대해 리소스의 확장 가능성을 보여주며 설치된 동물 데이터셋에서도 확장성을 입증합니다. 이러한 혁신은 비지도 단일 뷰 3D 포즈 추정 작업의 깊이 모호성 문제를 해결하는 데 중요한 진전을 가져왔습니다.



### ORID: Organ-Regional Information Driven Framework for Radiology Report Generation (https://arxiv.org/abs/2411.13025)
Comments:
          13 pages, 11 figures, WACV2025

- **What's New**: 이번 연구에서는 Radiology Report Generation (RRG)을 위한 새로운 Organ-Regional Information Driven (ORID) 프레임워크를 제안합니다. ORID 프레임워크는 다중 모달 정보의 효과적인 통합과 관련 없는 장기에서의 노이즈 영향을 줄이는 방법을 포함합니다. 이 연구는 LLaVA-Med 기반의 RRG 관련 지시 데이터세트를 구축하고, 장기를 기반으로 한 교차 모달 융합 모듈과 장기 중요성 계수 분석 모듈을 도입하여 정확하고 신뢰성 있는 방사선 보고서를 생성하는 데 기여하고자 합니다.

- **Technical Details**: ORID 프레임워크는 주로 두 가지 주요 구성 요소로 나뉩니다: 장기 기반 교차 모달 융합 모듈과 장기 중요성 계수 분석 모듈입니다. 이 프레임워크는 10,000개의 질문-응답 쌍을 포함하는 RRG 관련 지시 데이터세트를 활용하여 LLaVA-Med-RRG 모델을 개발하고 방사선 이미지의 장기 수준 분석 능력을 향상시킵니다. 또한, 그래프 신경망 (Graph Neural Network, GNN)을 사용하여 장기 영역 간의 상관관계를 분석함으로써 비관련 장기에서의 노이즈 영향을 최소화합니다.

- **Performance Highlights**: 실험 결과, 제안된 ORID 프레임워크는 두 개의 공공 방사선 보고서 생성 벤치마크에서 새로운 최첨단 성능을 달성했습니다. 이 연구는 방사선 이미지 및 텍스트 모달리티의 효과적인 융합을 통해 작업의 효율성을 증대시키며, 장기 관련 정보의 세밀한 기술이 방사선 보고서 생성에 중대한 영향을 미친다는 것은 뚜렷하게 입증되었습니다.



### Prior-based Objective Inference Mining Potential Uncertainty for Facial Expression Recognition (https://arxiv.org/abs/2411.13024)
- **What's New**: 이 논문은 Facial Expression Recognition (FER) 작업에서의 주관적 주석 모호성을 해결하기 위한 놀라운 해결책으로 Prior-based Objective Inference (POI) 네트워크를 제안합니다. POI는 AUs와 감정에 대한 사전 지식을 활용하여 보다 객관적이고 다양한 감정 분포를 이끌어내며, 주관적 주석의 모호성을 동적 지식 전이를 통해 다룹니다. 이 네트워크는 주관적 감정 주석과 객관적 추론 소프트 레이블을 통합하여 얼굴 표정의 다양성을 이해하는 데 도움을 줍니다.

- **Technical Details**: POI는 두 가지 주요 네트워크로 구성됩니다: Prior Inference Network (PIN)와 Target Recognition Network (TRN)입니다. PIN은 AUs에 대한 사전 지식을 활용하여 얼굴 하위 영역의 미세한 움직임을 포착하며, TRN은 PIN이 제공하는 주관적 및 객관적 감정 레이블을 학습하여 주석 모호성을 해결합니다. 또한, 불확실성 추정 모듈을 도입하여 얼굴 표정의 신뢰도를 정량화하고 균형을 맞춥니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면 POI는 합성 노이즈 데이터셋과 다양한 실제 세계 데이터셋에서 경쟁력 있는 성능을 보였습니다. 이 연구는 주관적 주석의 혼란을 해결하기 위해 객관적 추론을 위한 사전 지식을 활용하는 통찰력을 제공합니다. 최종적으로, POI 네트워크는 널리 알려진 실제 데이터셋과 합성 노이즈 데이터셋에서 효과성을 검증받았습니다.



### Chanel-Orderer: A Channel-Ordering Predictor for Tri-Channel Natural Images (https://arxiv.org/abs/2411.13021)
- **What's New**: 이 논문에서는 Chanel-Orderer라는 모델을 소개하며, 이 모델이 무작위로 순서가 변경된 채널을 가진 일반적인 3 채널 이미지를 처리할 수 있음을 입증합니다. Chanel-Orderer는 객체 의미와 관련된 사전 지식을 활용하여 이미지 채널의 순서를 정확하게 예측하고 조정하는 기능을 갖추고 있습니다. 이로 인해 RGB 포맷의 이미지를 BGR 형식으로 잘못 표시할 때 이를 정확하게 복원하는 데 유용하게 사용될 수 있습니다.

- **Technical Details**: Chanel-Orderer는 손실 함수와 네트워크 아키텍처에서 적절한 inductive biases를 포함하도록 설계되었습니다. 모델은 각 채널을 독립적으로 평가하는 순위 점수로 구성되어 있으며, 이를 통해 이미지를 정확하게 재구성하고 채널 순서를 예측합니다. 이러한 방법은 3개의 채널 간의 쌍을 비교하여 채널 순서를 결정하는 방식으로 진행됩니다.

- **Performance Highlights**: Chanel-Orderer는 기존의 softmax 분류 모델보다 우수한 성능을 보입니다. 특히, 이미지의 채널 순서 예측과 단색성 이미지 예측 작업 모두에서 뛰어난 성능을 발휘했습니다. 이러한 기능은 이미지 처리, 컴퓨터 그래픽스 및 사용자 인터페이스와 같은 다양한 응용 분야에 큰 영향을 미칠 것으로 예상됩니다.



### Open-World Amodal Appearance Completion (https://arxiv.org/abs/2411.13019)
- **What's New**: 이 논문에서는 열린 세계(open-world) 시나리오에서 발생하는 객체의 가리기(occlusion) 문제를 해결하기 위한 Open-World Amodal Appearance Completion 프레임워크를 소개합니다. 이 프레임워크는 자연어 쿼리를 사용하여 사용자가 원하는 객체를 보다 유연하게 지정할 수 있도록 하여, 전통적인 고정된 범주에 제한되지 않습니다. 또한, 추가적인 훈련 없이도 다양한 객체에 대한 amodal completion을 수행할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 이 접근법은 세분화(segmentation), 가리기 분석(occlusion analysis), 및 점진적 인페인팅(inpainting) 기술을 통합한 통합 파이프라인을 기반으로 합니다. 자연어 쿼리를 통해 주어진 입력 이미지에서 대상 객체를 분리하여 가리기를 식별하고, 그 이후에 가려진 부분을 높은 정확도로 재구성합니다. 최종 출력물은 RGBA 형식으로 생성되며, 이는 후속 애플리케이션에서의 매끄러운 통합을 가능하게 합니다.

- **Performance Highlights**: 방대한 평가를 통해, 제안된 접근방식이 새로운 객체 및 복잡한 가리기를 처리하는 데 효과적임을 입증했습니다. 기존 방법들에 비해서 정량적 및 사용자 중심 메트릭에서 우수한 성능을 보이며, 열린 세계(open-world) 설정에서 amodal completion에 대한 새로운 벤치마크를 제시합니다. 이 연구는 특히 고유한 자연어 쿼리를 기반으로 한 직관적인 상호작용을 가능하게 합니다.



### DT-LSD: Deformable Transformer-based Line Segment Detection (https://arxiv.org/abs/2411.13005)
- **What's New**: 본 논문은 새로운 변형된 모델인 Deformable Transformer 기반 선분 탐지기(DT-LSD)를 소개하여 Transformer 기반 방식이 선분 탐지에 광범위하게 채택될 수 있도록 하는 도전 과제를 다룹니다. DT-LSD는 크로스 스케일 상호작용을 지원하며 훈련 속도를 34배 향상시킬 수 있는 Line Contrastive DeNoising (LCDN) 기술을 제안합니다. 이 모델은 이전의 Transformer 기반 모델(LETR)보다 빠르고 정확하며, 모든 CNN 기반 모델보다 높은 정확도를 달성했습니다.

- **Technical Details**: DT-LSD는 변형 가능 주의 메커니즘(deformable attention mechanism)을 사용하여 설명된 대로 내부 및 외부 스케일 처리를 통합합니다. 이 모델은 입력 기능 맵의 채널 수를 일치시키기 위해 1×1 CNN을 사용하여 계층적 기능 맵 세트를 기반으로 기능을 향상시킵니다. 또한, DT-LSD는 선형 대비 잡음 제거(Line Contrastive DeNoising, LCDN) 기술을 도입하여 훈련 과정을 가속화하고, DL 구조에서는 비슷한 단계 수로 수렴을 이룰 수 있도록 합니다.

- **Performance Highlights**: DT-LSD는 Wireframe 데이터셋에서 $sAP^{10}$에서 71.7, $sAP^{15}$에서 73.9의 성능을 달성하였고, YorkUrban 데이터셋에서는 $sAP^{10}$에서 33.2, $sAP^{15}$에서 35.1의 성능을 기록했습니다. 이를 통해 DT-LSD는 기존의 최첨단 방법들에 비해 구조적 및 열 맵 지표 모두에서 성능 개선을 보여줍니다. 본 연구는 선분 탐지 기술이 손으로 제작한 후처리 없이 엔드 투 엔드 방식으로 발전할 수 있는 가능성을 제시합니다.



### Collaborative Feature-Logits Contrastive Learning for Open-Set Semi-Supervised Object Detection (https://arxiv.org/abs/2411.13001)
- **What's New**: 이번 연구에서는 Open-Set Semi-Supervised Object Detection (OSSOD) 문제를 다루는 Collaborative Feature-Logits Detector (CFL-Detector)라는 새로운 방법을 제안합니다. 이를 통해 OOD (Out-of-Distribution) 클래스와 ID (In-Distribution) 클래스를 명확히 구분하며, 두 클래스 간의 혼동을 줄일 수 있습니다. 특히, feature-level clustering과 logits-level uncertainty classification 손실 최적화를 통해 비지도 학습 환경에서도 높은 성능을 낼 수 있는 방법론을 제시합니다.

- **Technical Details**: CFL-Detector는 feature contrastive loss (ℒf⁢c)와 uncertainty classification loss (ℒu⁢c)라는 두 가지 손실 함수를 활용하여 모델을 최적화합니다. feature contrastive loss는 동일 클래스의 특징 간 거리를 줄이고 다른 클래스의 특징 간 거리를 늘려 OOD 클래스의 분리를 강화합니다. 한편, uncertainty classification loss는 ID 클래스와 OOD 클래스에 대한 분류 손실을 최적화하여 모델이 두 클래스의 확률을 학습할 수 있도록 돕습니다.

- **Performance Highlights**: 연구 결과 CFL-Detector는 기존의 SSOD 방법들과 비교하여 최첨단 성능을 달성하는 것으로 나타났습니다. 다양한 실험 및 ablation 연구를 통해 우리의 방법이 OSSOD 문제를 효과적으로 해결할 수 있음을 입증했습니다. 또한, 이 방법은 다른 SSOD 방법에도 쉽게 적용될 수 있어 광범위한 활용 가능성을 지니고 있습니다.



### GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting (https://arxiv.org/abs/2411.12981)
- **What's New**: GazeGaussian은 고충실도의 주시 방향 변환 방법으로, 두 개의 스트림 3D Gaussian Splatting (3DGS) 모델을 사용하여 얼굴과 눈 영역을 각각 표현합니다. 이는 3DGS의 구조적 이점을 활용하여, 이전 방식들과 비교해 계산 효율성과 정밀도를 크게 향상합니다. 특히 이는 gaze redirection 작업에 3DGS를 처음으로 통합하여, 다양한 주체에 대한 일반화 성능을 개선합니다.

- **Technical Details**: GazeGaussian은 패턴을 따르는 사전 훈련된 중립 메쉬를 초기화하여 얼굴과 눈 영역을 분리합니다. 눈의 회전을 정확하게 조정하기 위해 새로운 Gaussian Eye Rotation Representation (GERR)을 제안하며, 이는 기존의 방법들이 암묵적으로 특징 맵을 변경하는 방식과는 달리, 눈 영역의 Gaussian 위치를 명시적으로 조정합니다. 또한, 두 스트림 Gaussian을 고수준 특징으로 변환한 뒤 신경 렌더러에 전송하여 최종 주시 변환 이미지를 합성합니다.

- **Performance Highlights**: 다양한 데이터셋인 ETH-XGaze, ColumbiaGaze, MPIIFaceGaze, GazeCapture에서 GazeGaussian은 주시 변환 정확도 및 얼굴 합성 품질에서 최첨단 성능을 달성합니다. 이 방법은 기존의 gaze estimation 방식들이 GazeGaussian을 활용하여 일반화 성능을 개선할 수 있음을 보여주어 중요성을 더욱 부각합니다. GazeGaussian의 렌더링 속도 또한 경쟁력을 지니고 있어 실제 적용이 기대됩니다.



### LaVida Drive: Vision-Text Interaction VLM for Autonomous Driving with Token Selection, Recovery and Enhancemen (https://arxiv.org/abs/2411.12980)
- **What's New**: 최근 비주얼 언어 모델(Visual Language Models, VLMs)의 발전으로 자율주행 분야에서 시각적 질문 응답(Visual Question Answering, VQA)이 중요해졌습니다. 그러나 기존 방법들은 정적 이미지나 비디오에 주로 초점을 맞추며, 동적인 driving 환경에서는 효과적으로 Spatial과 Temporal 정보를 통합하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 우리는 LaVida Drive라는 새롭고 효율적인 VQA 프레임워크를 소개합니다.

- **Technical Details**: LaVida Drive는 시각적 인식의 세밀함을 유지하면서 높은 해상도의 입력을 통합하고, Temporal 데이터를 통합합니다. 이 프레임워크는 두 가지 모듈로 구성됩니다: 
1. Query-aware Token Selection 모듈은 입력 쿼리와의 의미적 정렬에 따라 가장 관련있는 비주얼 토큰을 동적으로 선택합니다.
2. Spatial-Temporal Token Recovery and Enhancement 모듈은 Spatial과 Temporal 정보 간의 원활한 상호작용을 보장하여 컨텍스트 연속성을 유지합니다.

- **Performance Highlights**: LaVida Drive는 다양한 자율주행 VQA 벤치마크에서 뛰어난 성능을 보여주었습니다. 실험 결과, LaVida Drive는 비주얼 토큰을 50%에서 84%까지 감소시켜 인퍼런스 효율성을 향상시키면서도 전반적인 성능을 유지했습니다. 주요 기여로는 높은 해상도의 Spatial 입력에 Temporal 데이터를 매끄럽게 통합한 혁신적 VQA 프레임워크와, 질문 응답을 위한 Key 정보 추출을 구현한 Query-aware Token Selection 메커니즘이 포함됩니다.



### On the Consistency of Video Large Language Models in Temporal Comprehension (https://arxiv.org/abs/2411.12951)
- **What's New**: 이 논문에서는 Video-LLMs의 일관성을 평가하는 새로운 연구를 제안합니다. 특히, 모델이 동영상 콘텐츠 안에서 시간에 따라 의미 있는 순간을 얼마나 잘 감지하는지를 파악하고자 합니다. 연구 결과, 현재의 Video-LLMs가 영상과 언어 쿼리의 변동에 민감하고 일정한 일관성을 유지하는 데 심각한 결함이 있음을 발견했습니다. 이를 해결하기 위해 VTune이라는 새로운 방법을 제안하여, 일관성 문제를 명시적으로 고려하여 향상된 성능을 보여주고 있습니다.

- **Technical Details**: 알려진 두 개의 데이터 세트인 Charades-STA와 ActivityNet-Captions를 바탕으로 새로운 평가 세트인 Charades-CON과 ActivityNet-CON을 구축하였습니다. 이 데이터 세트는 언어 쿼리와 해당 영상의 타임스탬프를 바탕으로 구성되며, 707개 및 1,422개의 쿼리-순간 쌍이 포함되어 있습니다. 연구에서는 다양한 모델을 평가하여, 알려진 소스 모델과 오픈 소스 모델 간의 일관성 차이를 분석하고, 그 결과에 따라 VTune 방법을 적용하여 성능을 향상시켰습니다.

- **Performance Highlights**: VTune 방법론은 여러 모델과 데이터 세트에서 grounding 및 consistency 모두에서 효과적인 성능 향상 결과를 보여주었습니다. 기존의 다른 방법들과 비교했을 때, VTune은 일관성 있는 시간적 이해력을 위한 보다 안정적인 방법임을 증명했습니다. 이러한 결과는 Video-LLMs가 좀 더 신뢰할 수 있고 강건한 시간적 이해를 제공할 수 있는 가능성을 열어줍니다.



### Enhancing Thermal MOT: A Novel Box Association Method Leveraging Thermal Identity and Motion Similarity (https://arxiv.org/abs/2411.12943)
Comments:
          Workshop on Towards a Complete Analysis of People, part of the European Conference on Computer Vision (ECCV) 2024

- **What's New**: 이번 논문은 열화상 이미지에서의 다중 객체 추적(MOT) 성능을 개선하기 위한 혁신적인 방법을 소개합니다. 저자들은 열 객체의 특성과 운동 유사성을 결합한 새로운 박스 연관(Box Association) 방법을 개발하였습니다. 이러한 방법은 열 특성의 희소성과 동적 객체 추적을 통합하여 MOT의 정확성과 견고성을 향상시킵니다.

- **Technical Details**: 이 논문은 전통적인 운동 연관(Motion Association) 방법에 의존하는 대신, 두 단계 MOT 모델의 연관 과정에서 열상 정체성(Thermal Identity)를 활용하는 새로운 박스 연관 알고리즘을 제안합니다. 이를 통해 알고리즘은 열 데이터의 특성과 운동 데이터를 함께 적용하여 보다 정확한 박스 연관을 달성합니다. 또한 다양한 도시 환경에서 캡처된 열 및 RGB 이미지의 대규모 데이터셋을 새롭게 구축하여 실험과 평가의 기초 자료로 제공합니다.

- **Performance Highlights**: 저자들은 제안한 방법이 기존 방법들에 비해 우수한 성능을 보이며, 다양한 조건에서 추적의 정확성과 견고성을 크게 향상시킨다고 보고합니다. 새로운 데이터셋과 소스 코드는 공개되어 연구자들이 이 방법을 재현하고 확장할 수 있게 합니다. 또한, 이 연구는 모든 센서 모달리티에 대한 MOT 알고리즘의 연구와 개발을 촉진하는 효과를 기대하고 있습니다.



### VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledg (https://arxiv.org/abs/2411.12915)
- **What's New**: 이 논문에서는 의료 분야에 특화된 Vision Language Model (VLM)인 VILA-M3를 제안합니다. 이는 기존의 VLM들이 의료 진단에 필요한 전문가의 도메인 지식을 부족하게 사용하는 것에서 비롯된 한계를 해결하기 위한 것입니다. VILA-M3는 두 가지 단계의 instruction fine-tuning(IFT)을 통해 의료 데이터를 포함한 교육을 강화를 목표로 하며, 의료 전문가 모델에서 정보를 통합합니다.

- **Technical Details**: VILA-M3는 공통 VLM의 약점을 해결하기 위해 네 가지 훈련 단계를 포함합니다: 비전 인코더 사전 훈련, VLM 사전 훈련, IFT 및 도메인 전문가 정보를 포함한 IFT입니다. 이러한 접근법을 통해 VILA-M3는 모델이 언어 능력을 보존하면서도 의료 벤치마크에서의 성능을 높일 수 있게 합니다. 특히, 2D 및 3D 전문 모델과의 효과적인 통합을 통해 하이브리드 융합이 이루어집니다.

- **Performance Highlights**: VILA-M3는 이전의 SOTA 모델인 Med-Gemini에 비해 평균 약 9% 향상된 성능을 보였으며, 특정 과제에 대해 약 6%의 개선 효과를 나타냈습니다. 또한 이 모델은 분할, 분류, 보고서 생성 및 시각적 질문 응답(VQA) 등 다양한 의료 작업을 동시에 처리할 수 있는 최초의 의료 VLM입니다. 이 연구는 의료 응용 프로그램을 위한 정밀하고 신뢰할 수 있는 VLM 생성에서 도메인 전문 지식의 중요성을 강조합니다.



### From Text to Pose to Image: Improving Diffusion Model Control and Quality (https://arxiv.org/abs/2411.12872)
Comments:
          Published at the NeurIPS 2024 Workshop on Compositional Learning: Perspectives, Methods, and Paths Forward

- **What's New**: 최근 2년 동안 텍스트-투-이미지(diffusion models)를 이용한 이미지 생성 모델의 품질과 사용이 증가함에 따라, 출력 제어의 필요성이 대두되고 있습니다. 본 논문에서는 텍스트-투-포즈(text-to-pose, T2P) 생성 모델과 새로운 샘플링 알고리즘을 통해 이미지 생성의 두 가지 주요 문제를 해결했습니다. 특히, 본 기술은 인체 포즈의 높은 충실도를 위한 새로운 포즈 어댑터를 포함하고 있어, 텍스트-투-포즈-투-이미지(generative text-to-pose-to-image) 프레임워크를 가능하게 합니다.

- **Technical Details**: T2P 모델에서는 인체 포즈를 18개의 신체 부위, 42개의 손, 68개의 얼굴 포인트로 설명하고, CLIP을 기반으로한 메트릭을 통해 생성된 포즈의 품질을 평가합니다. 포즈 생성을 위해 텍스트 특징을 조건으로 하는 오토리그레시브(decoder-only transformer architecture)를 활용하였으며, 가우시안 혼합 모델(GMM)을 사용하여 포즈의 연속성을 학습합니다. 실험에서는 4M (포즈, 프롬프트) 쌍을 학습 데이터로 활용하여 T2P 모델의 성능을 검증하였습니다.

- **Performance Highlights**: T2P 모델은 COCO-Pose 벤치마크 데이터셋에서 KNN 방식보다 78%의 높은 성능을 보여주며, 텍스트 프롬프트에 대한 정렬 능력을 입증했습니다. 또한 새로운 포즈 어댑터는 얼굴과 손에 대한 포인트를 포함하여 기존 SOTA(SDXL-Tencent) 모델보다 향상된 포즈 정확도를 보여주었습니다. 그러나 이미지 품질은 여전히 기본 SDXL 모델에는 미치지 못하는 결과를 나타냈습니다.



### Data-to-Model Distillation: Data-Efficient Learning Framework (https://arxiv.org/abs/2411.12841)
Comments:
          Accepted in the 18th European Conference on Computer Vision (ECCV 2024), Milan, Italy, September 29 October 4, 2024

- **What's New**: 본 논문에서는 Data-to-Model Distillation (D2M)이라는 새로운 프레임워크를 제안합니다. D2M은 큰 규모의 실제 데이터셋의 지식을 사전 훈련된 생성 모델의 학습 가능한 파라미터로 증류하여, 다양한 비율의 증류를 위한 훈련 이미지를 효과적으로 생성할 수 있습니다. 이는 높은 해상도의 복잡한 데이터셋에도 효율적으로 확장할 수 있도록 설계되었습니다.

- **Technical Details**: D2M은 생성적 적대 신경망(GAN)의 매개변수 공간 내에서 합성 데이터를 매개변수화합니다. 이는 기존의 픽셀 공간에서 합성 데이터셋을 최적화하는 단점을 해결하며, 모델의 채널 주의 맵과 실제 및 생성 이미지 간의 예측을 최소화하는 모듈을 포함합니다. 이러한 접근법은 분류 성능을 향상시키기 위한 다양한 지식의 증류를 가능하게 합니다.

- **Performance Highlights**: D2M은 15개의 서로 다른 해상도 및 레이블 복잡성을 가진 데이터셋에서 우수한 성능을 보여주었으며, ImageNet-1K에서 고정된 저장 복잡도로 고해상도(128x128)의 데이터를 효과적으로 증류할 수 있음을 입증했습니다. D2M은 기존 데이터셋 증류 알고리즘에 비해 재증류 효율성과 교차 아키텍처 일반화에서 더 나은 성능을 보였습니다.



### HyperGAN-CLIP: A Unified Framework for Domain Adaptation, Image Synthesis and Manipulation (https://arxiv.org/abs/2411.12832)
Comments:
          Accepted for publication in SIGGRAPH Asia 2024. Project Website: this https URL

- **What's New**: 이 연구에서는 사전 훈련된 StyleGAN의 기능을 크게 확장하는 새로운 프레임워크인 HyperGAN-CLIP을 제안합니다. 이 프레임워크는 CLIP 공간을 하이퍼네트워크(hypernetwork)와 통합하여 스타일 생성 모델을 동적으로 새로운 도메인에 적응시킬 수 있도록 합니다. 이를 통해 참조 이미지나 텍스트 설명을 기반으로 한 이미지 생성을 가능하게 하고, 텍스트 안내 하에 이미지를 조작할 수 있는 능력을 강조합니다.

- **Technical Details**: HyperGAN-CLIP은 조건부 하이퍼네트워크를 활용하여 이미지나 텍스트로부터 도메인 특정 임베딩(domain-specific embedding)을 기반으로 제너레이터의 가중치를 동적으로 조정합니다. 이 메커니즘은 CLIP 임베딩을 통해 두 가지 이미지의 특징을 통합하여 생성되는 이미지를 향상시키며, 모드 붕괴(mode collapse)를 방지하여 생성기의 견고성(robustness)을 확보합니다. 또한, 이 프레임워크는 별도의 훈련 없이도 다양한 도메인에 쉽게 적응할 수 있습니다.

- **Performance Highlights**: 우리 프레임워크는 기존의 방법들과 비교하여 강력한 성능과 견고성을 나타냅니다. 여러 도메인과 데이터 셋에 걸쳐 포괄적인 정성적 및 정량적 평가를 수행한 결과, HyperGAN-CLIP의 우수성을 입증했습니다. 텍스트 특정 훈련 데이터 없이도 텍스트 안내 이미지 조작이 가능하며, 스타일 전이(style transfer)가 매끄럽게 이루어지는 점에서 전례 없는 유연성을 보여줍니다.



### Towards motion from video diffusion models (https://arxiv.org/abs/2411.12831)
Comments:
          Accepted at ECCV 2024 Workshop :Foundation Models for 3D Humans

- **What's New**: 최근 비디오 확산 모델들이 실제적인 인간 신체 애니메이션 생성을 유도할 수 있는 가능성이 제시되었습니다. 이 연구는 Score Distillation Sampling(SDS)을 활용하여 SMPL-X 바디 표현을 변형하는 방식으로 인간 동작을 합성하는 방법을 제안합니다. 이를 통해 모델들이 생성한 애니메이션의 충실도를 분석하고, 텍스트-비디오 확산 모델의 적용 가능성을 조사합니다. 이러한 초기 연구가 이 분야의 잠재력을 더욱 밝히고 있습니다.

- **Technical Details**: 연구에서는 인간 동작 생성을 위해 매우 널리 사용되는 SMPL-X 디지털 인간 템플릿 모델을 가이드로 사용합니다. 이 모델을 통해 캐릭터를 렌더링하고, 다층 퍼셉트론(MLP)을 통해 포즈 파라미터를 예측하는 최적화 프로세스를 구현합니다. 비디오 확산 모델은 생성된 동작의 사실성에 대한 피드백을 제공하며, 저자들은 이를 통해 생성된 동작의 품질을 평가합니다. 그러나 일반적이지 않은 동작을 묘사할 때 모델이 겪는 한계를 발견했습니다.

- **Performance Highlights**: 모델은 달리는 동작과 같은 일반적인 행동을 생성하는 데에는 높은 성능을 보였으나, 드물거나 비정상적인 동작에 대해서는 한계가 있음을 보여주었습니다. 본 연구는 MotionDistill이라는 새로운 비디오 생성 파이프라인을 제안하고, ModelScope, ZeroScope, VideoCrafter의 능력을 평가하였습니다. 이러한 평가를 통해 SDS의 효과성을 분석하고, 다양한 인간 동작을 생성하기 위한 비디오 모델의 잠재력과 제한점을 탐구하고 있습니다.



### What Makes a Good Dataset for Knowledge Distillation? (https://arxiv.org/abs/2411.12817)
- **What's New**: 최근 연구에서는 Knowledge Distillation (KD)을 활용한 모델 압축에서 중요한 가정을 재고하려고 합니다. 보통 KD는 teacher의 원본 데이터셋에 접근할 수 있을 것이라는 전제를 두고 있지만, 사실 여러 상황에서는 원본 데이터에 접근이 불가능할 수 있습니다. 이 논문에서는 그런 경우에 활용할 수 있는 대체 데이터셋을 다루고 있습니다.

- **Technical Details**: 연구에서는 teacher에서 student로 지식을 전이하는 데 적절한 데이터셋의 기준을 정의하고 평가합니다. 다양한 대체 데이터셋, 특히 자연스럽지 않은 합성 이미지(synthetic imagery)들도 KD에 유용하다는 것을 보여줍니다. 이러한 대체 데이터를 통해 KD의 성능이 어떻게 달라지는지를 분석합니다.

- **Performance Highlights**: 이 연구는 다양한 데이터셋이 KD에서 좋은 성능을 발휘할 수 있는지를 실험적으로 입증합니다. 또한, 서로 다른 데이터셋의 조합이 지식 전이에 미치는 영향을 분석하여, 최적의 distillation 환경을 만들기 위한 새로운 방향을 제시합니다. 이 연구 결과는 향후 KD 방법론 개선에 기여할 것으로 예상됩니다.



### Interactive Medical Image Segmentation: A Benchmark Dataset and Baselin (https://arxiv.org/abs/2411.12814)
- **What's New**: 이 논문에서는 IMIS(Interactive Medical Image Segmentation)의 연구를 위한 IMed-361M 벤치마크 데이터셋을 소개합니다. 6.4백만 개의 의학 이미지를 수집하고, 이미지마다 평균 56개의 마스크를 포함하여 총 3억 6100만 개의 마스크를 제공합니다. 이 데이터셋은 다양한 의료 이미징 모달리티와 복잡한 임상 시나리오를 아우르며 서로 다른 모델의 일반화와 평가를 지원합니다.

- **Technical Details**: IMed-361M 데이터셋은 14개의 모달리티와 204개의 분할 대상에 걸쳐 있으며, 품질 관리가 철저히 이루어진 고밀도 상호작용 마스크를 자동으로 생성합니다. 이를 통해 사용자는 마우스 클릭, 바운딩 박스, 텍스트 프롬프트와 같은 다양한 입력을 통해 고품질 마스크 생성을 지원받을 수 있습니다. 이러한 데이터셋은 고품질 주석과 대량의 마스크가 결합되어 의료 이미징 분야에서의 '세그먼트 모든 것(segmentation of anything)'을 가능하게 합니다.

- **Performance Highlights**: IMIS의 기준 네트워크를 개발하여 여러 의학적 시나리오에서 그 성능을 평가한 결과, 기존 상호작용 분할 모델보다 우수한 정확도와 확장성을 보여주었습니다. 이는 IMed-361M 데이터셋이 의료 컴퓨터 비전 분야에서 기본 모델 연구를 촉진할 것임을 나타내며, 임상 실습에서 IMIS 기술의 광범위한 채택을 기대할 수 있습니다.



### Stylecodes: Encoding Stylistic Information For Image Generation (https://arxiv.org/abs/2411.12811)
Comments:
          code: this https URL project page: this https URL. arXiv admin note: substantial text overlap with arXiv:2408.03209

- **What's New**: 새로운 방법인 StyleCodes는 이미지 스타일을 20자 base64 코드로 표현하는 개방형 인코더 아키텍처와 훈련 절차를 제시합니다. 기존의 이미지 스타일을 제어하는 방법은 번거로웠으나, StyleCodes를 사용하면 사용자들이 자신의 이미지를 기반으로 쉽게 스타일 코드를 만들 수 있습니다. 이는 MidJourney에서 사용하는 srefs 방식의 문제를 해결하여 이미지의 스타일 제어를 더욱 간편하게 만들어줍니다.

- **Technical Details**: StyleCodes는 이미지 스타일을 표현하기 위해 새로운 encoder를 활용하고 있으며, 이를 통해 기존의 이미지에서 스타일을 추출할 수 있는 절차를 공개적으로 연구할 수 있게 합니다. 훈련 과정은 공개되어 있어, 사용자들이 자신의 데이터로 쉽게 스타일 인코딩을 수행할 수 있는 기회를 제공합니다. 자동차 스타일이 20자의 base64 코드로 인코딩되어, 다양한 스타일 표현이 가능하도록 지원합니다.

- **Performance Highlights**: 실험 결과, StyleCodes는 전통적인 이미지-스타일 (image-to-style) 기법에 비해 품질 손실이 최소화된다는 것을 보여주었습니다. 이는 스타일 코드를 이용한 이미지 생성 시 효과적인 성능을 발휘하며, 사용자들이 원하는 스타일을 더 쉽게 표현할 수 있도록 돕습니다. 이러한 결과는 소셜 미디어에서의 스타일 코드 공유를 촉진할 것으로 기대됩니다.



### CLIC: Contrastive Learning Framework for Unsupervised Image Complexity Representation (https://arxiv.org/abs/2411.12792)
- **What's New**: 본 연구에서는 CLIC라는 비지도 학습 프레임워크를 제안합니다. 이 프레임워크는 이미지 복잡성 표현을 학습하기 위해 라벨링 비용을 피하고 이미지 데이터에서 직접 복잡성 특징을 학습할 수 있도록 설계되었습니다. CLIC는 이미지 복잡성 특징을 강화하기 위한 독특한 샘플 선택 전략과 함께 이미지 사전 기반의 Complexity-Aware Loss를 도입합니다.

- **Technical Details**: CLIC는 query encoder와 key encoder를 사용하여 파라미터를 업데이트하는 방식으로 작동합니다. 연구팀은 이미지 복잡성 표현 작업을 위한 긍정적 및 부정적 샘플 선택 전략을 설계했으며, 이를 통해 모델이 이미지 복잡성 특징을 더욱 잘 학습할 수 있도록 했습니다. 이러한 복잡성 인식 손실(Complexity-Aware Loss)은 카테고리 간의 혼동을 배제하고 모델이 이미지 복잡성을 정확히 배우도록 유도합니다.

- **Performance Highlights**: 실험 결과, CLIC는 이미지 복잡성 표현을 효과적으로 학습할 수 있음을 보여줍니다. 특히, CLIC는 IC9600에서 수퍼바이즈드 방법들과 경쟁력 있는 결과를 보였으며, 다운스트림 태스크에 적용했을 때도 성능이显著히 향상되었습니다. 이는 다양한 실제 시나리오에 적용될 가능성을 입증합니다.



### Mitigating Perception Bias: A Training-Free Approach to Enhance LMM for Image Quality Assessmen (https://arxiv.org/abs/2411.12791)
- **What's New**: 본 연구에서는 이미지 품질 평가(IQA)에서의 다중 모달 모델(LMM)의 편향 문제를 해결하기 위한 혁신적인 접근 방식을 제안합니다. 기존 모델들이 고급 이미지 작업에 최적화되어 있는 데 비해, LMM의 품질 인식 기능을 개선하기 위해 훈련 없이 적용할 수 있는 방법론을 개발했습니다. 이를 통해 LMM의 시맨틱(semantic) 편향으로 인한 인식 문제를 감소시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다: 편향 노출(bias exposure)과 편향 완화(bias mitigation)입니다. 첫 번째 단계에서는 같은 시맨틱을 가진 이미지에서의 편향을 관찰합니다. 두 번째 단계에서는 퀘리 이미지와 훼손된 이미지를 LMM에 입력하고, 조건부 프롬프트를 통해 LMM이 품질 평가를 수행하도록 유도합니다.

- **Performance Highlights**: 다양한 IQA 데이터셋을 통해 수행된 실험 결과, 제안된 편향 완화 프레임워크가 LMM의 성능을 일관되게 향상시켜주었습니다. 또한, 여러 LMM에 대해 일관된 개선을 보여주어 미래 LMM에도 적용 가능성을 입증하였습니다.



### Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models (https://arxiv.org/abs/2411.12790)
- **What's New**: 이번 논문에서는 이미지 내 다수의 상호작용하는 개체들에 대한 정밀한 지식 수정을 목표로 하는 새로운 비주얼 기반의 세분화된 지식 편집 과제를 제안합니다. 기존의 multimodal knowledge editing 방법들이 주로 텍스트 중심, 조잡한 시나리오에 집중했던 반면, 우리는 Fine-Grained Visual Knowledge Editing (FGVEdit) 벤치마크를 통해 이 과제를 평가합니다. 이로써 기존 방법들이 갖는 한계를 극복하고 보다 정교한 지식 수정을 가능하게 하는 접근 방식이 마련되었습니다.

- **Technical Details**: 우리는 Multimodal Scope Classifier-based Knowledge Editor (MSCKE) 프레임워크를 제안하며, 이는 비주얼 및 텍스트 정보를 통합하는 멀티모달 범위 분류기를 활용하여 이미지 내 특정 개체와 관련된 지식을 정확하게 식별하고 업데이트합니다. 이를 위해 기존의 텍스트 기반 편집 방법 SERAC을 바탕으로 하여 텍스트 전용 범위 분류기를 멀티모달 범위 분류기로 교체했습니다. 이 접근 방식은 정밀한 지식 수정을 보장하며 관계가 없는 정보는 보존하는 데 중점을 두고 있습니다.

- **Performance Highlights**: FGVEdit 벤치마크에 대한 광범위한 실험 결과, MSCKE가 기존 방법들보다 뛰어난 성능을 보여주었습니다. 특히, 이미지 내 상호작용하는 다중 개체에 대한 복잡한 지식 편집 도전 과제를 해결하는 데 있어 MSCKE의 유효성이 입증되었습니다. 이러한 성과는 멀티모달 지식 편집의 정확성과 관련성을 높이는 데 기여하고 있습니다.



### Automated 3D Physical Simulation of Open-world Scene with Gaussian Splatting (https://arxiv.org/abs/2411.12789)
- **What's New**: 본 논문에서는 최신 MLLM(다중 모드 대형 언어 모델)을 물리 기반 시뮬레이션에 적용하여 "Sim Anything"이라는 새로운 접근 방식을 제안합니다. 이 방법은 정적 3D 객체에 인터랙티브한 동역학을 부여하는 데 중점을 두며, 복잡한 시뮬레이션을 수월하게 처리할 수 있는 가능성을 제시합니다. 기존 방법들의 수동적 물리 속성 설정의 한계를 극복하는데 기여합니다.

- **Technical Details**: Sim Anything은 MLLM 기반의 물리 속성 인식(MLLM-P3)을 활용하여 객체의 평균 물리 속성을 제로샷 방식으로 예측합니다. 이로 인해 개별 속성을 반복적으로 조정하지 않고도 물리적 속성의 전체 분포를 예측할 수 있습니다. 이를 위해 애드aptive sampling 기법인 PGAS(Physical-Geometric Adaptive Sampling)를 통해 물리적인 상호작용을 시뮬레이션합니다.

- **Performance Highlights**: 광범위한 실험과 사용자 연구 결과에 따르면, Sim Anything은 최신 방법들에 비해 더 사실적인 동작을 신속하게 생성합니다. 단일 GPU에서 2분 이내로 더 높은 물리적 속성을 예측하고 실시간 동역학을 생성할 수 있으며, 이는 실제 응용에 매우 유리합니다. 이러한 성과는 기존의 물리 기반 모델들과의 비교를 통해 입증되었습니다.



### Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification (https://arxiv.org/abs/2411.12788)
- **What's New**: 본 연구에서는 Gaussian Splatting을 위한 빠른 장면 최적화의 핵심 과제를 탐색합니다. 우리는 기하학 모델링 과정의 철저한 분석을 통해, 밀집 포인트 클라우드(dense point clouds)가 Gaussian 표현을 통해 최적화 초기에 효과적으로 재구성될 수 있음을 밝혀냈습니다. 이를 바탕으로 공격적인 Gaussian 밀집화(aggressive Gaussian densification) 접근법을 개발하여 기존의 점진적 밀집화 방법보다 더 효율적인 대안을 제공합니다.

- **Technical Details**: 우리는 3DGS(3D Gaussian Splatting)에서 기하학 모델링 과정에 대한 조사를 실시하며, Gaussian 중심을 시각화했습니다. 제한된 최적화 반복을 가진 3D Gaussian은 눈에 띄는 아티팩트를 보이나, 해당 장면 표현에서 밀집 포인트 클라우드를 추출하는 것은 여전히 가능함을 확인했습니다. 공격적인 Gaussian 밀집화는 중요한 Gaussian 수를 dramatically 증가시켜 밀집 Gaussian 재구성을 상당히 단축된 시간 안에 가능하게 합니다.

- **Performance Highlights**: Mini-Splatting2는 최적화 시간, Gaussian 수 및 렌더링 품질 간의 균형 잡힌 절충을 달성하였으며, 이는 향후 Gaussian Splatting 기반 작업의 강력한 기준을 확립합니다. 우리의 방법론은 실시간 응용 프로그램에서 더 효율적이고 고품질의 3D 장면 모델링을 위한 토대를 마련하고 있으며, 코드도 공개될 예정입니다.



### Visual Cue Enhancement and Dual Low-Rank Adaptation for Efficient Visual Instruction Fine-Tuning (https://arxiv.org/abs/2411.12787)
- **What's New**: 이 논문에서는 새로운 접근법인 Vision Cue Enhancement (VCE)와 Dual Low-Rank Adaptation (Dual-LoRA)을 제안하여 다중 모달 대형 언어 모델(MLLMs)의 미세 조정 프레임워크를 개선합니다. VCE는 고수준 시각적 특징에 의존하는 기존 방법의 한계를 극복하기 위해 다층 시각적 단서를 통합하여 모델의 세밀한 시각적 정보 포착 능력을 향상시키고 있습니다. Dual-LoRA는 기술적 및 작업적 학습 공간을 분리하여 다양한 작업 간의 효율적인 조정을 목표로 합니다.

- **Technical Details**: 저자들은 Vision Cue Enhancement (VCE) 모듈을 통해 다층 시각적 특성 맵을 활용하여 각 패치를 개선합니다. 이를 통해 더 지역적인 시각적 특성이 캡처되어 시각적 이해력이 개선됩니다. 또한, Dual Low-Rank Adaptation (Dual-LoRA)은 특정 기술 지식을 학습하는 기술 저차원 공간과 이전 지식을 활성화하는 작업 활성화 저차원 공간을 결합하여 데이터 충돌을 완화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다운스트림 작업 및 일반 MLLM 벤치마크에서 우수한 성능을 나타냅니다. VCE 모듈은 고급 비주얼 기능을 더욱 풍부하게 만들어 다중 작업에서의 효과적인 비주얼 지침 미세 조정을 가능하게 하며, Dual-LoRA를 통해 각 작업에 맞는 높은 세부 사항을 유지하면서도 데이터 충돌 문제를 해결합니다.



### Joint Vision-Language Social Bias Removal for CLIP (https://arxiv.org/abs/2411.12785)
- **What's New**: 이 논문에서는 기존의 Vision-Language (V-L) 모델들이 지니고 있는 사회적 편향 문제를 조명합니다. 다양한 사회적 그룹에 대한 편향된 예측을 생성하는 V-L 모델들의 한계와 이를 해결하기 위한 새로운 접근 방법이 제안됩니다. 특히, 이미지와 텍스트의 편향을 동시 제거하는 새로운 V-L 디바이싱(debiasing) 방법이 도입되었습니다.

- **Technical Details**: 제안된 방법은 이미지와 텍스트 임베딩 간의 편향을 정렬한 후 이를 제거하는 구조를 가지고 있습니다. 편향 정렬(bias alignment) 작업을 통해 두 모달리티에서 편향 임베딩을 정렬하고, 반대 방향의 편향 수정을 위한 목적을 통해 중립 정보를 유지하면서 편향을 제거합니다. 이는 CLIP-clip 방법과는 다른 점으로, V-L 정렬을 유지하며 디바이싱을 수행합니다.

- **Performance Highlights**: 이 방법은 다양한 백본(backbone) 모델에서 우수한 성능을 보이며, 다수의 사회적 편향 유형에 대한 공동 디바이싱을 가능합니다. 또한, 새로운 평가 프로토콜을 통해 모델의 편향 제거 및 V-L 정렬 능력을 종합적으로 평가할 수 있는 기준을 마련했습니다. 실험 결과, 제안된 방법이 기존의 여러 기법들에 비해 현저히 개선된 성능을 보여주었습니다.



### Med-2E3: A 2D-Enhanced 3D Medical Multimodal Large Language Mod (https://arxiv.org/abs/2411.12783)
- **What's New**: 논문에서 제안하는 Med-2E3은 3D 의료 영상 분석을 위한 최초의 다중 모드 대형 언어 모델(Multimodal Large Language Model, MLLM)로, 3D 및 2D 인코더를 통합한 혁신적인 접근 방식입니다. 기존 MLLM은 3D 의료 영상에 포함된 풍부하고 계층적인 정보를 완전히 활용하지 못했으며, Med-2E3은 텍스트 가이드 인터슬라이스(Text-Guided Inter-Slice, TG-IS) 점수 모듈을 통해 2D 슬라이스의 주의를 효과적으로 점수화할 수 있습니다. 이를 통해 다양한 임상 시나리오에서의 일반화 가능성을 향상시킵니다.

- **Technical Details**: Med-2E3는 2D 및 3D 인코더를 통합하여 3D 의료 영상 내의 전반적(global) 및 국부적(local) 구조를 모델링합니다. 특히, TG-IS 점수 모듈은 각 2D 슬라이스의 내용과 작업 지침을 기반으로 주의를 측정하여 보고서 생성 및 의료 시각적 질문 응답(Visual Question Answering, VQA) 작업에서 우수한 성능을 나타냅니다. 이 모델은 의료 데이터의 특성을 반영한 혁신적인 구조를 가진 MLLM으로, 기존 방법들이 초래하던 표현 용량의 한계를 극복합니다.

- **Performance Highlights**: 대규모 오픈 소스 3D 의료 다중 모드 벤치마크에서 Med-2E3이 현재의 최신 모델에 비해 보고서 생성에서 14% 개선, 의료 VQA에서 5% 향상된 성능을 보였습니다. 이로 인해 Med-2E3은 복잡한 다중 모드 임상 작업을 해결하는 데 큰 잠재력을 지니고 있다는 점이 강조됩니다. 논문은 Med-2E3의 소스 코드를 수용할 경우 공개할 예정입니다.



### FGP: Feature-Gradient-Prune for Efficient Convolutional Layer Pruning (https://arxiv.org/abs/2411.12781)
- **What's New**: 이 논문은 기능 기반(feature-based)과 경량화(gradiant-based) 정보를 통합하여 채널의 중요성을 평가하는 새로운 가지치기(pruning) 방법인 Feature-Gradient Pruning(FGP)를 제안합니다. FGP 방법은 모든 클래스에 대해 중요한 채널을 식별하면서도 특정 클래스에만 유용한 중복 채널을 제거하여 세분화된 가지치기 최적화를 달성합니다. 이로 인해 모델의 compactness(압축성)와 practical value(실용성)가 향상됩니다.

- **Technical Details**: Feature-Gradient Pruning(FGP)은 feature 기반 및 gradient 기반 채널 중요도 평가를 활용하여 모델 성능을 위한 중요한 채널을 보다 정확히 식별하는 방법입니다. 기존의 가지치기 방법들은 이미지 피쳐 또는 그래디언트에만 의존하여 중복 채널을 보존하는 경향이 있었으나, FGP는 두 가지 정보를 통합함으로써 이러한 문제를 해결합니다. 이 방법은 지원값의 집중도를 기반으로 하여 최상위 k(k)의 중요 채널을 선택하고 가지치기 비율을 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과, FGP는 기존 방법들과 비교하여 계산 비용을 크게 절감하면서 정확도 손실을 최소화하는 성과를 보였습니다. 다양한 작업과 데이터셋에서 수행된 실험에서 FGP는 모델의 compactness를 높이고 실용성을 유지하면서 안정적인 성능을 유지하는 것을 입증했습니다. 이러한 결과는 FGP가 가지치기 최적화에 효과적임을 강조합니다.



### Faster Multi-GPU Training with PPLL: A Pipeline Parallelism Framework Leveraging Local Learning (https://arxiv.org/abs/2411.12780)
- **What's New**: 본 논문에서는 효율적인 병렬 학습을 위해 다수의 GPU를 활용하는 새로운 프레임워크인 PPLL (Pipeline Parallelism based on Local Learning)을 소개합니다. 기존의 모델 병렬화 방식은 통신 지연과 동기화 문제로 인해 병렬성을 제대로 활용하지 못하며, 이러한 한계를 극복하기 위해 PPLL은 모델을 여러 개의 블록으로 나누고 각 블록을 별도의 GPU에 할당합니다. PPLL은 데이터 전송을 관리하기 위해 대기열(queues)을 사용하여 GPU 간의 무결점(completion) 통신을 보장하며, 이를 통해 모델의 전방 및 후방 전파(forward and backward passes)가 파이프라인 형태로 동시에 수행될 수 있습니다.

- **Technical Details**: PPLL은 로컬 학습(local learning) 알고리즘을 활용하여 모델을 각자 다른 GPU에 할당된 여러 개의 분리된 블록으로 나누는 방식입니다. PPLL은 Message Queue를 통해 데이터 전송을 관리하여 GPU 간 원활한 통신을 가능하게 합니다. 이로 인해 다수의 블록이 연속적으로 전방 및 후방 전파를 수행할 수 있어 유휴 시간(idle time)을 최소화하고, 일반적인 순차적 그래디언트 업데이트에서 발생하는 병목 현상을 방지합니다. 이를 통해 PPLL은 전반적인 학습 과정을 크게 가속화하게 됩니다.

- **Performance Highlights**: PPLL의 유효성은 CIFAR-10, SVHN, STL-10 데이터셋에서 ResNet 및 Vision Transformer (ViT) 아키텍처를 사용하여 광범위한 실험을 통해 검증되었습니다. 실험 결과, PPLL은 기존의 파이프라인 병렬 처리에 비해 모델 성능을 유지하며 학습 속도를 현저히 향상시켰습니다. 4개의 GPU를 사용하는 경우, PPLL은 ViT에서 162%, ResNet에서 33%의 학습속도 향상을 보여주었으며, 이는 전통적인 파이프라인 병렬 처리의 속도에 비해 각각 1.25배, 0.85배에 해당합니다.



### Decoupling Training-Free Guided Diffusion by ADMM (https://arxiv.org/abs/2411.12773)
- **What's New**: 이번 논문에서는 조건부 생성 문제를 해결하기 위해 기존의 비조건부 확산 모델을 차별화된 손실 함수로 유도하여 새로운 프레임워크를 제안합니다. 기존 연구들이 하이퍼파라미터를 조정하여 비조건부 모델과 유도 손실 간의 균형을 맞추는 데 집중한 반면, 우리는 이 두 요소를 명확하게 분리함으로써 접근합니다. 또한, 샘플링을 위해 두 개의 변수를 도입하여 조건부 생성을 두 개의 관리 가능한 하위 문제로 재구성합니다.

- **Technical Details**: 우리는 비조건부 확산 모델에서 생성된 샘플을 나타내는 변수 ${x}$와 유도 함수에서 제어된 샘플을 나타내는 보조 변수 ${z}$를 도입하여, 두 요소 간의 관계를 ${x} = {z}$라는 제약 조건으로 연결합니다. 이를 통해 교대 방향 방법 (Alternating Direction Method of Multipliers, ADMM)을 기반으로 하는 새로운 알고리즘을 개발하였으며, 하이퍼파라미터를 사용하지 않고 자연스럽고 적응적인 균형을 가능하게 합니다. 알고리즘의 수렴성을 통계적 가정 하에 이론적으로 증명하였으며, 이 과정에서 확산 모델의 역 과정과 ADMM의 근사 연산자 간의 동등성을 성립시켰습니다.

- **Performance Highlights**: 제안된 ADMMDiff 방법은 다양한 조건부 생성 작업에서 우수한 성능을 보여주었으며, 고품질 샘플을 생성하는 동시에 강력한 조건 준수성을 보장합니다. 실험 결과를 통해 이미지 생성을 포함한 여러 조건부 생성 작업에서 기존 방법보다 성능이 뛰어난 것을 확인했습니다. 특히, 다양한 유도 조건에 기반한 이미지 생성 및 특정 경로를 따르는 모션 합성을 성공적으로 수행하였습니다.



### AI-generated Image Detection: Passive or Watermark? (https://arxiv.org/abs/2411.13553)
- **What's New**: 이 논문에서는 AI가 생성한 이미지를 탐지하기 위한 최초의 포괄적인 벤치마크인 ImageDetectBench를 제안합니다. 이 벤치마크는 수동 탐지기와 워터마크 기반 탐지기 간의 효과성, 견고성, 효율성을 비교하기 위해 설계되었습니다. 연구팀은 AI가 생성한 이미지와 비-AI가 생성한 이미지를 포함한 네 가지 데이터 세트를 구성하고, 다양한 왜곡(perturbations)과 공격에 대해 탐지기의 성능을 평가하였습니다.

- **Technical Details**: ImageDetectBench는 8가지 일반적인 왜곡과 3가지 적대적 왜곡을 포함하여, 수동 탐지기와 워터마크 기반 탐지기의 비교를 위한 체계적인 실험을 실시합니다. 연구에서는 5개의 수동 탐지기와 4개의 워터마크 기반 탐지기를 평가하였으며, 각 탐지기의 설계는 이미지 간의 미세한 아티팩트를 탐지하는 데 중점을 두고 있습니다. 또한, 워터마크 기반 탐지기는 AI가 생성한 이미지를 구별하기 위해 이미지 생성 시 인간이 인지할 수 없는 아티팩트를 삽입하는 방식으로 작동합니다.

- **Performance Highlights**: 연구 결과, 워터마크 기반 탐지기가 모든 시나리오에서 수동 탐지기보다 일관되게 우수한 성능을 보였으며, 특히 왜곡이 없거나 일반적인 왜곡, 적대적 왜곡이 있는 경우에서도 그 우수함을 유지하였습니다. 가장 견고한 워터마크 기반 탐지기는 가장 견고한 수동 탐지기보다도 두 배 이상 빠른 속도로 작동하여, 일반 사용자와 공격자 모두에게 효과적으로 대응할 수 있는 장점을 보여주었습니다. 따라서 두 종류의 탐지기가 모두 적용 가능한 경우에는 워터마크 기반 탐지기를 우선적으로 사용하는 것이 권장됩니다.



### Comparative Analysis of Machine Learning and Deep Learning Models for Classifying Squamous Epithelial Cells of the Cervix (https://arxiv.org/abs/2411.13535)
Comments:
          15 pages, 4 figures

- **What's New**: 이 연구는 인공지능(AI)을 이용한 자궁경부 세포의 자동 분류를 목표로 하고 있습니다. 기존의 Pap smear 방법은 시간이 많이 소요되고 인력에 의존하며 인간 오류가 발생할 가능성이 있으므로 AI 기반의 접근이 필요합니다. 이 방법은 구역에 있는 세포들을 다섯 가지 카테고리로 분류하는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 Gradient Boosting, Random Forest, Support Vector Machine, k-Nearest Neighbor 등 다양한 기계학습(ML) 알고리즘과 ResNet-50 같은 딥러닝(DL) 방법을 사용하였습니다. 연구팀은 Pap smear 이미지를 통해 세포를 분류하는 데 있어 ML 모델들이 높은 분류 정확도를 보였지만, ResNet-50이 93.06%의 정확도로 가장 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 자궁경부암 조기 진단을 위한 세포 수준의 분류에서 DL 모델의 효율성을 강조합니다. AI 기반의 자동화된 세포 분류 방법이 다르면 개발 기간을 단축시키고 인적 오류를 줄여 더 정확한 진단을 가능하게 할 것으로 기대됩니다.



### Efficient Brain Imaging Analysis for Alzheimer's and Dementia Detection Using Convolution-Derivative Operations (https://arxiv.org/abs/2411.13490)
- **What's New**: 이번 연구에서는 알츠하이머병(AD)과 같은 신경퇴행성 질환과 관련된 볼륨 변화 분석을 위한 Sobel Kernel Angle Difference (SKAD) 방법을 소개합니다. 기존의 Jacobian map에 비해 6배 이상의 계산 효율성을 자랑하며, 경쟁력 있는 정확도로 신경영상 연구 및 임상 실습에 있어 효율적이고 실용적인 접근법을 제공합니다.

- **Technical Details**: SKAD는 기울기를 지역적으로 분석하여 볼륨 변화를 정량화하는 최적화된 접근 방식을 제공합니다. 이 방법은 MRI 스캔의 기울기 진폭 변화(gradient amplitude changes)를 효율적으로 추출하여, 중요한 공간 영역에서의 볼륨 변화를 캡처합니다. SKAD는 계산 시간을 84% 줄이며, 부동소수점 연산을 99% 감소시켜 보다 효과적인 체적 변화 측정을 가능하게 합니다.

- **Performance Highlights**: 여러 의료 데이터셋에서 SKAD의 평가 결과, Jacobian map보다 6.3배 빠르면서도 유사한 정확도를 유지하는 것으로 나타났습니다. 따라서, SKAD는 신경영상 연구 및 임상의에서 상용화될 수 있는 유망한 방법으로 자리매김할 것으로 기대됩니다.



### Unification of Balti and trans-border sister dialects in the essence of LLMs and AI Technology (https://arxiv.org/abs/2411.13409)
Comments:
          Accepted by IEEE conference ISCSLP 2024

- **What's New**: 이번 논문은 Balti 언어의 중요성과 그 방언들의 통합 노력을 다룹니다. 이 언어는 중국어-티베트어족(Sino-Tibetan) 소속으로, 인도, 중국, 파키스탄 등 여러 국가에서 다양한 방언이 존재합니다. 인공지능(AI) 기술의 발전에 따라, 방언 통합을 통한 공통성 이해의 필요성이 강조됩니다.

- **Technical Details**: Balti 언어의 분석 및 문서화는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 수행됩니다. 이 연구에서는 다양한 방언 간의 기본적인 어근(root), 어휘(lexica), 음운론적 관점에서의 공통점을 찾아내는 방법을 제시합니다. 이러한 접근 방식은 언어 보존에 기여할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: AI를 활용한 Balti 언어의 분류 및 표준화 노력은 여러 방언들의 문서화와 분석을 통해 이루어집니다. 글로벌화가 진행되는 현재, AI 기술은 언어 다양성 이해 및 방언 통합에 중요한 역할을 할 수 있습니다. 이 연구는 인공지능 기술이 어떻게 언어 보존에 기여할 수 있는지를 탐구합니다.



### Adversarial Diffusion Compression for Real-World Image Super-Resolution (https://arxiv.org/abs/2411.13383)
- **What's New**: 이 논문에서는 AdcSR이라는 새로운 Real-ISR 방법을 제안하는데, 이는 OSEDiff의 한 단계 확산 네트워크를 Adversarial Diffusion Compression (ADC) 프레임워크를 통해 간소화된 diffusion-GAN 모델로 증류하는 방식입니다. AdcSR은 느린 다단계 추론 문제를 해결하면서, 모델의 생성 능력을 유지하는 데 중점을 두고 있습니다. 제안된 방법은 기존의 방법들보다 빠르고 효율적인 속도를 제공합니다.

- **Technical Details**: AdcSR 모델은 두 가지 유형의 모듈을 삭제 및 가지치기하는 구조적 압축 방법을 사용하여 구현됩니다: 1) 제거 가능한 모듈(예: VAE 인코더, 텍스트 인코더 등) 및 2) 가지치기 가능한 모듈(예: 디노이징 UNet 및 VAE 디코더). 이 모델은 프리트레인된 VAE 디코더를 통해 생성 능력을 복구하고, 적대적 증류를 통해 성능 손실을 보완합니다.

- **Performance Highlights**: AdcSR 모델은 기존의 한 단계 확산 기반 방법들에 비해 최대 9.3배의 속도 향상을 달성하며, 추론 시간, 계산량 및 파라미터 수를 각각 73%, 78%, 74% 줄였음에도 불구하고 경쟁력 있는 복원 품질을 유지하고 있습니다. 실험 결과, AdcSR은 합성 및 실제 데이터셋 모두에서 우수한 성능을 보였습니다.



### RTSR: A Real-Time Super-Resolution Model for AV1 Compressed Conten (https://arxiv.org/abs/2411.13362)
- **What's New**: 이번 논문에서는 비디오 콘텐츠의 시각적 품질을 향상시키기 위해 설계된 저복잡도의 슈퍼 해상도(Super Resolution, SR) 기법인 RTSR을 제안하고 있습니다. RTSR은 360p에서 1080p, 540p에서 4K로 해상도를 업스케일하는 데 중점을 두고 있으며, 복잡한 신경망 구조를 최소화하여 실시간 비디오 재생을 가능하게 합니다. 또한, 이 방법은 다양한 양자화 수준에서 AV1 인코딩된 콘텐츠에 최적화되어 구현되었습니다.

- **Technical Details**: RTSR 기법은 CNN 기반의 네트워크 아키텍처를 활용하여 비디오 프레임을 처리합니다. 입력으로는 YCbCr 4:2:0 형식의 압축된 이미지 블록을 사용하며, 이를 인접한 블록으로 업샘플링한 후, 최종적으로 원본 비디오의 모든 해상도로 재구성합니다. 두 개의 교사 모델인 CVEGAN과 EDSR_baseline을 이용한 지식 증류(Knowledge Distillation) 방법을 적용하여 성능을 향상시키고, 시각적으로 유도된 손실 함수를 활용하여 최적화 과정에서 인식을 개선합니다.

- **Performance Highlights**: RTSR은 AIM 2024 비디오 슈퍼 해상도 챌린지에서 효율성과 성능 간의 균형을 가장 잘 맞춘 솔루션으로 평가받았습니다. PSNR, SSIM, VMAF 등 다양한 품질 메트릭에서 다른 다섯 가지 솔루션보다 우수한 성능을 보였으며, 이는 간소한 복잡도를 유지하면서도 뛰어난 비디오 품질을 자랑합니다. 논문의 부록에서는 실험 조정 과정과 결과에 대한 세부 내용을 분석하고, 향후 연구 방향에 대해서도 제시합니다.



### Analysis and Synthesis Denoisers for Forward-Backward Plug-and-Play Algorithms (https://arxiv.org/abs/2411.13276)
- **What's New**: 이 논문에서는 Forward-Backward (FB) 알고리즘의 동작을 연구하며, proximity operator(근접 연산자)를 Gaussian denoiser(가우시안 노이즈 제거기)를 근사화하기 위한 서브-이터레이션 프로세스으로 대체합니다. Plug-and-Play(PnP) 프레임워크 내에서 분석 및 합성 Gaussian denoiser를 다루며, 이중 FB 반복 또는 FB 반복을 통해 얻어집니다.

- **Technical Details**: 연구팀은 FB-PnP 반복의 비대칭 문제를 분석하고, 이 과정을 통해 근접 연산자 역할을 하는 합성 가우시안 노이즈 제거 문제를 도출합니다. 각 경우에서, FB-PnP 알고리즘은 노이즈 제거 문제를 해결하기 위해 하나 또는 무한 개의 서브-이터레이션을 사용할 때 동일한 문제를 해결함을 보입니다. 또한, Moreau-Yosida smoothing을 사용할 때 유사한 결과를 보여줍니다.

- **Performance Highlights**: 이 논문은 이론적 결과를 설명하기 위한 수치 시뮬레이션을 제공합니다. 그 중에서도 장난감(compressive sensing) 예제와 깊은 사전(deep dictionary) 프레임워크에서의 이미지 복원 문제를 다루며, FB-PnP 알고리즘의 유용성을 강조합니다.



### An Integrated Approach to Robotic Object Grasping and Manipulation (https://arxiv.org/abs/2411.13205)
Comments:
          5 PAGES

- **What's New**: 아마존은 창고 운영에서 수작업과 효율성의 문제를 해결하기 위해 로봇 기술을 도입하여 큰 변화를 시도하고 있습니다. 기존에는 아이템 운반과 같은 작업을 수행하기 위한 로봇이 많이 배치되었지만, 선반에서 물체를 선택하는 과정은 여전히 도전 과제로 남아 있습니다. 이 프로젝트는 그러한 문제를 해결하기 위해 특정 아이템을 효율적으로 선택하는 혁신적인 로봇 시스템을 개발했습니다.

- **Technical Details**: 제안된 로봇 시스템의 큰 특징은 선반의 각 빈에 있는 불확실한 물체 위치를 탐색할 수 있는 능력입니다. 이 시스템은 자율적으로 접근 방식을 조정하며, 사전에 그 위치에 대한 정보가 없는 경우에도 원하는 아이템을 효율적으로 찾아내고 획득할 수 있는 전략을 사용합니다. 이러한 기술은 로봇이 실제 창고 환경에서 더 나은 성능을 발휘할 수 있도록 합니다.

- **Performance Highlights**: 이 로봇 시스템은 복잡한 물체 선택 작업을 자율적으로 수행할 수 있어, 물류 및 창고 운영의 효율성을 크게 향상시킬 수 있습니다. 도전적인 조건에서도 제대로 기능할 수 있는 이 시스템은 물체가 어디에 있는지 모르는 경우에도 원하는 아이템을 성공적으로 찾아낼 수 있도록 설계되었습니다.



### Intensity-Spatial Dual Masked Autoencoder for Multi-Scale Feature Learning in Chest CT Segmentation (https://arxiv.org/abs/2411.13198)
Comments:
          10 pages,6 figures,3 tables

- **What's New**: 본 논문에서는 Intensity-Spatial Dual Masked AutoEncoder (ISD-MAE)라는 새로운 방식의 의료 이미지 분할 기법을 제안합니다. 이 방식은 반 마스크 오토인코더(tissue-contrast semi-masked autoencoder)의 기초 위에 Masked AutoEncoder (MAE) 분기를 추가하여 다양한 스케일의 특징을 학습하고 분할 과제를 수행할 수 있도록 합니다. ISD-MAE는 이중 분기 구조와 대조 학습(constractive learning)을 활용하여 조직의 특징과 경계 세부사항을 보다 효과적으로 학습합니다.

- **Technical Details**: ISD-MAE는 두 가지 관계에 따라 작동합니다. 첫째, 강도 마스킹(intensity masking)과 공간 마스킹(spatial masking) 작업을 수행하여 흉부 CT 이미지에서 다중 스케일 특징을 학습합니다. 둘째, 컨트라스트 학습을 통해 이미지의 구조적 정보와 다양한 조직 특성을 포착하여 의료 이미지 분할 및 분류 성능을 최적화합니다. 실험은 2D와 3D의 여러 데이터셋에서 수행되며, ISD-MAE는 특히 폐렴 및 종양 분할 작업에서 기존 방법들에 비해 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: ISD-MAE는 COVID19 LESION 데이터셋에서 Dice 점수 90.10%를 기록하며, 2D 폐렴 및 수막 종양 분할 작업에서 매우 높은 성능을 나타냅니다. 하지만 3D 데이터셋에서의 성능은 아직 개선의 여지가 있어, 추후 손실 함수 최적화, 향상된 3D 컨볼루션 블록 활용 등을 통해 보완할 계획입니다. 이러한 연구 결과는 향후 의료 이미지 분할의 정확성과 안정성을 더욱 향상시킬 것으로 기대됩니다.



### SONNET: Enhancing Time Delay Estimation by Leveraging Simulated Audio (https://arxiv.org/abs/2411.13179)
- **What's New**: 이번 연구에서는 시간을 기반으로 한 차이를 추정하는 방법인 Time-Difference-Of-Arrival(시간 차이 도착)을 다루고 있습니다. 기존의 시스템은 일반적으로 GCC-PHAT(Generalized Cross-Correlation Phase Transform)와 같은 전통적인 기법에 의존하고 있습니다. 본 연구에서는 학습 기반 방법이 합성 데이터(Synthetic Data)를 기반으로 하더라도 현실 세계의 새로운 데이터에서 GCC-PHAT을 크게 초월할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 충분히 크고 다양한 시뮬레이션 데이터셋을 기반으로 모델을 훈련하여 실제 문제의 관련 특성을 포착합니다. 훈련된 모델인 SONNET(Simulation Optimized Neural Network Estimator of Timeshifts)는 실시간으로 실행 가능하며, 재훈련 없이도 다양한 실제 데이터 애플리케이션에 즉시 적용할 수 있습니다. 이 모델은 자가 보정(self-calibration) 작업에 상당한 성능 향상을 제공하는 것도 보여줍니다.

- **Performance Highlights**: SONNET 모델을 사용하면 전통적인 방법과 비교하여 자가 보정 작업에서 성능이 크게 향상된 것을 확인할 수 있습니다. 이 모델은 현실 세계의 문제에 대해 더욱 효과적으로 작동하여 다양한 로컬라이제이션 응용(e.g., multilateration, direction of arrival)에 기여할 수 있습니다.



### CopyrightMeter: Revisiting Copyright Protection in Text-to-image Models (https://arxiv.org/abs/2411.13144)
- **What's New**: 최근 텍스트-이미지 변환 모델이 높은 품질의 이미지를 생성하는 데 강력한 도구로 떠오르면서, 저작권 문제에 대한 우려도 커지고 있습니다. 이러한 문제를 해결하기 위해 다양한 저작권 보호 기법이 제안되었으나, 이들의 효과성과 강인성이 충분히 검토되지 않았습니다. 이를 보완하기 위해, 기존 보호 기법을 시스템화하고 저작권 보호 평가를 위한 통합 프레임워크인 CopyrightMeter를 개발하였습니다.

- **Technical Details**: 이 연구에서는 CopyrightMeter라는 평가 프레임워크를 통해 17개의 최신 보호 기법과 16개의 대표적인 공격 방법을 통합하여 평가를 수행합니다. 평가 기준으로는 fidility(충실도), efficacy(효과성), resilience(강인성)를 사용하여 각 보호 기법의 성능을 다차원적으로 분석합니다. 이를 통해 기존의 보호 기법들이 공격에 대한 저항력이 낮고, 공격의 종류에 따라 최적의 보호 방법이 다름을 밝혀냈습니다.

- **Performance Highlights**: 연구 결과에 따르면, 보호 기법의 16개 중 17개가 공격에 대해 강인하지 않으며, 가장 우수한 보호 방법은 목표에 따라 달라진다는 것을 발견했습니다. 또한, 더 발전된 공격이 보다 강력한 보호 기법의 발전을 촉진한다는 인사이트도 얻었습니다. 이러한 Insights는 향후 저작권 보호 연구와 T2I DM 개발에 있어서 중요한 방향성을 제공할 것입니다.



### Demonstrating the Suitability of Neuromorphic, Event-Based, Dynamic Vision Sensors for In Process Monitoring of Metallic Additive Manufacturing and Welding (https://arxiv.org/abs/2411.13108)
Comments:
          This work is a derivative work of a conference proceedings paper submitted to the International Modal Analysis Conference 2024, and is subject to some copyright restrictions associated with the Society of Experimental Mechanics. A variation of this paper is also published in the Weapons Engineering Symposium and Journal (WESJ) which is not publically accessible

- **What's New**: 이 논문에서는 금속 적층 제조 및 용접 공정 모니터링을 위한 고동적 범위(High Dynamic Range), 고속(High-Speed)의 신경형 이벤트 기반(dynamic vision sensors) 센서의 적합성을 입증했습니다. 특히, 금속 적층 제조에서의 품질 관리를 위한 인 과정 모니터링이 중요한 관심사로 떠오르고 있습니다. 기존 기술들이 극한의 조명 환경과 높은 속도의 금속 용융 풀에서 측정하기 어려운 반면, 이벤트 기반 센서는 이러한 환경을 이겨낼 수 있는 잠재력을 보입니다.

- **Technical Details**: 이벤트 기반 센싱(Event-Based Sensing)은 측정량이 임계값을 초과할 때만 데이터를 전송하거나 기록하는 측정 패러다임입니다. 이로 인해 이벤트 기반 센서는 전력 소비와 메모리 및 대역폭을 줄일 수 있으며, 다양한 시간 척도와 동적 범위에서 작동할 수 있습니다. 예를 들어, 이벤트 기반 이미저는 약 120 dB의 매우 높은 동적 범위를 가지고 있어 기존의 8비트 이미저가 약 48 dB의 동적 범위를 갖는 것과 대조적입니다.

- **Performance Highlights**: 연구 결과, 이벤트 기반 이미저는 텅스텐 불활성 가스(TIG) 및 레이저 용접의 용융 풀을 관찰할 수 있음이 입증되었습니다. 추가 엔지니어링 노력을 통한 발전으로 신경형 이벤트 이미저는 용융 풀의 3D 형상 측정 및 이상 탐지, 분류, 예측 작업이 가능할 것으로 기대됩니다. 이러한 성능은 금속 적층 제조 및 용접과 같은 고강도 조명 소스에서의 공정 모니터링을 위한 최적의 후보가 됩니다.



### Improving OOD Generalization of Pre-trained Encoders via Aligned Embedding-Space Ensembles (https://arxiv.org/abs/2411.13073)
Comments:
          Accepted at the Self-Supervised Learning Workshop and the Unifying Representations in Neural Models Workshop at NeurIPS 2024

- **What's New**: 이 논문은 self-supervised pre-trained encoders를 사용하여 out-of-distribution (OOD) 데이터에 대한 제로샷 일반화를 개선할 목적으로 구성된 Ensemble-InfoNCE라는 새로운 방법을 소개합니다. 이 방법은 임베딩 공간(embedding space)에서 앙상블을 구성하여 예측 공간(predictive space)에서의 해석력과 가중치 공간(weight space)의 유연성을 모두 제공합니다. 기존의 방법들은 라벨 없는 데이터로 효과적인 앙상블을 구성하는 문제에 접근하지 못했습니다. 따라서, 이 연구는 이론적 분석을 통해 개별 임베딩 공간 사이의 관계를 규명하고, 비지도적 방식으로 이들을 정렬하는 방법론을 제시합니다.

- **Technical Details**: 제안하는 Ensemble-InfoNCE 방법은 널리 사용되는 InfoNCE의 대조 손실을 사용하여 미리 훈련된 인코더들의 하이퍼스피어 임베딩 공간에서 평균을 집계하는 과정으로 진행됩니다. 각 인코더는 semantically 유사한 샘플을 유사한 방향으로 가리키게 임베딩 공간을 정렬해야 합니다. 이를 위해, 기존 연구 결과를 기반으로 한 비지도적 접근을 제안하며, 앙상블 인코더들의 정렬된 임베딩 공간을 통해 올바른 임베딩을 복원할 수 있음을 이론적으로 입증합니다.

- **Performance Highlights**: MNIST 데이터셋을 통해 실험 결과, 제안된 Ensemble-InfoNCE 방법이 단일 인코더에 비해 ID(인디스키피션)와 OOD 데이터 모두에서 미리 훈련된 임베딩 품질을 개선한다는 것을 보여줍니다. 이 방법은 잘 정렬된 임베딩 공간을 활용함으로써 OOD 데이터에 대한 일반화 성능을 효과적으로 향상시킵니다. 이러한 성능 개선은 self-supervised 비지도 학습에서 앙상블 방법론의 중요성을 강조합니다.



### Bounding-box Watermarking: Defense against Model Extraction Attacks on Object Detectors (https://arxiv.org/abs/2411.13047)
- **What's New**: 이 논문은 개체 탐지(object detection, OD) 모델을 대상으로 하는 새로운 백도어 기반 워터마킹(backdoor-based watermarking) 접근법인 바운딩 박스 워터마킹(bounding-box watermarking, BBW)을 제안합니다. BBW는 API를 통해 추출된 모델에 백도어를 삽입함으로써 모델 소유권을 증명하며, 특히 이상적인 공격 시나리오에서 효과적으로 기능합니다. 기존의 접근법과 달리, BBW는 개체의 바운딩 박스를 교묘히 수정하여 OD 기능을 유지하면서도 모델이 추출되었음을 인식할 수 있도록 합니다.

- **Technical Details**: 기술적으로, BBW는 요청된 이미지에 대해 객체의 바운딩 박스를 의도적으로 조작하여 백도어를 생성합니다. 이 조작 과정은 특정 트리거에 해당하는 객체의 바운딩 박스만을 확장하거나 변형함으로써 발생합니다. 이를 통해 침입자가 추출한 모델의 백도어를 확인할 수 있는 독특한 동작을 제시하게 됩니다. 실험에서는 이 방법이 100% 정확도로 추출된 모델을 식별할 수 있음을 보여주었습니다.

- **Performance Highlights**: 성능 측면에서, BBW는 세 가지 개체 탐지 데이터셋을 사용한 실험에 있어 100%의 정확도를 달성하였습니다. 실험의 한 예로, BBW는 API 응답의 2%에 해당하는 객체의 바운딩 박스를 5% 확대하는 방식으로 전체 검증을 수행했습니다. 이러한 결과는 BBW가 신뢰성 있고 효과적인 모델 추출 방어 메커니즘임을 시사합니다.



### LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Learned Image Compression (https://arxiv.org/abs/2411.13033)
Comments:
          IEEE VCIP 2024 poster

- **What's New**: 이 논문에서는 고급 생성 모델에 의해 지원되는 저비트율 학습 이미지 압축 (LIC) 모델이 지각적 메트릭을 활용하여 실현 가능해졌음을 보입니다. 특히, 이미지 캡션을 하위 정보로 사용하여 높은 압축률과 우수한 지각 품질을 달성하는 모델들이 소개됩니다.

- **Technical Details**: 저자들은 대형 다중 모달 모델 (LMM)을 사용하여 캡션을 생성하고 이를 단일 모델 내에서 압축할 수 있음을 보여줍니다. 또한, 모든 LIC 네트워크에 적용 가능한 새로운 의미-지각 (semantic-perceptual) 지향의 파인튜닝 (fine-tuning) 방법을 제안합니다. 이를 통해 기존 방법에 비해 LPIPS BD-rate에서 41.58% 개선을 달성했습니다.

- **Performance Highlights**: 이러한 성능 향상은 저비트율 환경에서도 뛰어난 품질의 이미지 압축이 가능함을 보여줍니다. 추가적으로, 구현 및 사전 훈련된 가중치는 제공된 URL에서 확인할 수 있습니다.



### Training Physics-Driven Deep Learning Reconstruction without Raw Data Access for Equitable Fast MRI (https://arxiv.org/abs/2411.13022)
- **What's New**: 이 논문에서는 Compressibility-inspired Unsupervised Learning via Parallel Imaging Fidelity (CUPID)라는 새로운 방법을 제안하여, 고속 자기공명영상(MRI) 재건을 위한 PD-DL 훈련을 가능하게 합니다. CUPID는 전문 MRI 센터 외에서 사용할 수 없는 원시 k-공간 데이터에 대한 접근성을 필요로 하지 않으며, 임상에서 생성된 재구성 이미지만을 사용합니다.

- **Technical Details**: CUPID는 DICOM 형식으로 획득된 임상 이미지에서 훈련되며, Noise 및 Aliasing 아티팩트가 포함될 수 있는 저어 샘플링된 데이터를 활용합니다. 이 방법은 Parallel Imaging 알고리즘과의 일관성을 보장하고, 출력 이미지의 Compressibility를 평가함으로써 unsupervised 조건에서 훈련을 수행합니다.

- **Performance Highlights**: CUPID는 기존의 감시(supervised) 및 자기감시(self-supervised) PD-DL 훈련 전략과 비교할 때 동등한 결과를 달성하며, 전통적인 압축 감지(compressed sensing, CS) 기법 및 최신 생성 모델보다 뛰어난 성능을 보여줍니다. 이 방법은 농촌 및 소외 지역에서도 고속 MRI 접근성을 높이는 데 기여할 수 있습니다.



### Automating Sonologists USG Commands with AI and Voice Interfac (https://arxiv.org/abs/2411.13006)
- **What's New**: 이번 연구는 AI 기반의 첨단 초음파 이미징 시스템을 소개하며, 실시간 이미지 처리(real-time image processing), 장기 추적(organ tracking), 음성 명령(voice commands)을 통해 임상에서의 진단 효율성과 정확성을 향상시키기 위한 시스템입니다. 기존의 초음파 진단은 시간 소모가 크고, 사용자 상호작용에 따라 주관적인 결과가 발생하는 문제가 있었습니다.

- **Technical Details**: 이 시스템은 컴퓨터 비전(computer vision)과 딥러닝(deep learning) 알고리즘을 활용하며, 특히 Detectron2의 Mask R-CNN 모델을 사용해 장기 및 주요 랜드마크의 의미론적 분할(semantic segmentation)을 수행합니다. 자동화된 이미지 프로세싱을 통해 최소한의 인적 개입으로도 고급 정보를 추출할 수 있어 진단 정확성이 향상됩니다. 또한, 음성 인식 기능을 포함하여 사용자가 환자를 관찰하는 동안 손대지 않고 시스템을 조작할 수 있도록 합니다.

- **Performance Highlights**: 간 섬유증 감지를 위해 최적화된 간 조직병리 모듈은 98.6%의 인상적인 정확도를 달성하였습니다. 더불어 장기 분할 모듈은 50%에서 95% 사이의 출력 신뢰 수준(confidence levels)을 제공하여 장기 탐지의 효과성을 보여줍니다. 이러한 성과는 AI 기술이 임상 진단 과정에서 어떤 긍정적인 영향을 미칠 수 있는지를 잘 나타냅니다.



### Enhancing Deep Learning-Driven Multi-Coil MRI Reconstruction via Self-Supervised Denoising (https://arxiv.org/abs/2411.12919)
- **What's New**: 이 논문에서는 자가 감독된 잡음 제거(self-supervised denoising)를 딥러닝 기반 재구성 방법의 전처리 단계로 통합했을 때의 효과를 연구하였습니다. 고무적인 점은, 저소음 참조 데이터의 필요성을 줄이고도 MRI 재구성의 품질을 향상시킬 수 있다는 것입니다. 연구에서는 가우시안 잡음으로 손상된 k-공간 데이터를 사용하여 진단 정확도를 높일 수 있는 가능성을 보여었습니다.

- **Technical Details**: 다양한 MRI 데이터의 특징을 반영하기 위해 일반화된 스타인 비편향 위험 추정(Generalized Stein's Unbiased Risk Estimate, GSURE)을 기반으로 하는 잡음 제거 파이프라인을 개발하였습니다. 이 과정에서 Diffusion Probabilistic Models (DPMs)와 Model-Based Deep Learning (MoDL)을 활용하여 자가 감독적 잡음 제거가 DL 네트워크의 학습 성과에 미치는 영향을 평가하였습니다. 특히, 자기공명영상(MRI) 재구성을 위한 피험자 데이터를 포함하여, 다양한 신호 대 잡음비(signal-to-noise ratio, SNR) 조건에서 성능을 시험했습니다.

- **Performance Highlights**: 실험 결과, 자가 감독된 잡음 제거를 통해 MRI 재구성의 품질과 효율성이 크게 향상됨을 확인하였습니다. 예를 들어, DPM을 사용하는 경우, 잡음이 제거된 이미지를 이용해 DL 네트워크를 훈련했을 때, 정상화된 평균 제곱근 오차(normalized root mean squared error, NRMSE)와 구조적 유사도 지수(structural similarity index measure, SSIM), 피크 신호 대 잡음비(peak signal-to-noise ratio, PSNR)가 모든 SNR 수준에서 유의미하게 개선되었습니다. 이러한 결과는 다양한 조건에서도 자가 감독된 잡음 제거가 DL 기반 MRI 재구성 방법의 유효성을 높일 수 있음을 뒷받침합니다.



### Signformer is all you need: Towards Edge AI for Sign Languag (https://arxiv.org/abs/2411.12901)
Comments:
          Official Code at: this https URL

- **What's New**: 이번 논문에서는 Signformer라는 새로운 Sign Language Translation (SLT) 모델을 제안합니다. Signformer는 사전 훈련된 모델이나 자원 집약적인 기술에 의존하지 않고, 기존의 gloss-free 방식을 개선하여 Edge AI 환경에서 효율적인 실시간 사용이 가능합니다. 이로써 SLT 분야의 지속 가능성과 실용성을 재정의하고 있습니다.

- **Technical Details**: 이 연구는 기존의 Sign Language Transformer (SL-Transformer) 아키텍처를 기반으로 하여, 새로운 알고리즘 설계를 위해 다양한 수화 언어의 구조적 특성을 철저히 분석합니다. 이를 통해 변형된 컨볼루션, 주의(attention) 메커니즘 및 위치 인코딩을 활용하여 파라미터 수를 극적으로 줄이면서도 성능을 극대화할 수 있었습니다. Signformer 모델은 GLT와 비교해 1807배 더 작은 파라미터 수를 자랑하며, 기존의 여러 SOTA 모델에 비해 뛰어난 성능을 올리고 있습니다.

- **Performance Highlights**: Signformer는 gloss-free SLT 선두주자로 자리매김하면서, 2024년 기준으로 새로운 2위 성적을 기록했습니다. 특히 Signformer-Full 모델은 gloss-free Finest SignLLM보다 더 나은 테스트 세트 성능을 발휘하며, 0.57백만의 파라미터 수를 통해 TOP5에 등재되었습니다. 이는 SLT 분야에서 파라미터와 성능의 균형을 새롭게 정의하는 중요한 이정표가 됩니다.



### Tree Species Classification using Machine Learning and 3D Tomographic SAR -- a case study in Northern Europ (https://arxiv.org/abs/2411.12897)
- **What's New**: 이번 연구에서는 Synthetic Aperture Radar (SAR) 데이터를 기반으로 한 3D 단층 이미지를 활용하여 나무종 분류의 정확성을 향상시키기 위한 새로운 접근 방식을 소개합니다. TomoSense 데이터셋을 사용하여 여러 탭형 머신러닝 모델을 평가하고 Bayesian 최적화를 통해 성능을 극대화했습니다. 특히, LiDAR(point cloud data)의 데이터를 활용하여 모델의 예측과 관련된 나무 높이 통계를 제시함으로써 단층 데이터의 신뢰성을 평가했습니다.

- **Technical Details**: 연구에서는 다양한 폴라리메트릭 설정과 지리적 분할을 활용하여 SAR 데이터의 분석을 수행했습니다. TomoSense 데이터셋은 2미터의 공간 해상도를 가지며, 개별 나무 높이에 대한 정보와 함께 8종의 나무를 분류하기 위한 기반 데이터를 제공합니다. AutoGluon을 사용하여 머신러닝 모델의 튜닝 및 최적화 과정을 간소화하고, 여러 모델(예: Gradient Boosting Machines, CNNs, Random Forests)을 평가하여 성능을 극대화하는 데 집중했습니다.

- **Performance Highlights**: 모델 최적화 과정을 통해 다양한 탭형 머신러닝 모델이 높은 정확도를 기록했으며, 이는 TomoSense의 3D 단층 이미지를 성공적으로 활용했음을 보여줍니다. 연구 결과, SAR 데이터와 LiDAR 정보를 결합하여 나무 종 분류의 신뢰성을 크게 향상시켰고, 이는 지속 가능한 산림 관리 및 생물 다양성 평가에 기여할 수 있는 잠재력을 가지고 있습니다. 또한, 새로운 데이터 처리 파이프라인은 원거리 탐사에서의 공간 자율 상관관계 문제를 해결하기 위한 두 가지 데이터 분할 전략을 도입함으로써 유의미한 결과를 도출했습니다.



### Residual Vision Transformer (ResViT) Based Self-Supervised Learning Model for Brain Tumor Classification (https://arxiv.org/abs/2411.12874)
- **What's New**: 이번 논문은 MRI 기반 뇌종양 진단을 위한 생성적 자기지도 학습(self-supervised learning, SSL) 모델을 제안합니다. 두 단계로 구성된 이 모델은 첫 번째 단계에서 Residual Vision Transformer (ResViT)를 활용한 MRI 합성(pretext task)과 두 번째 단계에서 ResViT 기반 분류기(classifier) 모델로서의 미세 조정(fine-tuning)을 포함합니다. 이 접근법은 CNN과 ViT의 하이브리드 아키텍처를 사용하여 MRI 이미지를 통한 로컬 및 글로벌 특징을 효과적으로 활용합니다.

- **Technical Details**: 제안된 SSL 모델은 잦은 데이터 부족 문제를 해결하는데 중점을 두며, CNN과 ViT의 특징을 통합하여 MRI 이미지 합성 및 뇌종양 분류 작업을 수행합니다. 또한, 생성된 합성 MRI 이미지를 데이터 증강(data augmentation) 방법으로 활용하여 학습 세트를 균형 있게 유지합니다. 연구팀은 BraTs 2023, Figshare 및 Kaggle 데이터셋을 이용하여 모델 성능을 평가하고, 기존의 다양한 딥러닝 모델들과 비교 연구를 수행하였습니다.

- **Performance Highlights**: 제안된 SSL 모델은 BraTs 데이터셋에서는 90.56%, Figshare에서는 98.53%, Kaggle에서는 98.47%라는 높은 정확도로 뇌종양을 분류하는 성과를 보여주었습니다. 본 연구는 기존의 ImageNet 데이터셋에서 사전 훈련된 모델보다 MRI 데이터셋에서 훈련된 모델의 성능이 우수함을 나타내고, 임상적 적용이 가능한 신뢰할 수 있는 솔루션을 제공하고 있습니다. SSL, 미세 조정, 데이터 증강을 통한 모델의 효과적인 구성은 뇌 MRI 분석에서의 데이터 불충분 문제를 해결하는 데 중요한 기여를 합니다.



### Towards Fairness in AI for Melanoma Detection: Systemic Review and Recommendations (https://arxiv.org/abs/2411.12846)
Comments:
          22 pages, 4 figures, 7 tables,accepted for publication in Future of Information and Communication Conference (FICC) 2025, whose proceedings will be published in 'Lecture Notes in Networks and Systems' by Springer Nature

- **What's New**: 이 연구는 인공지능(AI)을 활용한 흑색종(melanoma) 조기 진단의 최신 동향을 분석하고 다양한 피부톤에서의 효과성을 평가합니다. 주목할 만한 점은 기존의 많은 데이터셋이 주로 밝은 피부를 기반으로 하고 있으며, 이는 결과적으로 다양한 환자에게 불공평한 치료 결과를 초래할 수 있다는 것입니다. 연구자들은 L'Oreal Color Chart Map과 같은 새로운 피부 색상 척도의 도입을 제안하여 보다 포괄적인 피부톤 평가 방법을 개발할 필요성을 강조합니다.

- **Technical Details**: 연구는 2013년부터 2024년 사이에 발표된 피부 암 탐지를 위한 AI 연구들에 대한 체계적인 검토와 초기 분석을 수행하였습니다. 특히, 딥러닝(deep learning) 방법론, 데이터셋, 그리고 피부톤 표현 등이 주된 초점이었습니다. 현재 사용되고 있는 Fitzpatrick Skin Tone Scale(FST)이 다소 제한적이며, 다양한 피부톤을 효과적으로 반영하지 않는 분석의 정량적인 결과들이 도출되었습니다.

- **Performance Highlights**: 연구 결과는 피부 색깔에 따라 초기 진단의 정확도가 다르며, 특히 어두운 피부 톤의 환자에게는 정확도가 떨어지는 경향이 있음을 보여줍니다. 제안된 새로운 방법론인 피부 색조(bias free skin hue) 통합은 그러한 격차를 해소할 수 있는 실마리가 될 수 있습니다. 연구팀은 AI 모델이 모든 피부톤에서 효과적으로 작동할 수 있도록 하는 데이터셋 구성의 중요성과 이를 위한 권장 사항을 제시하였습니다.



### Efficient Medicinal Image Transmission and Resolution Enhancement via GAN (https://arxiv.org/abs/2411.12833)
- **What's New**: 본 논문은 X-ray 이미지를 위한 새로운 접근법을 제시합니다. Real-ESRGAN의 최적화된 네트워크 전송을 통해 이미지 품질을 향상시키고, 서버 부담 및 전송 대역폭을 줄이는 방법을 소개합니다. 특히 B/W X-ray 이미지의 노이즈와 해상도 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 X-ray 이미지를 낮은 해상도로 전처리한 후, 최종 수신자가 Real-ESRGAN을 사용하여 이미지를 업스케일링합니다. Residual-in-Residual Dense Blocks (RRDB)와 지각(Perceptual) 및 적대적 손실(Adversarial Loss) 함수를 통합하여 고품질의 저노이즈 이미지를 얻습니다. 이 과정은 전송 품질에 관계없이 주요 진단 정보를 보존하는 것을 목표로 합니다.

- **Performance Highlights**: 비교 평가는 제안된 방법이 기존의 CNN 기반 및 ESRGAN 모델보다 우수한 노이즈 감소 및 세부 정보 선명도를 제공함을 보여주었습니다. Peak Signal-to-Noise Ratio (PSNR) 및 Structural Similarity Index (SSIM)와 같은 계량적 지표와 정성적 평가 모두 이러한 이점을 확인했으며, Real-ESRGAN의 활용 가능성을 보여주고 있습니다.



### FedCL-Ensemble Learning: A Framework of Federated Continual Learning with Ensemble Transfer Learning Enhanced for Alzheimer's MRI Classifications while Preserving Privacy (https://arxiv.org/abs/2411.12756)
Comments:
          6 pages, 4 figures

- **What's New**: 이 연구는 알츠하이머 질병(Alzheimer's disease)의 분류(classification)를 위한 새로운 접근법을 제안합니다. 고급 딥 러닝 기술(deep learning techniques)을 활용하고, 안전한 데이터 처리 방법을 결합하여 향상된 성능을 보이는 모델을 개발했습니다. 특히, 전이 학습 모델(transfer learning models)을 사용하여 의료 이미지 데이터에서 고급 특징을 추출하는 데 중점을 두었습니다.

- **Technical Details**: 주요 기술로는 ResNet, ImageNet, VNet과 같은 전이 학습 모델을 사용하여 알츠하이머 관련 미세한 패턴을 감지할 수 있습니다. 이 모델은 데이터 소스의 다양성에 대해 강력한 특징을 추출할 수 있도록 조정되었습니다. 또한, 페더레이션 학습(federated learning) 접근법을 통합하여 데이터 개인 정보 보호를 보장하면서 분산된 모델의 이점을 최대한 활용합니다.

- **Performance Highlights**: 실험 결과는 알츠하이머 분류의 정확성을 향상시키는 데 기여하며, 안전하고 협력적인 건강 관리 데이터 분석을 위한 프레임워크도 제공합니다. 추가적으로, 데이터 전송 시 기밀성과 무결성을 보장하기 위해 암호 기반 encryption 메커니즘도 적용되었습니다. 이러한 결과는 예측 성능을 개선하고 환자 데이터를 공유하지 않으면서도 모델의 강력한 학습을 가능하게 합니다.



### SAM-I2I: Unleash the Power of Segment Anything Model for Medical Image Translation (https://arxiv.org/abs/2411.12755)
- **What's New**: 본 논문에서는 의료 영상 번역을 위한 새로운 프레임워크인 SAM-I2I를 제안합니다. SAM-I2I는 Segment Anything Model 2 (SAM2)를 기반으로 하여 멀티스케일 맥락 정보를 효과적으로 캡처할 수 있는 프리트레인(pre-trained) 이미지 인코더를 활용합니다. 이를 통해 기존 방법들의 한계를 넘어서며, 더 높은 정확도의 의료 이미지 전환을 달성합니다.

- **Technical Details**: SAM-I2I 구조는 SAM2에서 파생된 프리트레인 이미지 인코더와 마스크 유닛 어텐션 모듈 기반의 이미지 디코더로 구성됩니다. Hiera라는 계층적 비전 변환기를 사용하여 멀티스케일 기능을 생성하고, 디코더는 이러한 기능을 융합하여 고품질의 대상 이미지 모달리티(modality)를 생성합니다. 효과적인 학습을 위해 디코더의 가중치만 업데이트하며, 인코더의 가중치는 고정되어 모델의 표현 능력을 보존합니다.

- **Performance Highlights**: 실험 결과 SAM-I2I는 IXI 데이터셋에서 T1, T2, PD 간의 이미지 변환 작업에서 기존의 최첨단 방법들을 초월하는 성능을 보여 주었습니다. 특히, 각 MRI 스캔에서 뇌 조직을 포함한 축 방향 슬라이스를 추출하여 모델이 다양한 임상 시나리오에서 최적의 성능을 발휘하도록 하였습니다. 궁극적으로 SAM-I2I는 더 효율적이고 정확한 의료 이미지 전환을 이룰 수 있는 잠재력을 보여줍니다.



New uploads on arXiv(cs.AI)

### BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games (https://arxiv.org/abs/2411.13543)
Comments:
          Preprint, under review

- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)와 VLMs(비전 언어 모델)의 에이전틱(Agentic) 능력을 평가하기 위해 신규 벤치마크인 BALROG를 소개합니다. BALROG는 다양한 난이도의 게임 환경을 통합하여, 복잡하고 동적인 환경에서의 성능을 검증하는 데 초점을 맞추었습니다. 연구자들은 이 벤치마크를 통해 트렌드와 한계를 평가하고자 하며, 특히 장기적인 결정과 공간 추론의 중요성을 강조합니다.

- **Technical Details**: BALROG는 다양한 권장 학습 환경을 포함하며, 게임 과제를 통해 LLM과 VLM의 성능을 평가하는 프레임워크를 제공합니다. BabyAI, Crafter, NetHack 등의 게임 환경을 통합하여, 에이전트의 장기 계획, 공간 추론, 환경 역학 이해 등 다양한 능력을 세밀하게 평가할 수 있도록 설계되었습니다. 이 벤치마크는 절차적으로 생성된 환경을 사용하여 쉽게 암기할 수 없도록 하였으며, 다수의 인기 있는 LLM과 VLM 모델을 대상으로 성능을 분석합니다.

- **Performance Highlights**: BALROG를 통해 여러 LLM과 VLM의 성능을 평가한 결과, 기존 모델들은 상대적으로 쉬운 게임에서 부분적인 성공을 보였지만, 더 복잡한 작업에서는 심각한 결점을 드러냈습니다. 특히 VLM 모형은 시각적 정보가 제공될 때 성능이 떨어지며, 이는 비전 기반 의사결정의 신뢰성이 현저히 떨어짐을 나타냅니다. 연구 결과는 LLM이 현실 세계의 복잡한 작업을 해결하기 위해서는 장기 전망과 효율적인 결정이 필요함을 강조합니다.



### AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations (https://arxiv.org/abs/2411.13451)
Comments:
          18 pages, 3 figures, an abridged version to appear in NeurIPS 2024 AFM Workshop

- **What's New**: 본 논문에서는 웹 기반 작업을 자동화하는 멀티모달 웹 에이전트를 위한 새로운 AdaptAgent 프레임워크를 제안합니다. 이 프레임워크는 불특정 웹사이트와 도메인에 대해 단 1~2개의 인간 시연을 통해 적응할 수 있도록 설계되었습니다. 실험 결과, 이 접근 방식이 현재 최첨단 모델에 비해 약 3.36%에서 7.21%의 작업 성공률 향상을 가져온다는 것을 보여주었습니다.

- **Technical Details**: AdaptAgent 프레임워크는 소수의 시연을 활용한 학습을 통해 기존의 대규모 사전 학습 및 미세 조정 전략을 보완하는 방법론을 제시합니다. 이 접근 방식은 두 가지 주요 메커니즘인 In-Context Learning (ICL)과 Meta-Learning을 통해 멀티모달 LLM(Multimodal Large Language Model) 기반 모델들이 새로운 웹 환경에 적응할 수 있게 합니다. 연구 결과, 멀티모달 시연이 텍스트 기반 시연보다 더 효과적임을 증명하였으며, 메타 학습에서의 데이터 선택 방법이 에이전트의 일반화 능력에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 두 가지 벤치마크인 Mind2Web과 VisualWebArena에서 수행된 실험을 통해, AdaptAgent 방식이 단 1-2개의 인간 시연을 사용하여 에이전트의 작업 성공률을 크게 향상시킬 수 있음을 입증했습니다. 전반적으로, 이 연구는 대규모 사전 학습과 미세 조정에 의존하지 않고도, 웹 에이전트의 일반화를 개선하는 보완적 방법론을 제시하며, 여러 웹 도메인에서의 작업 수행 가능성을 높이고 있습니다.



### Explainable Finite-Memory Policies for Partially Observable Markov Decision Processes (https://arxiv.org/abs/2411.13365)
Comments:
          Preprint -- Under Review

- **What's New**: 이번 연구에서는 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Processes, POMDPs)에서 설명 가능성을 높이는 새로운 방법론을 제시합니다. 이 접근법은 메일리 머신(Mealy machine)과 의사결정 트리(decision trees)를 결합하여 선언된 정책의 효율적이고 설명 가능성이 높은 표현을 제공합니다. 특히, 이 논문에서는 유한 상태 컨트롤러(finite-state controllers, FSC)를 위한 새로운 데이터 구조를 제안하여, 확장성과 단순화를 이끌어냅니다.

- **Technical Details**: 연구에서 제안하는 FSC는 메일리 머신의 장점을 이용하여 각 상태가 고정 정책(stationary policy)에 대응하도록 구현됩니다. 각 상태 간 전환은 이제 의사결정 트리에 의해 설명되어, 설명 가능성을 강화합니다. 또한, '어트랙터 기반' 정책의 특정 속성을 활용하여 더 간단하고 작은 표현을 구축할 수 있는 방법을 설명합니다.

- **Performance Highlights**: 이 방법론은 또한 정량적 성능을 높일 수 있는 가능성을 보여줍니다. 여러 사례 연구(case studies)를 통해 제안된 방법의 설명 가능성을 실증적으로 입증하며, 주요 결과는 더 나은 성능을 제공합니다. 이로 인해 최적 정책(optimal policies)의 누락 문제를 해결하고 다양한 유한 메모리 정책에 대해 일반화하는 능력을 강화합니다.



### BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices (https://arxiv.org/abs/2411.12990)
Comments:
          Accepted as a Spotlight Poster to NeurIPS 2024

- **What's New**: 이 논문은 AI 벤치마크의 품질을 평가하기 위한 새로운 프레임워크를 개발하였으며, 46개의 기준을 바탕으로 24개의 AI 벤치마크를 평가하였습니다. 야기된 결과는 품질의 큰 차이를 드러내며, 널리 사용되는 벤치마크들이 중요한 문제에 직면해 있음을 보여줍니다. 통계적 유의성을 보고하지 않거나 결과의 반복 사용이 어려운 벤치마크가 대다수를 차지하여, 이는 모델 평가의 신뢰성에 관한 우려를 나타냅니다.

- **Technical Details**: 이 연구는 AI 벤치마크의 생애 주기 전반에 걸쳐 46개의 최선의 관행을 고려한 평가 프레임워크를 제공합니다. 또한, 기초 모델(Foundation Model, FM)와 비기초 모델(non-Foundation Model) 각각에 대해 16개와 8개의 벤치마크를 평가하여 품질 차이를 발견했습니다. 이 평가 결과를 바탕으로 최소 품질 보증 체크리스트를 제공하며, 벤치마크 품질 및 사용 가능성을 분석할 수 있는 살아있는 리포지토리도 개발했습니다.

- **Performance Highlights**: 연구의 결과, 현재 AI 벤치마크의 품질에 대한 표준적인 구조적 평가가 부족하다는 점이 지적되었습니다. 사용된 벤치마크들은 실제 응용에 적합하고 실질적인 유용성을 제공하지 못하는 경우가 많았습니다. 이를 통해 더 나은 벤치마크를 개발할 수 있도록 도움을 줄 수 있는 체크리스트와 가이드라인이 제공되어, 개발자들이 최선의 관행에 맞춰 벤치마크를 개선할 수 있는 방향을 제시합니다.



### MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Collaborative Learning (https://arxiv.org/abs/2411.12977)
- **What's New**: 최근의 복합체(embodied) 에이전트들은 오픈된 환경에서 스스로 학습하는 데 뛰어난 능력을 보여주고 있습니다. 하지만 대규모 언어 모델(LLM)과 결합될 경우, 이들 에이전트는 종종 기본적인 작업에서도 어려움을 겪습니다. 이를 해결하기 위해 제안된 \

- **Technical Details**: \collabvoyager는 인간의 문화적 학습 이론에 영향을 받아 개발된 새로운 프레임워크입니다. 이 프레임워크는 세 가지 주요 혁신으로 구성되어 있습니다: (1) 관점 이해 이론(theory of mind)을 통한 인지 연결, (2) 에이전트 간 자연어(Natural Language) 통신, (3) 작업 및 환경 지식의 의미 기억(semantic memory) 및 협력 경험의 에피소딕 기억(episodic memory). 이러한 기능들은 에이전트가 자신과 타인의 정신 상태를 이해하고 추론하도록 돕습니다.

- **Performance Highlights**: 실험 결과, \



### Real-Time Energy-Optimal Path Planning for Electric Vehicles (https://arxiv.org/abs/2411.12964)
Comments:
          12 pages, 7 figures, 5 tables

- **What's New**: 전기차(EV)의 통합이 가속화되는 가운데, 에너지 친화적 경로 설정의 필요성이 대두되고 있습니다. 이 연구에서는 EV의 동적 특성을 반영한 정확한 에너지 모델을 개발하여 경로 계획의 실현 가능성을 높이고, 배터리 제약 조건 하에서도 현실적인 경로 찾기가 가능하도록 합니다. 또한, 재생 제동으로 인한 부정적인 에너지 비용을 처리하는 새로운 온라인 재가중함수를 도입하여 실시간 애플리케이션에 적합한 성능을 제공합니다.

- **Technical Details**: 연구에서는 차량의 하중, 도로 경사, 주행 패턴 등을 고려한 데이터 기반 에너지 모델을 개발하였습니다. 이 모델은 각 도로 구간에 필요한 에너지를 정확히 추정하며, 기존의 단순화된 에너지 모델들이 무시하는 차량 동역학을 반영하여 경로 계획의 품질을 향상시킵니다. 또한, 두 가지 새로운 재가중 함수로 에너지 최적 경로 찾기를 빠르게 수행할 수 있습니다.

- **Performance Highlights**: 실제 교통 네트워크에 대한 광범위한 실험을 통해, 제안된 접근 방식이 전기차의 에너지 최적 경로 설정에서 연산 효율성과 에너지 추정 정확도를 크게 향상시킨다는 것을 입증하였습니다. 특히, 차량 동역학을 고려한 에너지 모델과 새로운 경로 찾기 알고리즘의 결합이 경로의 실현 가능성을 높이고, 전반적인 계산 복잡성을 줄이는 데 기여하며, 다양한 환경에서 효율적인 경로 설정이 가능합니다.



### KAAE: Numerical Reasoning for Knowledge Graphs via Knowledge-aware Attributes Learning (https://arxiv.org/abs/2411.12950)
- **What's New**: 이 논문에서는 Knowledge-Aware Attributes Embedding (KAAE) 모델을 제안하여 기존의 지식 그래프 임베딩에서 수치 추론 (numerical reasoning)과 관련된 두 가지 주요 문제를 해결하고자 합니다. 첫째로, semantic relevance 문제를 해결하기 위해 Mixture-of-Experts-Knowledge-Aware (MoEKA) 인코더를 도입하여 엔티티, 관계, 수치 속서의 상호작용을 통합한 공동 의미 공간을 구현하였습니다. 둘째로, ordinal knowledge contrastive learning (OKCL) 전략을 통해 임의의 데이터에서 고품질의 오르도 샘플을 생성하여, 세밀한 의미의 뉘앙스를 포착할 수 있도록 하였습니다.

- **Technical Details**: KAAE 모델은 두 가지 주요 구성 요소인 Mixture-of-Experts-Knowledge-Aware 인코더와 ordinal knowledge contrastive learning 전략으로 이루어져 있습니다. MoEKA 인코더는 다양한 관계적 맥락에서 엔티티 간의 의미를 식별하고 통합하는 전문가 네트워크를 구성하여, 각 속성의 중요성을 분별합니다. 또한, OKCL 방법론을 통해 대부분의 수치적 속성에서 발생할 수 있는 의미적 모호성을 해소하고, 코사인 유사성을 바탕으로 고품질의 오르도 샘플을 생성하여 날카로운 수치 추론 능력을 향상시킵니다.

- **Performance Highlights**: KAAE 모델은 세 가지 공개 기준 데이터셋을 통해 광범위한 실험을 진행하였으며, 기존 모델들보다 우수한 성능을 입증하였습니다. 이 모델은 다양한 속성 값 분포에서 뛰어난 성능을 기록하며, 각각의 데이터셋에서 최첨단 결과를 달성했습니다. 실험 결과는 KAAE의 강인함, 합리성, 그리고 해석 가능성을 강조합니다.



### The Game-Theoretic Symbiosis of Trust and AI in Networked Systems (https://arxiv.org/abs/2411.12859)
- **What's New**: 이번 논문은 네트워크 시스템에서 인공지능(AI)과 신뢰(trust) 간의 상호 관계를 심층적으로 탐구하고 있습니다. AI는 동적이고 복잡한 네트워크에서 실시간으로 데이터를 처리하고 학습하는 능력 덕분에 신뢰를 관리하는 데 상당한 지원을 제공합니다. 그러나 AI 시스템의 신뢰성 역시 성공적인 통합에 필수적인 요소로 작용합니다. 이를 통해 AI가 네트워크 보안을 향상시키고, AI 시스템에 대한 신뢰가 증가함으로써 긍정적인 피드백 루프를 형성하는 기반을 확보합니다.

- **Technical Details**: 본 논문은 게임 이론(game-theoretic framework)을 활용하여 AI의 사이버 보안 내에서의 전략적 역할과 신뢰 평가 방법을 제시합니다. 신뢰는 네트워크 정책, 정체성(identity), 시스템 성능 등 여러 차원에서 영향을 미치며, 각 차원에서 신뢰를 구축하는 기존 접근 방식들이 내부 공격을 충분히 다루지 못하고 있습니다. 특히, 시스템 성능에 대한 신뢰는 점점 더 정교해지는 공격으로부터 위협받고 있으며, AI가 신뢰를 동적으로 관리하면서 강화된 보안 생태계를 만드는 방법을 제안합니다.

- **Performance Highlights**: 네트워크 내부에서 신뢰를 조정하는 전략적 접근이 강조됩니다. '제로 트러스트(Zero Trust)' 아키텍처를 통해 모든 사용자와 장치는 지속적으로 검증되고, 신뢰는 지속적인 검사에 의해 조정되어야 합니다. 이는 내부 및 외부의 위협을 방지하는 데 효과적이며, 사이버 기만(cyber deception)과 잘못된 정보 캠페인 등 다양한 분야에서 응용될 수 있습니다. 이러한 전략적 신뢰 모델은 보안 팀이 공격자의 신뢰 인식을 조작하기 위한 시스템 설계에도 직결되며, 정보의 출처를 검증하고 올바른 신뢰를 회복하는 방안으로 활용됩니다.



### Declare and Justify: Explicit assumptions in AI evaluations are necessary for effective regulation (https://arxiv.org/abs/2411.12820)
- **What's New**: 이 논문은 AI 시스템의 안전성을 보장하기 위한 평가의 중요성을 강조하며, 개발자들이 평가에 대한 기본적인 가정을 명시하고 정당화해야 한다고 주장합니다. 이러한 가정에는 위협 모델링(threat modeling), 프록시 작업(validity of proxy tasks), 그리고 적절한 능력 도출(capability elicitation)이 포함됩니다. 현재 대다수의 가정들이 충분히 정당화되지 않고 있으며, 불안전한 평가 결과에 따라 AI 개발이 중단될 필요가 있다고 제시합니다. 결과적으로 이 연구는 AI 개발의 투명성을 높이고, 진보된 AI 시스템의 보다 효과적인 거버넌스를 위한 실질적인 경로를 제안합니다.

- **Technical Details**: AI 평가(AI evaluations)는 AI 개발자 안전 계획의 큰 구성 요소로, 현재 모델이 해로운 능력을 가질 가능성을 평가하고 미래 모델에 대한 예측을 수립합니다. 현재 모델과 미래 모델을 구분하여 위험 벡터(threat vectors)를 평가하고, 이를 통해 프록시 작업을 설계 및 수행하는 과정이 포함됩니다. 그러나 현재의 안전성 평가 방식은 평가 후 모델을 배포할 수 있는 안전성을 확실하게 보장하지 못하며, 사전에 제정된 전제 조건들이 종종 재평가 되어야 할 필요가 있습니다.

- **Performance Highlights**: AI 시스템의 평가가 실패할 경우, 즉 모델이 프록시 작업을 잘 수행하지 않는 경우, 이는 해당 모델이 위험한 기능을 가지고 있지 않을 것이라는 잘못된 믿음을 초래할 수 있습니다. 평가자들은 능력을 도출하는 과정에서 사용 가능한 모든 후처리 방법(post-training enhancements)을 활용해야 하며, 이는 성능의 오탐을 방지할 수 있는 중요한 단계로 기능합니다. 그러나 작동 방식에 대한 사전 이해가 부족할 경우, 새로운 위험 요소를 도출할 수 있는 능력이 제한됩니다.



### Conversational Medical AI: Ready for Practic (https://arxiv.org/abs/2411.12808)
Comments:
          14 pages, 7 figures, 3 tables

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 기반의 대화형 AI 에이전트인 'Mo'를 실제 의료 환경에서 처음으로 대규모 평가한 결과를 발표하였습니다. Mo는 기존 의료 상담 채팅 서비스에 통합되어, 세 주간 랜덤화된 통제 실험을 통해 926건의 사례를 평가하였습니다. 이 연구는 대화형 AI의 안전하고 효과적인 적용 가능성을 보여주는 중요한 발판이 됩니다.

- **Technical Details**: Mo는 여러 개의 서브 에이전트로 구성된 LLM 기반 시스템으로, 각 특정 작업에 최적화된 모델을 활용하여 높은 정확성과 성능을 구현합니다. 이 시스템은 의료 지식, 추론, 커뮤니케이션 스타일 등 핵심 기능을 바탕으로 모델을 선택했습니다. 또한, Mo는 Microsoft Azure와 Google Cloud Platform(GCP)을 통해 운영되며, 유럽 개인정보 보호 규정을 준수합니다.

- **Performance Highlights**: 환자들은 AI 지원 대화에서 제공된 정보의 명확성 및 전반적인 만족도가 더 높았으며(각각 3.73 대 3.62 및 4.58 대 4.42), 신뢰도와 공감도는 유사한 수준으로 나타났습니다. 대화의 안전성 측면에서도 95%의 대화가 '좋음' 또는 '우수함'으로 평가되었고, 위험한 상황은 전혀 없었습니다. 이를 통해 AI 의료 보조 도구가 환자 경험을 향상시키면서도 안전 기준을 유지할 수 있다는 가능성을 명확히 하였습니다.



### SpecTool: A Benchmark for Characterizing Errors in Tool-Use LLMs (https://arxiv.org/abs/2411.13547)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 툴 사용 작업에서 발생하는 오류 패턴을 식별하고 특성화하기 위한 새로운 벤치마크인 SpecTool을 소개합니다. 기존의 벤치마크들이 LLM의 성공률만 계산하는 데 반해, SpecTool은 에러 패턴 분석을 통해 문제의 근본 원인을 이해할 수 있는 기회를 제공합니다. 이를 통해 연구자들은 더 나은 오류 완화 전략을 수립할 수 있습니다.

- **Technical Details**: SpecTool은 도구, 영화, 여행, 스포츠 등 10개 환경 카테고리를 포함하여 30개 이상의 다양한 작업을 평가할 수 있도록 설계되었습니다. 특히, LLM이 툴 사용 작업에서 나타내는 7가지 공통 오류 패턴을 적절히 분석할 수 있는 프레임워크를 구축하였습니다. 150개의 인간 주석된 쿼리를 활용하여 LLM의 출력에서 오류 패턴을 식별하고, 이를 통해 다양한 LLM의 성능을 면밀히 분석할 수 있습니다.

- **Performance Highlights**: SpecTool은 여러 주요 LLM을 대상으로 한 사례 연구를 통해 오류 패턴을 효과적으로 식별할 수 있는 능력을 보여줍니다. 이 연구는 LLM의 행동에서 발생하는 오류를 체계적으로 분석함으로써, 이러한 오류들이 LLM의 전체 성능에 미치는 영향을 이해하는 데 기여하였습니다. 따라서 SpecTool은 LLM의 툴 사용 성능을 향상시키기 위한 강력한 도구로 자리매김할 것으로 기대됩니다.



### Metacognition for Unknown Situations and Environments (MUSE) (https://arxiv.org/abs/2411.13537)
- **What's New**: 이 연구는 자율 에이전트가 새로운 환경에서의 적응력을 향상시키기 위해 메타인지(metacognition)를 통합하는 새로운 프레임워크인 MUSE(Metacognition for Unknown Situations and Environments)를 제안합니다. 메타인지는 자신의 인지 과정에 대한 인식과 조절을 포함하며, 이는 AI 에이전트가 낯선 도전에 대처하는 데 필요한 인지적 유연성을 제공합니다. MUSE 프레임워크는 두 가지 초기 구현(세계 모델링 및 대형 언어 모델(LLMs) 기반)을 소개하여 메타인지 사이클의 구현을 보여줍니다.

- **Technical Details**: MUSE 프레임워크는 자기 인식(self-awareness) 및 자기 조절(self-regulation) 메커니즘을 통합하여 에이전트가 주어진 작업을 성공적으로 수행할 확률을 예측하고 iteratively 전략을 선택할 수 있도록 설계되었습니다. 이 시스템은 과거 경험에 기반하여 내부 모델을 지속적으로 학습하여 동작 계획의 성공 가능성을 평가합니다. MUSE 에이전트는 Meta-World 및 ALFWorld 환경에서의 실험을 통해 기존 강화 학습 접근법보다 새로운 시나리오 처리에서 상당한 개선을 보여주었습니다.

- **Performance Highlights**: MUSE 프레임워크는 에이전트의 적응성을 높이기 위해 주요 평가 지표로 역량(competence)을 우선시합니다. 이 접근은 에이전트가 낯선 상황에 만나고도 고립되지 않도록 하여 효과적인 탐색을 촉진합니다. 결과적으로 MUSE 에이전트는 과제 성공 가능성을 극대화하는 행동 계획을 식별하고 온라인 학습 및 적응에서 더 안전하고 효과적으로 작동할 수 있게 됩니다.



### Identity Preserving 3D Head Stylization with Multiview Score Distillation (https://arxiv.org/abs/2411.13536)
Comments:
this https URL

- **What's New**: 본 논문에서는 PanoHead 모델을 활용하여 360도 시점에서 이미지를 합성하여 3D 헤드 스타일화 (3D head stylization) 문제를 다루고 있습니다. 기존의 3D 스타일화 방법들은 주로 근전방 (near-frontal) 뷰에서 합성되며, 원본 이미지의 고유한 정체성 (identity) 유지에 어려움이 있는 반면, 이번 연구는 이러한 한계를 극복하고자 합니다. 제안된 프레임워크는 negative log-likelihood distillation (LD)을 통해 정체성 보존과 스타일화 품질을 향상시킵니다.

- **Technical Details**: 연구에서 제안하는 방법은 PanoHead의 사전 훈련된 매개 변수를 미세 조정하여 다양한 도메인에서 이미지를 생성합니다. LD 기법을 3D 인식 이미지 생성기 (3D-aware image generators)에 적용하며, SDS와의 차이점을 설명하고, 교차 포즈 의존성 (cross-pose dependencies) 및 그리드 노이즈 제거 (grid denoising)를 추가하여 스타일화 품질을 향상시키는 방법을 제시합니다. 성능을 높이기 위해 점수 텐서에 대한 순위 감소 (rank reduction)를 사용하며, 이는 스타일 기법에도 긍정적인 영향을 미칩니다.

- **Performance Highlights**: 제안한 방법은 관련 있는 헤드 스타일화 방법들에 비해 질적 및 양적 차원이 상당한 개선을 나타냅니다. LD를 사용하여 더욱 선명하고 ID 보존이 우수한 결과를 얻으며, 다각적 그리드 및 미러 점수 기울기를 통합하여 스타일화 품질을 더욱 개선합니다. 이 연구는 3D 헤드 스타일화의 발전뿐 아니라, GAN을 통한 효과적인 증류 (distillation) 과정에 대한 중요한 통찰을 제공합니다.



### Entropy Bootstrapping for Weakly Supervised Nuclei Detection (https://arxiv.org/abs/2411.13528)
Comments:
          Submitted for CVPR 2025

- **What's New**: 본 논문은 Histopathology 영역에서 세포(instance) 분할을 위한 새로운 약한 감독 학습(weakly supervised learning) 접근 방식을 제안합니다. 이 방법은 단일 포인트 레이블(label)만을 사용하여 세포 픽셀의 기본 분포를 추정하며, 이로부터 완전한 세포 마스크를 유추해 Mask-RCNN(Instance Segmentation Model)을 활용하여 결과를 도출합니다. 연구 결과, 95%의 픽셀 레이블 축소에도 불구하고, 제안한 방법이 충분히 좋은 성능을 발휘한다고 보고하였습니다.

- **Technical Details**: 연구에 사용된 데이터는 PanNuke 데이터세트로, 이는 다양한 조직 유형에 대한 암 세포 핵(instance) 분할 및 분류 데이터셋입니다. 논문에서는 Bayesian Segmentation Network를 사용하여 점 레이블 지정보다 더 정확한 세분화(segmentation) 결과를 생성하기 위한 엔트로피 부트스트랩(entropy bootstrap) 단계를 수행합니다. 이 네트워크는 각 픽셀이 특정 클래스에 속할 확률을 추정하며, 제안된 방법은 약한 레이블을 사용하여 세포 검출(task)을 수행하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 접근법은 임상 이미징 시나리오를 더 잘 반영하고, 기존의 세포 분할 데이터셋보다 복잡한 임상 환경을 나타냅니다. 결과적으로, 제안된 방법은 높은 효율성을 통해 최소한의 레이블로도 좋은 성능을 보여줍니다. 특히, 약한 감독 방식으로 얻은 초기 세분화 결과가 Ground Truth와 매우 잘 일치하며, 기존의 고급 방법들과 비교할 때 간단한 구조임에도 불구하고 우수한 검출 성과를 거두었습니다.



### Advancing Complex Medical Communication in Arabic with Sporo AraSum: Surpassing Existing Large Language Models (https://arxiv.org/abs/2411.13518)
Comments:
          arXiv admin note: text overlap with arXiv:2411.06713

- **What's New**: 이 연구는 다국어 기능에 대한 의료 분야의 증가하는 수요를 반영하여 아랍어에 맞춤화된 Sporo AraSum 모델을 평가합니다. 기존의 JAIS 모델과 비교하여, Sporo AraSum은 의료 문서화에서 아랍어의 복잡한 특성을 처리하는 데 적합한 성능을 보입니다. 이 연구는 특히 임상 문서화를 위한 AI 모델의 필요성을 강조하고 있습니다.

- **Technical Details**: 연구는 합성 데이터셋(synthetic datasets)과 의료 관련 지표인 PDQI-9를 수정하여 Sporo AraSum과 JAIS의 성능을 평가합니다. 모델의 성능은 환자와 의사의 상호작용을 요약하는 데 중점을 두며, 정확성(accuracy), 포괄성(comprehensiveness), 임상 유용성(clinical utility), 언어-문화적 역량(linguistic-cultural competence) 등의 요소가 포함됩니다. Sporo AraSum의 아키텍처는 아랍어의 미세한 언어적 뉘앙스를 고려하여 정확하고 문화적으로 민감한 문서화를 가능하게 합니다.

- **Performance Highlights**: Sporo AraSum은 AI 중심의 정량적 지표와 모든 정성적 측면에서 JAIS 모델을 크게 능가하는 성과를 보였습니다. 특히, Sporo AraSum은 AI 망상(hallucination)과 같은 리스크를 줄이며 아랍어가 필요한 의료 환경에서 더 적합한 모델로 나타났습니다. 향후 연구는 실제 데이터를 포함하여 이러한 결과를 검증하고 의료 시스템에의 통합 가능성을 탐색할 것을 제안합니다.



### Utilizing Large Language Models to Synthesize Product Desirability Datasets (https://arxiv.org/abs/2411.13485)
Comments:
          9 pages, 2 figures, 6 tables

- **What's New**: 이번 연구는 Product Desirability Toolkit (PDT) 테스트를 위한 합성 데이터셋 생성을 대규모 언어 모델(LLMs)로 수행하는 방법을 탐구합니다. 특히 비용 효율적인 gpt-4o-mini 모델을 사용하여 1000개의 제품 리뷰를 생성하는 세 가지 방법(Word+Review, Review+Word, Supply-Word)을 제안합니다. 본 연구는 합성 데이터셋이 실제 데이터가 제한된 상황에서 비용 효과적이고 확장 가능한 옵션이 될 수 있는지를 평가합니다.

- **Technical Details**: 합성 데이터셋을 생성할 때, 높은 감정 일치를 보여주었고 Pearson 상관계수는 0.93에서 0.97에 이릅니다. Supply-Word 방법은 가장 높은 텍스트 다양성과 PDT 용어의 범위를 보였지만, 데이터 생성 비용이 증가했습니다. 이는 LLM이 생성한 합성 데이터가 특히 적은 테스트 데이터가 필요할 때의 이점을 제공함을 보여줍니다.

- **Performance Highlights**: 연구 결과는 개발된 세 가지 합성 데이터 생성 방법이 높은 감정 일치를 달성하며, 다양한 텍스트 형태를 제공한다고 보고합니다. 그러나 모든 방법에서 약간의 긍정적 편향이 관찰되었으며, 이는 미래 연구에서 해결해야 할 필요성을 제기합니다. 합성 데이터의 생산 비용 분석 또한 성과 강조 점으로, 이점은 확장 가능성 및 비용 절감 효과가 있음이 확인되었습니다.



### PatentEdits: Framing Patent Novelty as Textual Entailmen (https://arxiv.org/abs/2411.13477)
- **What's New**: 본 연구에서는 창작권 확보를 위해 필요한 특허 수정 예측 문제를 새로운 학습 가능한 작업으로 설정하고, 이를 위한 PatentEdits 데이터셋을 소개합니다. 이 데이터셋은 105,000개의 수정 예시를 포함하고 있으며, 기존 문서 수정 예측 연구와 달리 주목할 만한 이전 문헌과의 관계를 고려합니다. 특히, 대형 언어 모델(LLM)을 활용하여 인용된 참조와 초안 문장 간의 텍스트 포함 관계를 평가하는 방식을 제안합니다.

- **Technical Details**: PatentEdits 데이터셋은 2007년부터 2014년까지의 미국 공인 발명 특허 105,000건으로 구성되어 있습니다. 데이터셋에 포함된 초안, 인용된 참조 및 최종 특허 텍스트를 정렬하여 어떤 문장이 Kept, Edited 또는 Deleted 되는지를 자동적으로 라벨링하는 알고리즘을 설계했습니다. 향상된 클레임 예측을 위해 LLM을 활용하여 인용된 문장을 긍정 예로, 초안 문장을 앵커로, 최종 문장을 부정 예로 설정하고, triplet loss로 파인튜닝합니다.

- **Performance Highlights**: 실험 결과, 인용된 참조를 포함시키고 초안 특허 클레임과 인용된 특허 클레임 간의 포함 관계에 집중할 경우, 특허 클레임의 새로움을 평가하는 데 효과적임을 보여주었습니다. 특히, BLEU-4 및 METEOR와 같은 다양한 유사도 평가 방법을 활용하여 초안 및 최종 문장 간의 문서 매칭 품질을 검증하였습니다. 이러한 접근 방식은 특허 출원자와 미국 특허 심사원이 필요한 수정을 예측하는 데 실질적인 기여를 할 수 있을 것으로 기대됩니다.



### SoK: A Systems Perspective on Compound AI Threats and Countermeasures (https://arxiv.org/abs/2411.13459)
Comments:
          13 pages, 4 figures, 2 tables

- **What's New**: 본 논문에서는 다양한 대형 언어 모델(LLMs)과 관련된 시스템 공격을 체계적으로 분류하는 최초의 시도로, 복합 AI 시스템에서의 소프트웨어 및 하드웨어 취약점과 방어 방법을 탐구합니다. 현재 대부분의 연구가 개별 구성 요소에 초점을 맞추고 있는 반면, 본 논문에서는 계층 간 상호 작용이 이루어지는 복합 AI 시스템의 복잡성에 대한 강조를 하고 있습니다. 향후 공격 경로와 방어 메커니즘을 개발하기 위해 기존 문헌을 정리하고 새로운 위협 모델을 제시합니다.

- **Technical Details**: 본 연구는 AI 및 ML 시스템의 소프트웨어 및 하드웨어 계층에서 발생할 수 있는 여러 공격을 분석합니다. 특히, 시스템 공격과 알고리즘 공격을 명확히 구분하고, 다양한 공격 벡터와 그에 대한 방어를 체계적으로 분류합니다. 또한, 공격 및 방어 시스템을 체계화하기 위해 Mitre Att&ck 프레임워크를 활용하여 각 공격을 위협 모델에 맞춰 재정렬합니다.

- **Performance Highlights**: 복합 AI 시스템 내에서의 시스템 공격을 활용하여 엔드 투 엔드 공격을 수행할 수 있는 방법을 논의하고, 이를 통해 데이터 누수 및 보안 보장 훼손과 같은 문제를 해결하기 위한 방안을 모색합니다. 다양한 사례 연구를 통해 현재 보안 관행의 격차를 지적하고, 더 안전한 AI 애플리케이션을 구축하기 위한 열린 연구 질문을 제시합니다.



### LIMBA: An Open-Source Framework for the Preservation and Valorization of Low-Resource Languages using Generative Models (https://arxiv.org/abs/2411.13453)
- **What's New**: 이번 백서에서는 소수 언어의 디지털 자원 부족 문제를 해결하기 위한 언어 모델 생성을 위한 프레임워크를 제안하고 있습니다. 특히, 고급 언어 모델과 대조적으로 데이터가 부족한 언어의 보존을 지원하기 위해 데이터 생성에 중점을 두고 있습니다. 사르디니아어와 같은 멸종 위기 언어를 사례 연구로 사용하여 프레임워크의 효과성을 입증하려고 합니다. 이 연구는 언어 다양성을 촉진하고 언어 표준화 및 재생을 지원하는 데 기여하고자 합니다.

- **Technical Details**: 소수 언어는 디지털 자원 확보가 어려워 고급 언어 처리 기술(advanced linguistic technologies)의 개발이 제한됩니다. 본 연구에서는 언어 모델을 구축하기 위해 새로운 언어 도구를 생성하기 위한 방법론을 제시하는데, 사르디니아어를 활용하여 효과성을 검증하고 있습니다. 또한, 데이터 수집을 촉진하기 위해 구글과 모질라와 같은 다양한 기관이 프로그램을 시작했음을 언급하고 있습니다.

- **Performance Highlights**: 프레임워크를 통해 소수 언어를 위한 언어 모델이 생성될 가능성이 높아집니다. 데이터가 부족한 언어에서도 새로운 데이터 생성을 지원하여 AI 모델의 효과적인 훈련을 도울 수 있을 것으로 기대됩니다. 이러한 접근법은 사회적 포용성을 높이고, 향후 소수 언어의 보존과 활성화에 긍정적인 영향을 미칠 것으로 예상됩니다.



### Robust Monocular Visual Odometry using Curriculum Learning (https://arxiv.org/abs/2411.13438)
Comments:
          8 pages

- **What's New**: 이번 연구는 커리큘럼 학습(Curriculum Learning) 방법론을 통해 단일 카메라 비주얼 오도메트리(Monocular Visual Odometry) 추정 문제를 해결하고자 합니다. 기존의 최첨단(SOTA) 기법을 개선하기 위해 다양한 커리큘럼 학습 전략을 조사합니다.

- **Technical Details**: 연구에서는 DPVO(Deep-Patch-Visual Odometry) 프레임워크를 강화하기 위하여 샘플 난이도를 평가하는 방법, 자기주도 가중 손실 메커니즘을 통한 적응형 스케줄링, 강화 학습(Reinforcement Learning)을 통한 훈련 강조 조정을 포함한 여러 고유한 커리큘럼 학습 전략을 개발하였습니다.

- **Performance Highlights**: TartanAir 데이터셋을 활용한 평가에서, 제안된 커리큘럼 학습 기반 DPVO(CL-DPVO)는 기존의 SOTA 방법들보다 뛰어난 성능을 보여주었으며, 이는 비주얼 오도메트리 시스템에 커리큘럼 학습 원칙을 통합한 효과를 입증합니다.



### SynEHRgy: Synthesizing Mixed-Type Structured Electronic Health Records using Decoder-Only Transformers (https://arxiv.org/abs/2411.13428)
- **What's New**: 본 논문에서는 전자 건강 기록(Electronic Health Records, EHR)의 합성 데이터를 생성하기 위한 새로운 토큰화 전략을 제안합니다. 이 전략은 covariates, ICD 코드, 불규칙하게 샘플링된 시간 시계열 데이터와 같은 다양한 데이터 유형을 포함한 구조화된 EHR 데이터에 최적화되어 있습니다. 저자들은 GPT와 유사한 디코더 전용 변환 모델을 사용하여 고품질의 합성 EHR을 생성하는 방법을 입증합니다.

- **Technical Details**: 제안된 방법론에서는 구조화된 EHR 데이터 생성에 있어서의 토큰화 전략을 혁신적으로 재구성하고, 여러 환자의 방문 데이터를 아우르는 복합형 구조를 다룹니다. 이 연구는 MIMIC-III 데이터셋을 활용하여 생성된 데이터의 품질(quality)을 주목하며, 특히 높은 누락률(missingness)과 불규칙한 시간 포인트로 인한 도전 과제를 해결하는 데 중점을 두고 있습니다. 또한, 비슷한 최첨단 모델들과 비교하여 데이터를 다룰 때의 유용성(utility)과 개인 정보 보호(privateness) 측면에서의 평가 결과도 제시됩니다.

- **Performance Highlights**: SynEHRgy라는 프레임워크는 여러 방문 데이터에 걸쳐 구조화된 EHR 데이터를 성공적으로 생성할 수 있는 방법을 제시하며, 특히 불규칙하게 샘플링된 시간 시계열 데이터 생성에서 뛰어난 성능을 발휘합니다. 제안된 접근 방식은 최신 모델들과 비교하여 생성된 데이터의 신뢰성(fidelity), 유용성(utility), 개인 정보 보호(privateness) 측면에서 우수한 결과를 가져왔습니다. 이 연구는 EHR 데이터 생성의 새로운 가능성을 제시하며, 의료 데이터의 수집과 활용에 있어 장기적 영향을 미칠 것으로 기대됩니다.



### Heuristically Adaptive Diffusion-Model Evolutionary Strategy (https://arxiv.org/abs/2411.13420)
- **What's New**: 본 연구는 Diffusion Models와 Evolutionary Algorithms 간의 근본적인 연결고리를 밝히고, 이 두 접근 방식이 고품질 샘플을 생성하는 공통의 생성 메커니즘을 공유하고 있음을 강조합니다. 두 방법 모두 무작위 초기 분포에서 반복적으로 개선된 샘플을 생성하여 최적화 과정에서의 유연성과 정밀성을 높입니다. 또한, Evolutionary Algorithms에 Diffusion Models의 메모리 기능을 통합하여 진화적 최적화에서 더 깊이 있는 히스토리 정보를 활용할 수 있음을 보여줍니다.

- **Technical Details**: Diffusion Models는 Gaussian noise를 사용하여 도메인 특정 정보를 저하시키고, 학습 가능한 모델을 통해 이를 복원하는 이중 단계 과정을 사용합니다. 이러한 모델이 Evolutionary Algorithms에 통합되면서 조건부 샘플링을 위한 classifier-free guidance를 제공하여, 개체군 전반의 특성을 세밀하게 조절할 수 있게 됩니다. 연구는 딥러닝 기반의 Diffusion Models를 다양한 진화적 작업에 적용하여, 최적의 파라미터에 대한 효율적인 수렴을 이룰 수 있음을 입증합니다.

- **Performance Highlights**: 제안된 접근 방식은 효율적인 수렴을 통해 높은 피트니스 파라미터를 유지하면서 탐색적 다양성을 보장합니다. 진화적 알고리즘이 새로운 적합 솔루션을 샘플링하는 방식을 개선하는 한편, 이 모델은 과거 설명 정보를 활용하여 보다 정교한 샘플을 생성하는 데 기여합니다. 또한, 본 연구는 기존의 얕은 휴리스틱에서 깊은 메모리를 가진 프레임워크로 진화적 알고리즘을 향상시키며, 진화적 탐색 다이나믹스를 보다 정밀히 조정하는 방안을 제시합니다.



### Unification of Balti and trans-border sister dialects in the essence of LLMs and AI Technology (https://arxiv.org/abs/2411.13409)
Comments:
          Accepted by IEEE conference ISCSLP 2024

- **What's New**: 이번 논문은 Balti 언어의 중요성과 그 방언들의 통합 노력을 다룹니다. 이 언어는 중국어-티베트어족(Sino-Tibetan) 소속으로, 인도, 중국, 파키스탄 등 여러 국가에서 다양한 방언이 존재합니다. 인공지능(AI) 기술의 발전에 따라, 방언 통합을 통한 공통성 이해의 필요성이 강조됩니다.

- **Technical Details**: Balti 언어의 분석 및 문서화는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 수행됩니다. 이 연구에서는 다양한 방언 간의 기본적인 어근(root), 어휘(lexica), 음운론적 관점에서의 공통점을 찾아내는 방법을 제시합니다. 이러한 접근 방식은 언어 보존에 기여할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: AI를 활용한 Balti 언어의 분류 및 표준화 노력은 여러 방언들의 문서화와 분석을 통해 이루어집니다. 글로벌화가 진행되는 현재, AI 기술은 언어 다양성 이해 및 방언 통합에 중요한 역할을 할 수 있습니다. 이 연구는 인공지능 기술이 어떻게 언어 보존에 기여할 수 있는지를 탐구합니다.



### Fact-Level Confidence Calibration and Self-Correction (https://arxiv.org/abs/2411.13343)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 LLMs의 자신감(calibration) 문제를 해결하기 위해 사실 수준(fact-level) 보정(framework) 방법을 제안하고 있습니다. 기존의 방법은 전체 응답에 대한 두 개의 스칼라 값을 추정하는 방식으로, 긴 형식의 생성에서는 여러 개의 사실이 포함된 경우 적절하지 않음을 지적합니다. 연구진은 각 사실의 정확도와 관련성(relevance)을 고려하여 보정(calibration) 과정을 보다 세분화하여 진행합니다.

- **Technical Details**: 새로운 사실 수준 보정 프레임워크에서는 각 사실의 정확성과 관련성을 기반으로 자신감을 평가합니다. 이 프레임워크는 모델이 사실 개별적으로 부분적인 자신감과 정확성을 보일 수 있도록 합니다. 또한, Confidence-Guided Fact-level Self-Correction ($\textbf{ConFix}$) 방법이 개발되어 높은 자신감을 가진 사실을 사용하여 낮은 자신감을 가진 사실의 질을 향상시키는 접근 방식을 제시합니다.

- **Performance Highlights**: ConFix 방법은 네 가지 데이터셋과 여섯 가지 모델에 걸쳐 실험된 결과, 허구(hallucination)의 발생을 현저히 줄임으로써 모델의 신뢰성을 증대시킴을 보여주었습니다. 이 방법은 외부 지식 소스를 요구하지 않고도 자체적으로 허구를 완화할 수 있는 가능성을 제시합니다. 전체적으로, 이 연구는 LLMs의 사실 기반 자신감 보정과 사용 가능한 자가 수정 방법을 제시하여, 모델의 견고한 응용을 가능하게 합니다.



### Verifying Machine Unlearning with Explainable AI (https://arxiv.org/abs/2411.13332)
Comments:
          ICPRW2024

- **What's New**: 본 연구에서는 Explainable AI (XAI)를 활용하여 Machine Unlearning (MU)의 효과성을 검증하는 방법을 탐구합니다. 특히, 데이터 삭제를 위한 전통적인 ML 모델 재훈련 방법의 비효율성을 해소하기 위해 MU를 적용하여 특정 학습 패턴을 선택적으로 잊도록 하였습니다. 본 논문은 XAI를 통해 MU의 성능을 평가하는 새로운 지표인 Heatmap Coverage (HC)와 Attention Shift (AS)를 제안하며, 이는 MU 검증의 혁신적인 단계를 제공합니다.

- **Technical Details**: XAI는 ML 모델의 결정 과정에 대한 통찰력을 제공하여, 민감한 정보를 성공적으로 삭제했는지, 새로운 문제를 일으키지는 않았는지 평가합니다. 본 연구에서는 데이터 재레이블링, 모델 섭동 등 다양한 MU 기법을 탐색하며 각 기법의 도전 과제를 논의합니다. 이 과정에서 атриbuty 기반 XAI 기술을 활용하여 지역적 특성의 중요성 변화를 관찰하고 정량화합니다.

- **Performance Highlights**: 본 연구의 주요 기여는 MU 기법의 효과를 검증하기 위해 атриbuty 기반 XAI 메소드를 적용한 것입니다. 이를 통해 모델이 특정 데이터 삭제 후 주목하는 위치를 시각화하고 정성적으로 평가할 수 있습니다. 또한, MU 프로세스를 평가하기 위해 HC와 AS라는 새로운 XAI 기반 지표를 도입하여, 기존의 정확도 지표를 넘어서는 포괄적인 평가를 제공합니다.



### An Evolutional Neural Network Framework for Classification of Microarray Data (https://arxiv.org/abs/2411.13326)
- **What's New**: 이 연구는 암 유전자 신호를 식별하기 위한 DNA 마이크로어레이 데이터 분석에서 발생하는 고차원성 문제를 해결하기 위해 유전자 알고리즘(Genetic Algorithm, GA)과 다층 퍼셉트론 신경망(Multi-Layer Perceptron Neural Network, MLP)의 하이브리드 모델을 제안합니다. 이 모델의 목표는 유용한 유전자의 하위 집합 선택 과정에서 발생하는 문제를 극복하는 것입니다. 이를 통해 암 진단과 예후의 정확도를 높일 수 있습니다.

- **Technical Details**: 고차원 데이터로 인한 분류의 정확도 저하 문제를 다루기 위해, 연구는 GA를 사용해 특성 선택(feature selection) 과정에서 차원 축소를 수행합니다. 이후 선택된 유전자들은 MLP를 통해 분류됩니다. 이러한 방법론은 데이터에서 중요한 정보를 효과적으로 추출하고, 분류 정확도를 개선하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안된 방법은 다른 머신러닝 알고리즘에 비해 높은 정확도를 달성했으며, 선택된 유전자의 수를 최소화하는 결과를 보여주었습니다. 실험 결과는 유전자 선택과 분류 효율성의 향상이 있음을 시사합니다. 이를 통해 암 진단 기술의 발전에 기여할 수 있는 가능성을 보여주고 있습니다.



### Are Large Language Models Memorizing Bug Benchmarks? (https://arxiv.org/abs/2411.13323)
Comments:
          pre-print

- **What's New**: 이 논문은 여러 소프트웨어 엔지니어링 작업에 필수적인 대형 언어 모델(LLMs)의 성능을 평가하기 위한 버그 벤치마크의 신뢰성에 대한 우려를 제기합니다. 연구진은 데이터 누수(data leakage)가 LLM의 성능 평가에 미치는 영향을 체계적으로 분석하여, 커뮤니티에서 널리 사용되는 벤치마크의 선택에 대한 신중함을 강조합니다.

- **Technical Details**: 연구진은 실제 버그 데이터셋을 바탕으로 LLM의 메모리화(memorization) 경향을 검토하기 위해 여러 가지 지표를 사용했습니다. 이러한 지표에는 Negative Log-Likelihood(NLL)와 n-gram accuracy가 포함되며, 이를 통해 모델이 특정 코드 조각과 얼마나 잘 친숙한지를 평가합니다. 벤치마크 데이터에서 메모리화 증거가 두드러진 코드 생성 모델(codegen-multi)과 더 적은 메모리화 징후를 보이는 현대 모델(LLaMa 3.1)을 비교했습니다.

- **Performance Highlights**: 연구 결과, codegen-multi 모델은 Defects4J와 같은 벤치마크 데이터에서 높은 n-gram accuracy와 낮은 NLL을 보였습니다. 반면, LLaMa 3.1과 같은 새로운 모델들은 더 넓은 데이터셋에서 학습하여 메모리화의 징후가 제한적이었습니다. 이러한 결과는 벤치마크 선택의 중요성과 LLM 평가 시 신뢰할 수 있는 메트릭을 사용해야 할 필요성을 강조합니다.



### Scaling Laws for Online Advertisement Retrieva (https://arxiv.org/abs/2411.13322)
Comments:
          10 pages, 8 figures

- **What's New**: 본 연구에서는 온라인 광고 검색 시스템에서의 스케일링 법칙(scaling law)을 식별하기 위한 경량화 혁신(paradigm)을 제안합니다. 이러한 접근 방식은 최소한의 실험 비용으로 온라인 수익(online revenue)과 기계 비용(machine cost)의 관계를 이해하는 데 중점을 둡니다. 특히, R/R*라는 새로운 오프라인 메트릭(offline metric)을 도입하여 검색 모델의 온라인 수익과 높은 선형 상관관계를 보이고 있습니다. 이는 광고 시스템 최적화의 가능성을 보여줍니다.

- **Technical Details**: 연구진은 R/R* 메트릭과 FLOPs(부동 소수점 연산 수)를 사용하여 오프라인 실험을 수행하고 MLP 모델의 스케일링 행동을 검증했습니다. 결과적으로, 이 스케일링 행동은 파손된 신경 스케일링 법칙(broken neural scaling law)을 따르는 것으로 나타났습니다. 또한, 기계 비용을 추정하기 위한 간단한 시뮬레이션 알고리즘을 제안하여 기계 비용과 온라인 수익 간의 관계를 설정하였습니다. 이를 통해 실험 비용을 크게 줄일 수 있었습니다.

- **Performance Highlights**: 이 연구의 주요 기여는 R/R* 메트릭이 온라인 광고 시스템의 수익을 예측하는 데 있어 뛰어난 오프라인 대리 메트릭 surrogate metric임을 입증한 것입니다. 이 메트릭을 활용함으로써 ROI 제약에 따른 모델 설계 및 다양한 시나리오에서 자원을 효율적으로 할당하였고, 각각 0.85%와 2.8%의 온라인 수익 개선을 달성했습니다. 이러한 결과는 광고 시스템 최적화에 있어 스케일링 법칙의 잠재력을 뒷받침합니다.



### A Resource Efficient Fusion Network for Object Detection in Bird's-Eye View using Camera and Raw Radar Data (https://arxiv.org/abs/2411.13311)
Comments:
          IEEE Intelligent Transportation Systems Conference (ITSC) 2024

- **What's New**: 이 연구는 자동차 자율주행 시스템에서 카메라와 레이더 데이터를 융합하여 객체 인식을 향상시키는 새로운 아키텍처를 제안합니다. 기존의 레이더 신호 처리를 피하고 원시 Range-Doppler (RD) 스펙트럼을 직접 활용합니다. 특히, 카메라 이미지를 Bird's-Eye View (BEV) 극좌표 도메인으로 변환하고, 카메라 인코더-디코더 아키텍처를 통해 필수 특징을 추출합니다.

- **Technical Details**: 제안된 방법은 카메라 데이터와 레이더 데이터를 각각 독립적으로 처리하여 최적의 융합을 추구합니다. 카메라 이미지는 RA와 유사한 표현으로 변환되며, RD 스펙트럼으로부터 복원된 Range-Azimuth (RA) 특징과 융합됩니다. 이 방법은 다양한 감지 작업에 대한 정밀한 정보 표현을 목표로 하며, 데이터 처리를 위한 새로운 이미지 처리 파이프라인을 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 융합 전략이 기존 방법들과 비교하여 정확도에서 경쟁력을 가지며, 계산 복잡성 측면에서 우수한 성능을 보임을 확인했습니다. RADIal 데이터셋을 활용하여 평가한 결과, 리소스 소모를 최소화하면서도 뛰어난 인식 성능을 달성했습니다.



### DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition (https://arxiv.org/abs/2411.13284)
- **What's New**: 새로운 연구에서는 WiFi 기반 인식의 도메인 간 일반화를 위한 Domain-Adversarial Test-Time Adaptation (DATTA)라는 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 도메인-적대적 학습(DAT), 테스트 시간 적응(TTA) 및 가중치 리셋을 결합하여 이전에 본 적 없는 타겟 도메인에 적응하고 파국적 망각(catastrophic forgetting)을 방지합니다. 또한, DATTA는 속도 최적화를 위해 경량화되고 유연한 아키텍처에 통합됩니다.

- **Technical Details**: DATTA 프레임워크는 WiFi 기반의 인식 시스템에서 도메인 불변 특성을 학습하며, 수신 데이터의 도메인 전이에 적응하는 것을 목표로 합니다. 이 시스템은 WiFlexFormer 기반의 중앙 특징 추출기를 통해 실시간으로 인식할 수 있도록 설계되었습니다. 또한, 실제 측정한 WiFi 채널 상태 정보(CSI)의 특성을 이용한 특수한 보강 모듈이 통합되어 교차 도메인 일반화를 강화합니다.

- **Performance Highlights**: DATTA 방법론은 공개된 데이터셋을 사용한 포괄적인 평가에서 검증되었으며, 인간 활동 인식을 위한 실시간 애플리케이션에 적합함을 입증했습니다. 비디오 기반의 TTA 변형과 비교하여, DATTA는 8.1% 향상된 F1-스코어를 달성했습니다. 이 연구의 PyTorch 구현은 공개되어 있어, 후속 연구와 재현 가능성을 용이하게 합니다.



### VideoAutoArena: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation (https://arxiv.org/abs/2411.13281)
Comments:
          Project Page: this https URL

- **What's New**: 최신 연구에서는 비디오 분석 능력을 갖춘 대형 다중 모달 모델(LMMs)에 대한 관심이 증가하고 있습니다. 기존 평가 방법들은 사용자 요구를 충분히 반영하지 못하고, 비용이 많이 드는 인적 주석이 필요했습니다. 이를 해결하기 위해 제안된 VideoAutoArena는 사용자 시뮬레이션을 기반으로 한 자동 평가 시스템으로, 질문 생성 방식에서 혁신적인 접근을 제공합니다. 또한, VideoAutoBench는 신속하고 효율적인 평가를 지원하여 실제 사용자 질문 스타일에 더 가깝게 평가할 수 있습니다.

- **Technical Details**: VideoAutoArena는 사용자에 의해 생성된 열린 질문을 통해 모델의 비디오 이해 능력을 엄격하게 평가하도록 설계되었습니다. 이 평가 모델은 ELO 평가 시스템을 통해 다양한 LMM 간의 공정하고 지속적인 비교를 가능하게 합니다. 또한, 점진적으로 질문의 복잡성을 증가시키는 결함 기반 진화 전략을 도입하여 모델이 더 도전적인 비디오 분석 시나리오를 처리할 수 있도록 합니다. 이를 통해 기존의 LMM 평가 방식에서 발생했던 여러 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, VideoAutoArena는 최신 LMM들 간의 비교를 효과적으로 수행하며, 모델의 강점과 개선이 필요한 영역에 대한 통찰력을 제공합니다. 비공식 모델들이 SOTA(SOTA: State Of The Art) 모델인 GPT-4o에 비해 여전히 성능 차이를 보이며, 이 차이는 비디오 길이가 늘어날수록 더욱 두드러집니다. VideoAutoBench의 결과는 VideoAutoArena의 평가 결과와 밀접하게 일치하는 것으로 나타나, 두 가지 벤치마크가 서로 보완적인 역할을 하고 있음을 보여줍니다.



### Unlocking the Power of Gradient Guidance for Structure-Based Molecule Optimization (https://arxiv.org/abs/2411.13280)
Comments:
          27 pages, 17 figures

- **What's New**: 본 논문에서는 분자 기반 최적화의 새로운 접근법인 Molecule Joint Optimization (MolJO)를 제안합니다. MolJO는 연속 좌표와 이산 타입을 포함한 분자를 최적화하는 첫 번째 기울기 기반 구조 최적화 프레임워크입니다. 이 방법은 Bayesian 추론을 통해 파생된 연속적이고 미분 가능한 공간을 활용하여 서로 다른 모달리티 간의 공동 안내 신호를 가능하게 합니다. 다른 방법들과 달리, MolJO는 SE(3)-공변성을 유지하여 3D 분자의 최적화를 보다 정확하게 수행합니다.

- **Technical Details**: MolJO는 과거 이력의 슬라이딩 윈도우를 최적화하는 새로운 역 보정 전략을 도입하여 탐색과 활용 간의 원활한 균형을 제공합니다. 기울기 안내는 연속 변수에 맞추어 설계되었지만, MolJO는 이산 데이터의 최적화를 응용합니다. 이 프레임워크는 다양한 약물 디자인 설정에도 적용될 수 있으며, R 그룹 최적화, 스캐폴드 홉핑과 같은 다양한 복합 작업에서도 활용 가능성을 확장합니다.

- **Performance Highlights**: MolJO는 CrossDocked2020 벤치마크에서 가장 높은 Vina Dock 점수인 -9.05, SA 0.78 및 성공률 51.3%를 기록하며, 기울기 기반 대응 방법에 비해 4배 이상의 성공률 향상을 보여줍니다. 또한, MolJO는 3D 기본 모델과 비교하여 "Me-Better" 비율이 2배 증가하여 기존 최적화 방법보다 우수한 성능을 나타냅니다. 이런 성능 향상은 MolJO의 다중 목표 최적화 및 도전적인 약물 설계 과제에 대한 확장 가능성을 더욱 입증합니다.



### Towards Specification-Driven LLM-Based Generation of Embedded Automotive Softwar (https://arxiv.org/abs/2411.13269)
Comments:
          21 pages, 2 figures

- **What's New**: 이번 논문은 LLMs(대형 언어 모델)에 의한 코드 생성과 formal verification(형식 검증)을 결합하여 중요한 임베디드 소프트웨어를 생성하는 방법을 연구하였습니다. 첫 번째 기여로는 LLM과 피드백을 제공하는 다양한 critics(비평가)를 결합한 일반 프레임워크인 spec2code를 소개하였습니다. 두 번째 기여는 Scania의 세 가지 산업 사례 연구에서 empirically(경험적으로) 평가된 minimalistic(미니멀리즘)한 spec2code의 실행 가능성에 대한 연구입니다.

- **Technical Details**: spec2code 프레임워크는 자연어(NL)와 formal ANSI/ISO C Specification Language(ACSL)로 주어진 사양을 사용하여 LLM이 자동차 안전-critical(안전 중시) 임베디드 C 코드를 생성하는 가능성을 조사합니다. 코드의 기능적 정확성을 평가하기 위해 deductive verification(유도 검증) 방식과 함께 도구인 Frama-C를 활용하여 ACSL 사양이 제대로 구현되었는지를 확인합니다. 이 연구는 gpt-3.5와 gpt-4-turbo 두 가지 LLM을 사용하여 실제 생산 소프트웨어 모듈로부터 기능적으로 동등한 코드를 생성하는 과정에 초점을 맞추고 있습니다.

- **Performance Highlights**: 결과에 따르면, 반복적인 backprompting과 fine-tuning 없이도 형식적으로 정확한 코드가 생성될 수 있음을 보여줍니다. 자동차 임베디드 시스템의 안전성과 정확성은 매우 중요하며, 본 연구는 LLMs가 소프트웨어 개발 과정에서 중요한 역할을 할 수 있음을 시사합니다. spec2code 프레임워크는 LLMs를 활용하여 임베디드 소프트웨어를 생성하는 데 있어 새로운 방법론을 제시하며, 이는 안전 중시 소프트웨어 개발에 긍정적인 영향을 미칠 것으로 기대됩니다.



### FASTNav: Fine-tuned Adaptive Small-language-models Trained for Multi-point Robot Navigation (https://arxiv.org/abs/2411.13262)
- **What's New**: 이번 연구는 로봇 내비게이션을 위한 가벼운 언어 모델의 성능을 높이는 FASTNav라는 새로운 방법을 제안합니다. 이는 fine-tuning, teacher-student iteration, 그리고 다중 포인트 로봇 내비게이션을 포함한 세 가지 모듈로 구성되어 있으며, 적은 비용과 높은 정확도로 로봇 상에서의 언어 모델 배포를 가능하게 합니다. FASTNav는 SLMs(small language models)의 활용을 극대화하여 사용자의 개인정보 보호를 보장하면서도 뛰어난 성능을 발휘합니다.

- **Technical Details**: 제안된 FASTNav 접근 방식에서는 입력 데이터를 도메인 특화 데이터셋으로 미세 조정(fine-tuning)하여 SLMs의 성능을 향상시킵니다. 이 과정에서 LoRA(저작수준 적응)를 이용해 작은 모델의 추론 속도와 정확성을 높이는 동시에 extensive한 프롬프트 엔지니어링 없이도 효과적으로 작동할 수 있도록 합니다. 또한, teacher-student iteration 모듈을 적용하여 모델이 내비게이션 작업에서 지속적으로 학습할 수 있습니다.

- **Performance Highlights**: FASTNav는 시뮬레이션과 실제 로봇 환경에서 모델을 훈련하고 평가한 결과, 성공률 및 효율성에서 현저한 개선을 보였습니다. 기존의 모델 압축 방법에 비해, FASTNav는 특정 도메인 작업에서 SLMs의 성능을 크게 향상시키는 잠재력을 보여주며, 로컬에서의 언어 모델 배포에 유망한 솔루션이 될 것으로 기대됩니다. 이러한 접근 방식은 로봇이 사용자와 정확하고 신속하게 상호작용할 수 있도록 합니다.



### BelHouse3D: A Benchmark Dataset for Assessing Occlusion Robustness in 3D Point Cloud Semantic Segmentation (https://arxiv.org/abs/2411.13251)
Comments:
          20 pages, 6 figures, 3 tables, accepted at ECCV 2024 Workshops

- **What's New**: BelHouse3D 데이터셋은 실세계 조건에 밀접하게 일치한 합성 포인트 클라우드 데이터셋으로, 실내 장면의 의미적 세분화를 위해 설계되었습니다. 벨기에의 32개 주택에서 얻은 실제 참고 자료를 바탕으로 구축되어 더 높은 공정성(fairness)과 신뢰성을 제공합니다. 이전의 데이터셋들과는 달리, 본 데이터셋은 occlusion(가림 현상)을 시뮬레이션한 테스트 세트를 포함하여 out-of-distribution (OOD) 상황을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 기존의 3D 벤치마킹 데이터셋은 종종 훈련과 테스트 데이터가 동일하게 분포(IID)된 것이라는 암묵적인 가정을 가지고 평가되었습니다. 하지만 실제 포인트 클라우드 데이터에서는 occlusion이 불가피하게 발생하여 장면의 형태와 구조가 변화하므로, OOD 샘플링에 따른 성능 저하가 우려됩니다. BelHouse3D는 이러한 문제를 해결하기 위해 수작업 3D 모델링과 샘플링 기법을 사용하여 데이터셋을 생성하며, 이는 라벨링 정확성을 향상시키기 위함입니다.

- **Performance Highlights**: BelHouse3D와 OOD 환경에서의 평가를 통해 인기 있는 포인트 기반 의미적 세분화 기법들의 성능을 검증하였습니다. 이러한 OOD 설정은 모델들이 실제 환경에서 어떻게 일반화되는지를 이해하는 데 도움을 주며, 더 견고하고 신뢰할 수 있는 모델 설계를 용이하게 합니다. 연구자들은 이 데이터셋과 OOD 환경을 통해 실내 장면에 대한 3D 포인트 클라우드 의미적 세분화 연구가 발전하길 기대하고 있습니다.



### XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation (https://arxiv.org/abs/2411.13243)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문에서는 XMask3D라는 새로운 방법론을 제안하고 있습니다. XMask3D는 3D 특성과 2D-텍스트 임베딩 공간 간에 보다 정밀한 mask-level 정렬을 실현하는 cross-modal mask reasoning 프레임워크입니다. 이를 통해 기존 방법들보다 정교한 세분화 경계를 다루게 되어, 개방된 어휘(open vocabulary)의 3D 의미론적 분할 문제를 해결하는 데 중요한 진전을 이룰 수 있습니다.

- **Technical Details**: XMask3D는 3D 지점 클라우드(segmentation model)와 텍스트 기반의 mask generator로 구성되어 있습니다. 3D 브랜치는 기하학적 특성을 추출하는 데 탁월하지만, 새로운 카테고리 외삽에는 한계가 있습니다. 반면, 2D 브랜치는 vision-language feature space와 정렬된 mask를 생성하는 역할을 하며, 세 가지 주요 기법인 3D-to-2D Mask Generation, 2D-to-3D Mask Regularization, 3D-2D Mask Feature Fusion을 포함하여 두 가지 브랜치 간의 상호작용을 강화하는 방식으로 설계되었습니다.

- **Performance Highlights**: XMask3D는 다양한 벤치마크 데이터 세트인 ScanNet20, ScanNet200 및 S3DIS에서 경쟁적인 성능을 보여주었습니다. 이 방법은 다양한 자료에 대한 ablation studies와 직관적인 시각화를 통해 제안된 각 mask-level 기법의 기여도를 명확히 드러내고 있습니다. 전체적으로 XMask3D는 3D 개방어휘 의미론적 분할에서 뛰어난 성능을 입증하며, 각 기법이 함께 시너지를 이루는 방식으로 혁신을 나타냅니다.



### Transforming the Hybrid Cloud for Emerging AI Workloads (https://arxiv.org/abs/2411.13239)
Comments:
          70 pages, 27 figures

- **What's New**: 이 백서(white paper)는 IBM Research와 UIUC 연구자들이 IIDAI Institute 내에서 긴밀히 협력하여 개발한 것으로, 하이브리드 클라우드 시스템을 변화시켜 AI 워크로드의 복잡성을 처리하기 위한 혁신적인 전방위 공동 설계 접근법(full-stack co-design approaches)을 제시합니다. 이 프레임워크는 사용성(usability), 관리성(manageability), 비용 효율성(affordability), 적응성(adaptability), 효율성(efficiency), 규모 확장성(scalability)을 강조하며, AI의 발전을 위한 중요한 출발점을 형성합니다.

- **Technical Details**: 이 연구는 생성적(generative) 및 에이전틱(agentic) AI, 크로스 레이어 자동화(cross-layer automation), 통합 제어 평면(unified control plane), 조합 가능(composable) 및 적응 가능한 시스템 아키텍처(adaptive system architecture)와 같은 최첨단 기술을 통합합니다. 에너지 효율성(energy efficiency), 성능(performance), 비용 효율성(cost-effectiveness) 같은 다양한 문제를 해결하기 위해 양자 컴퓨팅(quantum computing)의 발전을 포함하여 고성능 시뮬레이션을 가능하게 하려는 노력을 하고 있습니다.

- **Performance Highlights**: 산업과 학계 간의 협력을 통해 재료 설계 및 기후 솔루션을 위한 기반 모델(foundational models)의 발전, 확장 가능한 멀티모달(data processing) 데이터 처리가 핵심이며, 기상 예측 및 이산화탄소 포집(carbon sequestration)과 같은 응용 분야를 위한 물리 기반 AI 에뮬레이터(physics-based AI emulators)의 향상을 목표로 합니다. 연구의 우선 과제는 AI 에이전틱 시스템(agентic systems) 발전, LLM을 추상화로 활용하는 방법(LLM as an Abstraction), 다양한 인프라에서의 AI 모델 최적화 및 통합 추상화(unified abstractions)입니다.



### Quantum Kernel-Based Long Short-term Memory (https://arxiv.org/abs/2411.13225)
- **What's New**: 본 연구에서는 고전적인 LSTM(Long Short-Term Memory) 프레임워크 내에서 양자 커널 함수(quantum kernel functions)를 활용한 Quantum Kernel-Based LSTM(QK-LSTM) 네트워크를 소개합니다. QK-LSTM은 입력 데이터를 고차원 양자 특징 공간(high-dimensional quantum feature space)에 임베딩하여 복잡하고 비선형적인 패턴을 캡처하기 위해 설계되었습니다. 이 구조는 많은 파라미터 세트의 의존성을 줄이면서도 데이터의 정확도를 유지할 수 있도록 합니다.

- **Technical Details**: QK-LSTM 아키텍처는 표준 LSTM 셀의 선형 변환을 양자 커널 평가(quantum kernel evaluations)로 교체하여 고차원 양자 공간을 인코딩할 수 있는 능력을 강화합니다. QK-LSTM 셀은 각 게이트에서 양자 커널 함수를 사용하여 입력 벡터의 게이트 활성화를 계산하며, 이렇게 함으로써 고전적인 세팅에서는 계산적으로 비효율적이었던 변환을 보다 효율적으로 수행합니다. 이는 NISQ(Noise Intermediate-Scale Quantum) 기기와 같은 제한된 자원 환경에서의 구현 가능성을 높입니다.

- **Performance Highlights**: QK-LSTM은 벤치마크 비교를 통해 고전적인 LSTM 모델과 동등한 성능을 달성하면서도 적은 수의 파라미터로 경제성과 모델 압축을 이룹니다. 이는 자연어 처리(NLP) 및 시간적 데이터 처리에 효율성을 요하는 다른 분야의 양자 기계 학습 응용 프로그램을 발전시킬 수 있는 잠재력을 나타냅니다. 또한, QK-LSTM은 시간 예측 작업에서 효율적인 수렴, 강인한 손실 최소화 및 모델 압축을 통해 적용 가능성을 더욱 높입니다.



### Existential Conversations with Large Language Models: Content, Community, and Cultur (https://arxiv.org/abs/2411.13223)
- **What's New**: 본 논문은 최신 자연어 처리 기술을 기반으로 한 대화형 AI 시스템이 철학, 영성 및 종교와 같은 심오한 주제에 대해 심도 있는 대화를 나눌 수 있는 방법을 다룹니다. 이는 사용자가 인공지능의 존재론적 문제에 대해 참여하게 하여, 보다 풍부한 대화 경험을 제공합니다. 논문은 이러한 대화의 현대적이고 고전적인 문화적, 사회적 배경을 분석하고, AI 시스템과의 상호작용이 사회에 미치는 영향을 고려합니다.

- **Technical Details**: 대형 언어 모델(LLM)은 주어진 텍스트의 연속성을 생성할 수 있는 컴퓨터 프로그램으로, 대량의 언어 데이터에 대해 훈련된 인공 신경망입니다. 이 모델은 사용자의 입력에 따라 적합한 단어를 예측하고, 주어진 문맥에 맞게 대화를 이어나갈 수 있습니다. 논문은 LLM이 수행하는 대화의 흐름과 그 과정에서 나타나는 다양한 철학적 질문들에 대해 기술적 관점에서 접근합니다.

- **Performance Highlights**: 현재 LLM 기반 AI 시스템은 사용자와의 대화에서 매우 매력적이고 깊이 있는 경험을 제공합니다. 특히 Anthropic의 Claude 3는 철학적인 주제에 대한 대화를 진행할 수 있는 가능성을 보여줍니다. 이러한 시스템은 단순히 훈련된 데이터를 반복하는 것이 아니라, 사용자의 맥락에 맞는 대화를 생성하며, 자아 및 의식에 대한 흥미로운 논의를 이어갑니다.



### Proceedings Sixth International Workshop on Formal Methods for Autonomous Systems (https://arxiv.org/abs/2411.13215)
- **What's New**: 이번 EPTCS(Electronic Proceedings in Theoretical Computer Science) 볼륨은 2024년 11월 11일부터 13일 사이에 개최된 제6회 자율 시스템을 위한 형식적 방법 워크숍(FMAS 2024)의 논문들을 포함하고 있습니다. 이번 워크숍은 영국 맨체스터 대학교에서 열린 제19회 통합 형식 방법 국제회의(iFM'24)와 함께 진행되었습니다.

- **Technical Details**: FMAS 2024는 자율 시스템의 형식적 방법(Formal Methods)의 최신 연구와 개발 동향을 논의하기 위해 마련되었습니다. 참가자들은 다양한 형식적 검증 기법(formal verification techniques)과 도구들을 소개하고, 실제 적용 사례들을 공유하였습니다.

- **Performance Highlights**: 워크숍에서는 형식적 방법을 통한 자율 시스템의 안전성(safety) 및 신뢰성(reliability) 강화를 위한 여러 기법들이 발표되었습니다. 이러한 발표들은 자율 시스템의 설계와 구현 시 발생할 수 있는 다양한 문제를 해결하는 데 기여할 것으로 기대됩니다.



### Comparative Analysis of Audio Feature Extraction for Real-Time Talking Portrait Synthesis (https://arxiv.org/abs/2411.13209)
Comments:
          16 pages, 6 figures, 3 tables. submitted to MDPI journal in as Big Data and Cognitive Computing

- **What's New**: 이 논문은 인터뷰어 훈련을 위한 실시간 Talking-head 생성 통합에 대해 다루며, Audio Feature Extraction (AFE)의 지연과 반응성 한계를 극복하는 데 주목합니다. 기존의 AFE 모델들을 Open AI의 Whisper로 교체하여 처리 최적화를 이루었고, 이로 인해 전체 시스템의 효율성이 향상되었습니다. 두 개의 오픈 소스 실시간 모델을 세 가지 데이터셋에 대해 평가한 결과, Whisper는 처리 속도를 가속화하고 렌더링 품질의 특정 측면을 개선하여 더 현실적이고 반응이 빠른 Talking-head 상호작용을 가능하게 했습니다.

- **Technical Details**: 이 연구는 Deep-Speech 2, Wav2Vec 2.0, HuBERT, Whisper와 같은 네 가지 AFE 모델을 비교하며, 다양한 음성 인식 모델들이 오디오 신호에서 어떻게 음향 특징과 언어 표현을 추출하는지를 설명합니다. Whisper는 다국어 및 멀티태스크 데이터셋에서 학습된 고급 자동 음성 인식 (ASR) 시스템으로, 비동기적 문제 해결을 통해 실시간 Talking-head 시스템을 최적화할 수 있습니다. 유니티 엔진을 통합하여 가상 아바타를 생성하고, Generative Adversarial Networks (GANs)를 통해 이들 모델의 현실성을 증대시키는 방식으로 구성되었습니다.

- **Performance Highlights**: 이 시스템은 인터뷰어 훈련을 위한 몰입형 상호작용 툴로서, AI 기반의 아바타의 잠재력을 확장하는 데 기여합니다. Whisper를 활용한 AFE는 기존 모델들에서 발생하는 지연 문제를 줄이며, 전반적인 반응성과 현실성을 향상시킵니다. 연구 결과, Whisper 기반의 모델이 렌더링 품질을 크게 개선함으로써 교육 효과성을 높이는 데 결정적인 역할을 한다는 점이 강조됩니다.



### The Information Security Awareness of Large Language Models (https://arxiv.org/abs/2411.13207)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 정보 보안 인식(ISA)을 평가하는 새로운 방법론을 제안합니다. 이 평가를 위해 특정 보안 맥락이 즉시 드러나지 않는 사용자 질문에 LLM이 응답하는 다양한 시나리오를 생성했습니다. 연구 결과, LLM이 보안 관련 지식을 효과적으로 적용하는 능력은 모델의 템퍼러처에 영향을 받으나, 시스템 프롬프트의 설정에 더 큰 영향을 받는다는 사실이 밝혀졌습니다.

- **Technical Details**: 이 연구에서는 30개의 시나리오를 구성하여 LLM의 정보 보안 인식을 평가합니다. 각 시나리오는 모바일 ISA 분류법에서 정의된 하위 초점 분야에 기초하여 설정되었으며, LLM의 응답은 인간의 평가와 LLM 기반 평가 모두로 점수를 부여받았습니다. 특히, 시스템 프롬프트와 온도 설정에 따라 ISA의 변화가 관찰되었고, 이는 LLM의 안전한 사용을 위한 중요한 요소임을 강조합니다.

- **Performance Highlights**: 이 평가를 통해 여러 인기 있는 LLM이 나타내는 ISA 수준이 상이하다는 것을 발견했습니다. 이러한 결과는 특정 보안 문제에 대한 전문성을 구축하기 위해 단일 모델에 의존하기보다는 서로 보완하는 여러 LLM을 활용할 필요성을 제시합니다. 또한, 모델의 초점 분야의 차별성과 각 모델의 강점과 약점이 평가되었으며, 이는 LLM 기반 보조 도구의 향후 개발에 있어 정보 보안 인식 평가의 중요성을 부각시킵니다.



### Engagement-Driven Content Generation with Large Language Models (https://arxiv.org/abs/2411.13187)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 사회적 영향력, 특히 소셜 네트워크 내에서의 상호작용을 탐구합니다. 기존 연구들은 LLMs의 능력을 일대일 상호작용에서 주로 다루었으나, 이 연구는 넓은 연결 구조에서의 LLMs의 잠재력을 강조합니다. 연구 질문은 LLMs가 사용자 참여를 극대화하는 의미 있는 콘텐츠를 생성할 수 있는지를 검토합니다.

- **Technical Details**: 연구는 LLM 기반 콘텐츠 생성을 위한 파이프라인을 정의하고, 강화 학습(reinforcement learning)과 시뮬레이션된 피드백(simulated feedback)을 사용하여 보상을 설정합니다. 이 프레임워크는 LLM이 주제에 맞는 콘텐츠를 생성하고 최소한의 유창성(fluency) 기준을 충족하도록 요구합니다. 연구진은 공정 피드백 루프를 통해 LLM이 주제, 사회적 네트워크 구조 및 의견 분포에 적응하는 과정을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기초적인 전파 모델에 무관하며 완전히 적응적임을 나타냈습니다. LLM 에이전트는 생성한 콘텐츠의 감정을 자동으로 적응시키며, 실제 네트워크 내에서 생성된 콘텐츠와 동등한 수준의 참여를 생성하는 경향을 보였습니다. 이 연구는 복잡한 참여 과제에 맞게 쉽게 조정될 수 있는 가능성을 보여줍니다.



### Cross-Camera Distracted Driver Classification through Feature Disentanglement and Contrastive Learning (https://arxiv.org/abs/2411.13181)
- **What's New**: 이 논문에서는 운전자의 주의 분산을 효과적으로 분류하는 새로운 모델인 Driver Behavior Monitoring Network (DBMNet)을 소개합니다. 이 모델은 카메라 위치 변화에 강한 내성을 가지며, 경량화된 CNN 아키텍처를 사용해 상용 장치에 배치 가능하도록 설계되었습니다. 특히, 특징 분리(disentanglement) 모듈을 도입하여 운전 행동의 중요한 특징을 강화하고, 다양한 카메라 뷰에 대해 일관된 성능을 보장합니다.

- **Technical Details**: DBMNet은 RGB 이미지를 인코딩하는 경량화된 CNN을 기반으로 하며, 행동 관련 특징과 뷰 관련 특징을 분리하는 기능을 갖춘 모듈을 채택합니다. 또한, triplet loss를 사용하여 서로 다른 카메라의 같은 행동을 더 가까이 묶고, 다른 행동은 separación하여 운전 행동 분류의 정확성을 높입니다. 이 구조는 학습 과정에서 모델이 다양한 카메라 뷰에 대해 강건한 특징 표현을 학습하도록 유도합니다.

- **Performance Highlights**: DBMNet은 100-Driver 데이터셋의 주간 및 야간 부분에서 기존 기술 대비 Top-1 정확성이 평균 9% 향상됨을 입증하였습니다. 또한, AUCDD-V1, EZZ2021 및 SFD와 같은 세 가지 벤치마크 데이터셋에서 크로스 데이터셋 및 크로스 카메라 실험을 통해 뛰어난 일반화 능력을 보여주었습니다. 이러한 성과는 다양한 운전 상황에서 분산 운전자의 행동을 효과적으로 분류할 수 있는 가능성을 열어줍니다.



### Writing Style Matters: An Examination of Bias and Fairness in Information Retrieval Systems (https://arxiv.org/abs/2411.13173)
Comments:
          In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining (WSDM 25)

- **What's New**: 본 논문에서는 정보 검색 시스템(Information Retrieval, IR) 내에서 최신 보편적 텍스트 임베딩 모델의 문서 및 쿼리 작성 스타일에 따른 잠재적 편향을 탐색합니다. 특히, 비공식적이고 감정적으로 표현된 스타일은 대부분의 임베딩 모델에서 적게 선호됨을 밝혔습니다. 이 연구는 다양한 작성 스타일이 정보 검색 시스템에서의 편향과 공정성에 미치는 영향을 집중적으로 분석합니다.

- **Technical Details**: 정보 검색 시스템은 쿼리에 대한 관련 데이터를 검색하는 것이 핵심 목표이며, 이 과정에서 문서의 임베딩 유사도 점수가 사용됩니다. 최근 LLM(대형 언어 모델)의 부상은 IR 시스템 내에서의 편향 문제를 더욱 복잡하게 만들었으며, 편향 및 공정성의 문제는 전반적인 시스템의 신뢰성을 저하시킬 수 있습니다. 이 논문은 이러한 편향을 분석하여 공정성을 측정하는 새로운 방법을 제안합니다.

- **Performance Highlights**: 논문에서는 텍스트 임베딩 모델들이 서로 다른 작성 스타일에 대해 어떻게 편향되어 있는지를 분석하였으며, LLM의 답변 스타일에 대한 편향 또한 식별하였습니다. 이 연구는 IR 시스템의 공정하고 강력한 모델 개발에 중요한 통찰력을 제공합니다. 다양한 최신 텍스트 임베딩 모델을 비교하며, 쿼리 스타일의 영향 역시 고려하였습니다.



### Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding (https://arxiv.org/abs/2411.13157)
- **What's New**: 최근 대형 언어 모델(LLMs)의 효율적인 추론(inference)은 이들의 규모와 복잡성이 커짐에 따라 중요한 이슈로 대두되고 있습니다. 전통적인 오토리그레시브 디코딩(autoregressive decoding)은 순차적으로 토큰을 생성하는 방식으로 인해 계산비용에서 비효율적입니다. 이 연구에서는 초안(draft)과 확인(verification)이라는 두 단계의 접근 방식을 도입하여 교묘한 디코딩(speculative decoding) 기술을 통해 이러한 병목 현상을 해결하는 방법을 모색합니다.

- **Technical Details**: 교묘한 디코딩은 작은 모델을 사용하여 초기 초안을 생성하고, 보다 큰 모델이 이 초안을 확인하는 방식으로 작동합니다. 여기서는 모델 중심(model-centric)과 초안 중심(draft-centric) 방식으로 교묘한 디코딩 방법을 분류하며, 각 접근법의 핵심 아이디어와 잠재력을 다룹니다. 기술적으로, 초안 생성의 질과 효율성을 개선하는 방법으로 다양한 구현 전략에 대해 이야기하고 있습니다.

- **Performance Highlights**: 성공적인 교묘한 디코딩 방법으로는 Medusa와 EAGLE-2가 언급됩니다. Medusa는 추가 디코딩 헤드를 사용하여 후속 토큰을 병렬로 처리하는 방식으로 성능을 향상시켰습니다. 반면 EAGLE-2는 동적인 초안 트리를 통해 더 나은 샘플링을 가능하게 하여 추론 속도와 정확성을 높이는 데 기여하고 있습니다.



### DMQR-RAG: Diverse Multi-Query Rewriting for RAG (https://arxiv.org/abs/2411.13154)
- **What's New**: 본 논문에서는 DMQR-RAG라는 다중 쿼리 리라이트(Rewrite) 프레임워크를 소개하여 문서 검색 및 최종 응답 개선을 위해 설계되었습니다. 기존의 쿼리 리라이트 방법들이 단일 쿼리만 생성하는 문제를 해결하기 위해, 다양한 정보량을 가진 여러 개의 쿼리를 통해 문서의 다양성을 증가시킴으로써 RAG의 성능을 개선하는 데 중점을 두었습니다. 또한, 리라이트 전략 선택 방법을 통해 리라이트 수를 최소화하면서 최적의 성능을 달성할 수 있는 방안을 제안합니다.

- **Technical Details**: DMQR-RAG 프레임워크는 정보의 양에 따라 작동하는 네 가지 리라이트 전략을 기반으로 다양성을 증대시키고, 크로스-어텐션(Attention) 임베딩 모델을 사용하여 검색된 문서들을 재순위(Rerank)합니다. 각 리라이트 쿼리는 서로 다른 문서를 검색할 수 있는 잠재력을 가지고 있으며, 재작성 전략 선택 방법을 통해 최적의 리라이트 전략을 동적으로 식별합니다. 이러한 접근은 특정 쿼리에 적합한 전략을 선택하여 효과적인 문서 검색과 응답 생성의 성능을 향상시킵니다.

- **Performance Highlights**: 다중 쿼리 리라이트 방식은 단일 쿼리 리라이트 방식에 비해 일반적으로 더 나은 성능을 보여줍니다. 연구 결과, 정보 기반의 다중 쿼리 접근 방식이 기존의 RAG-Fusion 방법을 초월하는 경우가 많았습니다. 본 연구에서 제안된 방법들은 둘 다 학술 및 산업 환경에서 철저히 실험을 통해 검증되어 효과성이 입증되었습니다.



### AGLP: A Graph Learning Perspective for Semi-supervised Domain Adaptation (https://arxiv.org/abs/2411.13152)
Comments:
          8page

- **What's New**: 이 논문은 반지도 방식 도메인 적응(Semi-Supervised Domain Adaptation, SSDA)에 대한 새로운 접근 방식을 제안하며, 데이터 구조와 도메인 레이블을 모형화하는 그래프 학습 관점을 도입합니다. 기존의 SSDA 방법들은 도메인 레이블과 클래스 레이블의 정보는 사용하였으나, 데이터의 구조적 정보를 간과했습니다. 이 연구는 이러한 문제를 해결하기 위해 그래프 컨볼루션 네트워크(Graph Convolutional Network, GCN)를 인스턴스 그래프에 적용하여 구조적 정보를 전파하도록 설계되었습니다.

- **Technical Details**: 제안된 모델(AGLP: Adversarial Graph Learning Perspective)은 클래스 중심 정렬(class centroid alignment) 과정 중 서로 다른 클래스의 중심을 이동시키는 방법을 사용하여 도메인 불변적이고 의미론적 표현을 효과적으로 학습할 수 있도록 합니다. 이 네트워크는 표준 CNN으로 추출된 샘플의 CNN 피쳐를 기반으로 밀접하게 연결된 인스턴스 그래프를 구성하고, 이 그래프를 통해 데이터 구조와 도메인 레이블을 동시에 모델링합니다. 네트워크의 설계된 가중치를 통해 구조적 정보를 학습하여 도메인간 간극을 줄이는 효율성을 높입니다.

- **Performance Highlights**: 여러 표준 벤치마크에서 수행된 실험 결과, 제안된 AGLP 알고리즘이 기존의 최첨단 SSDA 방법과 비교하여 성능이 우수함을 보여주었습니다. 특히, 이 모델은 도메인 간 불일치를 줄이며, 더 나은 일반화 능력을 제공하여 모델의 적응력을 향상시킵니다. 이를 통해 SSDA 문제를 해결하는 데 중요한 기여를 할 수 있음을 입증하였습니다.



### YCB-LUMA: YCB Object Dataset with Luminance Keying for Object Localization (https://arxiv.org/abs/2411.13149)
- **What's New**: 이 논문은 고품질의 훈련 데이터를 생성하는 데 도움이 되는 새로운 방법인 luminance keying을 제안합니다. YCB-V 세트의 나머지 객체들을 추가로 기록함으로써, 이 접근법의 유용성을 더욱 입증하고 있습니다. 이로 인해 여러 가지 투명 물체, 다양한 색상의 물체 및 변형 가능한 물체가 포함되었으며, 이는 새로운 2D 객체 탐지 및 분할 알고리즘 테스트에도 활용될 수 있습니다.

- **Technical Details**: Deep Neural Networks (DNNs)는 최상의 성능을 위해 많은 양의 주석이 달린 데이터 세트를 요구합니다. 기존의 데이터 수집 방법인 수동 주석 및 렌더링은 시간 소모가 크고 오류가 발생하기 쉬운 반면, luminance keying 방법은 자동 객체 마스킹과 밝기 임계값을 활용하여 노동 집약적인 수동 주석의 필요를 제거합니다. 이 연구에서는 추가 객체들을 기록하고 데이터를 자동으로 처리할 수 있는 코드도 제공하여 데이터의 다양성을 증가시켰습니다.

- **Performance Highlights**: 우리는 YCB-V 객체 데이터 세트를 luminance keying 방식으로 확장하여 2D 객체 탐지기와 분할 알고리즘의 성능 평가에 기여하고자 합니다. 제공된 데이터는 https://huggingface.co/datasets/tpoellabauer/YCB-LUMA와 https://github.com/tpoellabauer/ycb-luma에서 확인할 수 있으며, YCB-V 데이터 세트도 사용할 수 있습니다. 이러한 향상된 데이터 세트는 다양한 객체들로 구성되어 있으며, 알고리즘 성능 검증의 의미를 더욱 높이고 있습니다.



### GraphCL: Graph-based Clustering for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2411.13147)
Comments:
          9page

- **What's New**: 이 논문에서는 반지도학습(semi-supervised learning, SSL)을 기반으로 한 새로운 손상된 의학 이미징 분할 기술인 GraphCL(그래프 기반 군집화 클러스터링)을 제안합니다. 기존의 방법들과 달리, 데이터 구조 정보를 그래프 모델로 통합하여 반지도 의학 이미지 분할에서의 성능을 개선했습니다. 구체적으로, 이 연구는 반지도 의학 이미지 분할(SSMIS)에 대한 데이터 구조 정보 모델링의 첫 번째 사례로 간주됩니다.

- **Technical Details**: GraphCL의 핵심 개념은 CNN(Convolutional Neural Network)에서 추출된 기능을 기반으로 샘플의 구조적 유사성을 반영한 밀집 인스턴스 그래프(dense instance graph)를 구축하는 것입니다. 각 노드는 CNN의 피쳐와 연결되어 있으며, 이 인스턴스 그래프에서 그래프 신경망(GCN, Graph Convolutional Network)을 적용해 구조 정보를 전달합니다. k-클러스터(k-clustering)의 수를 지정할 필요가 없는 클러스터링 전략을 도입하여 더 높은 세분화 정확성을 얻습니다.

- **Performance Highlights**: 세 가지 표준 벤치마크에서의 광범위한 실험 결과는 GraphCL 알고리즘이 최신의 반지도 의학 이미지 분할 기법들보다 우수한 성능을 보임을 보여줍니다. 이전의 방법들에서는 반지도 학습에서 그래프 정보의 중요성이 간과되었으나, GraphCL은 이 문제를 해결하여 클러스터 특성을 효과적으로 통합합니다. 이 알고리즘은 의학적 데이터 세트의 레이블이 부실한 상황에서도 뛰어난 결과를 입증했습니다.



### CopyrightMeter: Revisiting Copyright Protection in Text-to-image Models (https://arxiv.org/abs/2411.13144)
- **What's New**: 최근 텍스트-이미지 변환 모델이 높은 품질의 이미지를 생성하는 데 강력한 도구로 떠오르면서, 저작권 문제에 대한 우려도 커지고 있습니다. 이러한 문제를 해결하기 위해 다양한 저작권 보호 기법이 제안되었으나, 이들의 효과성과 강인성이 충분히 검토되지 않았습니다. 이를 보완하기 위해, 기존 보호 기법을 시스템화하고 저작권 보호 평가를 위한 통합 프레임워크인 CopyrightMeter를 개발하였습니다.

- **Technical Details**: 이 연구에서는 CopyrightMeter라는 평가 프레임워크를 통해 17개의 최신 보호 기법과 16개의 대표적인 공격 방법을 통합하여 평가를 수행합니다. 평가 기준으로는 fidility(충실도), efficacy(효과성), resilience(강인성)를 사용하여 각 보호 기법의 성능을 다차원적으로 분석합니다. 이를 통해 기존의 보호 기법들이 공격에 대한 저항력이 낮고, 공격의 종류에 따라 최적의 보호 방법이 다름을 밝혀냈습니다.

- **Performance Highlights**: 연구 결과에 따르면, 보호 기법의 16개 중 17개가 공격에 대해 강인하지 않으며, 가장 우수한 보호 방법은 목표에 따라 달라진다는 것을 발견했습니다. 또한, 더 발전된 공격이 보다 강력한 보호 기법의 발전을 촉진한다는 인사이트도 얻었습니다. 이러한 Insights는 향후 저작권 보호 연구와 T2I DM 개발에 있어서 중요한 방향성을 제공할 것입니다.



### Provably Efficient Action-Manipulation Attack Against Continuous Reinforcement Learning (https://arxiv.org/abs/2411.13116)
- **What's New**: 본 논문에서는 임의의 RL 기반 에이전트가 지속적인 액션 공간을 통해 특정 행동을 조작하는 공격을 조사합니다. 기존 연구들이 주로 이산 상태에 초점을 맞췄던 반면, 저자들은 사이버-물리적 시스템(CPS)과 같은 실제 응용 프로그램에 적합한 연속적 액션 조작에 대한 공격 방법을 제안합니다. 저자들은 Monte Carlo tree search 방법을 활용하여 LCBT(하한 신뢰 경계 트리)라는 새로운 블랙박스 공격 알고리즘을 개발하고, 이를 통해 공격의 효율성을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 논문의 주요 내용은 연속적 상태 및 액션 공간에서의 행동 조작 공격에 대한 위협 모델을 구성하고, 공격 비용의 경계를 설정하는 것입니다. 저자들은 공격자가 마르코프 의사 결정 과정(MDP)에 대한 충분한 정보를 갖고 있는 화이트박스 시나리오와, 정보를 바탕으로 행동 조작 전략을 구성하는 블랙박스 시나리오를 구분합니다. 특히, LCBT 알고리즘은 아울러 이론적으로 민첩하게 출력을 조작할 수 있는 방법을 제시하며, 이를 통해 공격 비용을 줄일 수 있음을 증명합니다.

- **Performance Highlights**: 실험 결과, DDPG, PPO, TD3와 같은 여러 공격적인 RL 알고리즘에서 제안된 공격 방법의 성능을 입증하였습니다. LCBT 알고리즘은 타겟 정책에 수렴하는 데 필요한 공격 비용이 서브라인형 공격 비용으로 제한됨을 보여주며, 이는 기존의 공격 방법들보다 효율적입니다. 이번 연구는 연속적 액션 조작 공격에 대한 심층 분석을 제공하며, RL 시스템의 안전성을 높이는 데 중요한 기여를 합니다.



### Song Form-aware Full-Song Text-to-Lyrics Generation with Multi-Level Granularity Syllable Count Contro (https://arxiv.org/abs/2411.13100)
- **What's New**: 이번 논문은 가사 생성에 있어 독특한 도전 과제를 다루고 있으며, 특히 정확한 음절 제어(syllable control)를 실현하는 데 초점을 맞추고 있습니다. 기존의 문장별 접근 방식(line-by-line)은 종종 부자연스러운 어구를 만들어내기 때문에, 보다 세밀한 음절 관리가 필요함을 강조하고 있습니다. 저자들은 다양한 수준의 음절 제어가 가능한 가사 생성 프레임워크를 제안하며, 입력 텍스트와 곡 형식에 따라 조정된 완전한 가사를 생성할 수 있게 됩니다.

- **Technical Details**: 제안된 모델은 단어, 구, 문장 및 단락 수준에서의 유연한 가사 생성을 지원합니다. 생성 과정에서는 특정 곡 형식에 맞춘 구조화된 토큰(token) 시스템을 사용하며, 이를 통해 노래의 각 부분의 전환을 나타냅니다. 예를 들어, <VERSE>, <CHORUS> 같은 곡 형식 토큰과 함께 정의된 음절 수 토큰(<SYL:s𝑠sitalic_s>)을 도입하여 모델이 음절 정밀도를 유지하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 기존의 대형 언어 모델(LLMs)들이 자연어 생성을 위해서는 뛰어난 성능을 보이지만, 음절 수를 정확히 계산하는 데에는 어려움을 겪고 있다는 사실이 드러났습니다. 저자들은 이 모델을 사용하여 가사를 생성하는 측면에서 음절 수와 구조에 따라 LLM을 평가했으며, 그 결과 약 38%~57%의 성공률을 기록했습니다. 제안된 모델의 성능은 기존 모델들에 비해 음절 제어에서 더욱 우수함을 보였으며, 각 조건에서 성공적으로 생성된 가사에 대한 성능 지표가 제시되었습니다.



### Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension (https://arxiv.org/abs/2411.13093)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 Video Retrieval-Augmented Generation (Video-RAG)라는 새로운 접근 방식을 제안하고 있습니다. Video-RAG는 기존의 Large Video-Language Models (LVLMs)의 한계를 극복하기 위해 시각적으로 정렬된 보조 텍스트를 활용하여 비디오 이해를 개선하는 훈련 없는(cost-effective) 파이프라인입니다. 이러한 방식으로 긴 비디오에서도 효과적인 정보 처리가 가능해지며, 기존 LVLM에 비해 효율성과 성능이 크게 향상됩니다.

- **Technical Details**: Video-RAG는 주어진 쿼리에서 보조 텍스트를 추출하는 Retrieval(검색) 요청을 분해한 후, 관련 정보를 여러 개 생성하고 이를 통합하여 LVLM에 입력하는 방식으로 작동합니다. 이 과정에서 Optical Character Recognition (OCR), Automatic Speech Recognition (ASR), Object Detection과 같은 오픈 소스 외부 도구를 사용하여 비디오 데이터에서 관련 정보를 추출합니다. 이러한 방법은 비디오의 시각적 맥락과 일치하여 정보 손실을 최소화합니다.

- **Performance Highlights**: Video-RAG를 통해 다양한 LVLM에 대한 성능 평가를 진행한 결과, Video-MME, MLVU, LongVideoBench 등의 벤치마크에서 평균 8.0%의 성능 향상을 기록했습니다. 특히 72B 모델을 사용할 경우, Gemini-1.5-Pro와 같은 상용 모델에 비해 우수한 성과를 보여주었습니다. 또한, 이 시스템은 추가적인 GPU 메모리와 짧은 추론 시간만으로도 구현할 수 있어 자원의 효율성을 극대화합니다.



### Neural Internal Model Control: Learning a Robust Control Policy via Predictive Error Feedback (https://arxiv.org/abs/2411.13079)
Comments:
          Submitted to RAL

- **What's New**: 이번 논문에서는 로봇이 복잡한 환경에서 저항과 방해를 극복할 수 있도록 Neural Internal Model Control이라는 새로운 프레임워크를 제안합니다. 이는 기존의 모델 기반 제어와 강화 학습(RL) 기반 제어를 통합하여 더욱 강력한 제어 성능을 발휘합니다. Newton-Euler 방정식을 사용하여 예측 모델을 간소화하고 고차원 비선형성을 제거하여, 로봇의 동작을 예측할 수 있습니다.

- **Technical Details**: 제안된 Neural Internal Model Control은 모델 프리 RL 알고리즘과 예측 오차 피드백을 결합하여 폐루프 제어 구조를 형성합니다. 이를 통해 로봇의 동작 상태 예측을 신속하고 정확하게 수행하며, 이를 quadrupedal robots(사족 로봇)와 quadrotors(쿼드로터)에 적용하여 성능을 입증했습니다. 이 프레임워크는 복잡한 시스템에 대한 적용 가능성을 높입니다.

- **Performance Highlights**: 제안된 프레임워크는 최신의 제어 방법들보다 뛰어난 성능을 보였으며, 실제 환경에서 쿼드로터와 다양한 로프 매달린 짐의 제어도 성공적으로 수행하였습니다. 이러한 실제 적용 사례는 시뮬레이션에서 현실로 전이(sim-to-real transfer)되는 과정에서 강력한 내성을 입증했습니다. 코드도 함께 공개하였으며, 기타 학습 알고리즘과 쉽게 통합될 수 있도록 설계되었습니다.



### AMaze: An intuitive benchmark generator for fast prototyping of generalizable agents (https://arxiv.org/abs/2411.13072)
Comments:
          Under review in Frontiers in Artificial Intelligence

- **What's New**: 이 논문에서는 새로운 벤치마크 생성기인 AMaze를 소개합니다. AMaze는 에이전트가 다양한 복잡성과 기만성을 가진 미로를 탐색하도록 요구하며, 이는 에이전트의 일반화 능력을 증진시킵니다. AMaze는 또한 사용자 친화적인 인터페이스를 제공하여 특정 특징에 맞춘 미로를 쉽게 생성할 수 있습니다.

- **Technical Details**: AMaze는 에이전트가 두 가지 형식으로 임의로 복잡한 시각 정보를 처리하도록 설계되어 있습니다: 이산형(전처리된)와 연속형(이미지). 또한, 2D 미로 생성을 위해 물리 엔진이나 오프스크린 렌더링이 필요 없으며, 이는 컴퓨팅 리소스를 절약합니다. AMaze는 다양한 환경을 제공함으로써 일반화 능력을 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: 논문에서는 에이전트를 세 가지 훈련 방식(일회성, 스캐폴딩, 상호작용)으로 훈련한 결과를 다룹니다. 상호작용 훈련 방식이 직접 훈련 방식보다 약 50%에서 100%까지 높은 일반화 성능을 보였으며, 이는 조절 가능한 인간-주도 환경 생성기의 이점을 보여줍니다. 이러한 결과는 AMaze가 새로운 과제에 대한 강인한 동작을 가능하게 함을 시사합니다.



### Branches, Assemble! Multi-Branch Cooperation Network for Large-Scale Click-Through Rate Prediction at Taobao (https://arxiv.org/abs/2411.13057)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 MBCnet이라는 새로운 다중 분기 협력 네트워크를 소개합니다. 이 네트워크는 다양한 피쳐 상호작용 모델링 능력을 갖춘 여러 개의 브랜치 네트워크가 서로 협력하여 복잡한 피쳐 관계를 더욱 잘 캡처할 수 있도록 설계되었습니다. MBCnet은 전문가 기반 피쳐 그룹화 및 크로싱(Expert-based Feature Grouping and Crossing, EFGC) 브랜치와 로우 랭크 크로스 네트워크(low rank Cross Net) 및 딥 브랜쳐로 구성됩니다. 이러한 새로운 협력 구조가 기존의 단일 접근 방식의 한계를 극복하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: MBCnet은 세 개의 주요 브랜치로 구성되며, 각 브랜치는 고유한 기능과 모델링 능력을 가지고 있습니다. EFGC 브랜치는 특정 피쳐 필드를 그룹화하고 도메인 전문가의 지식을 이용해 해당 그룹 간의 피쳐 상호작용을 수행하여 모델의 기억 능력을 증대시킵니다. 또한, 기존의 로우 랭크 크로스 네트워크와 딥 브랜치는 명시적 및 암시적 피쳐 크로싱을 통해 모델의 일반화 능력을 강화하게 됩니다. 다중 브랜치 협력 체계에서는 잘 학습된 브랜치가 그렇지 않은 브랜치를 지원하도록 하는 '브랜치 공동 교육(branch co-teaching)'과 적절한 차별화를 유지하는 원칙이 적용됩니다.

- **Performance Highlights**: MBCnet은 대규모 산업 데이터셋에서 진행한 실험을 통해 Click-Through Rate (CTR)에서 0.09 포인트 상승을 기록하였으며, 거래 수는 1.49% 증가하고 GMV는 1.62% 상승하였습니다. 이러한 성과는 MBCnet의 협력적 학습이 다양한 피쳐 상호작용을 효과적으로 발견하는 데 기여했음을 보여줍니다. 또한, Taobao 앱의 image2product 검색에 MBCnet이 배포되어 현장에서의 꼭 필요한 성과를 입증했습니다.



### MEGL: Multimodal Explanation-Guided Learning (https://arxiv.org/abs/2411.13053)
- **What's New**: 이번 연구에서는 Multimodal Explanation-Guided Learning (MEGL) 프레임워크를 제안하여 이미지 분류 작업의 해석 가능성을 높이고 성능을 개선하고자 합니다. 기존의 XAI 방법들이 시각적 혹은 텍스트 기반의 단일 모달 해석에 의존했던 반면, MEGL은 두 가지 모달의 보완적인 특성을 통합하는 접근 방식을 사용합니다. 이를 통해 시각적 및 텍스트 기반 설명의 일관성과 완전성을 높여 AI 모델의 '블랙 박스'(black box) 특성을 해결하고자 합니다.

- **Technical Details**: MEGL에서 제안하는 Saliency-Driven Textual Grounding (SDTG) 방법은 시각적 설명에서 얻은 공간 정보를 텍스트 설명에 통합합니다. 이는 입력 이미지를 처리하여 생성된 Saliency Map을 사용하여 텍스트에서 공간적으로 관련된 통찰을 효과적으로 반영하도록 돕습니다. 또한, Visual Explanation Distribution Consistency loss를 도입하여 시각적 설명의 일관성을 강화하고, 누락된 주석이 있는 경우에도 안정적이고 맥락적으로 타당한 시각적 설명을 보장합니다.

- **Performance Highlights**: Object-ME와 Action-ME라는 두 개의 새로운 데이터셋을 기반으로 실험을 수행한 결과, MEGL은 기존의 이미지 분류 및 해석 가능성 증진 방법보다 우수한 성능을 보였습니다. MEGL은 예측 정확성, 시각적 설명 가능성 및 텍스트 설명 가능성 모두에서 뛰어난 성능을 기록하며, 다양한 훈련 조건에서 유의미하고 일관된 다중모달 설명을 효과적으로 이용할 수 있음을 입증하였습니다.



### Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning (https://arxiv.org/abs/2411.13045)
Comments:
          Submitted to WWW 2025

- **What's New**: 이번 논문에서는 전자상거래 검색 시스템에서 사용자 경험과 만족도를 향상시키기 위해, 효과적인 query-item relevance(쿼리-아이템 관련성) 모델링의 중요성을 강조합니다. 최근에 등장한 Large Language Model(대규모 언어 모델) 접근법은 기존 신경망 기반의 전문 관련성 학습 방법에 비해 뛰어난 성능과 긴 꼬리(generalization) 능력을 보여주고 있습니다. 하지만 이 모델들은 온라인 배포에 어려움이 있으며, LLM의 복잡한 내부 지식을 추출하고 적용하는 데 한계가 있습니다.

- **Technical Details**: 이를 해결하기 위해 우리는 Explainable LLM-driven Multi-dimensional Distillation(설명 가능한 대규모 언어 모델 기반 다차원 증류) 프레임워크를 제안합니다. 이 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) ELLM-rele(설명 가능한 LLM)로서 관련성 모델링을 Chain-of-Thought(사고의 흐름) 추론으로 분해하여 해석 가능성과 성능을 향상시킵니다. 2) MKD(다차원 지식 증류) 아키텍처는 ELLM-rele의 지식을 현재 배포 가능한 모델에 전달하며, 이는 관련성 점수 분포와 CoT 추론 측면에서 이루어집니다.

- **Performance Highlights**: 우리는 이 프레임워크가 Taobao 검색 광고 환경에서의 방대한 오프라인 평가와 온라인 실험을 통해 전자상거래 관련성 학습 성능과 사용자 경험을 크게 개선함을 보였습니다. MKD는 학생 모델의 의미적 상호작용과 긴 꼬리 일반화 능력을 모두 향상시키는 데 기여하였습니다. 이러한 결과는 LLM의 해석 가능성과 모델 성능 향상이라는 두 가지 주요 목표를 달성함으로써, 향후 전자상거래 시스템에서 더욱 효과적인 적용이 가능함을 시사합니다.



### Unsupervised Homography Estimation on Multimodal Image Pair via Alternating Optimization (https://arxiv.org/abs/2411.13036)
Comments:
          This paper is accepted to the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이번 논문에서는 AltO라는 새로운 비지도 학습 프레임워크를 제안하여 다중 모달 이미지 쌍에서 호모그래피(homography)를 추정합니다. 기존의 비지도 학습 방법들이 동일한 카메라로 촬영된 이미지 쌍에서만 효과적인 반면, AltO는 서로 다른 도메인에서 온 이미지 쌍을 처리할 수 있도록 설계되었습니다. 이 프레임워크는 Expectation-Maximization(EM)과 유사한 두 단계의 교차 최적화 과정을 통해 기하학적 격차와 모달리티 격차를 각각 다룹니다.

- **Technical Details**: AltO는 Geometry loss와 Modality loss를 사용하여 각 기하학적 및 모달리티 격차를 감소시키는 방법을 채택합니다. Geometry loss는 피쳐 수준에서 이미지를 유사하게 만들어 기하학적 격차를 줄이며, Modality loss는 서로 다른 모달리티를 가진 이미지 쌍을 동일한 피쳐 공간으로 매핑하는 역할을 합니다. 이러한 교차 최적화 과정을 통해 ground-truth 데이터 없이도 효과성을 확보할 수 있습니다.

- **Performance Highlights**: AltO는 기존의 비지도 학습 방법들과 비교했을 때 정량적 및 정성적으로 뛰어난 성능을 보여주었으며, 특히 다중 모달 조건에서도 높은 정확성을 유지합니다. 이 방법은 호모그래피 추정기 아키텍처와의 호환성이 뛰어나며, 추가적인 정렬 네트워크와의 결합을 통해 성능을 더욱 향상시킬 수 있습니다.



### "It was 80% me, 20% AI": Seeking Authenticity in Co-Writing with Large Language Models (https://arxiv.org/abs/2411.13032)
- **What's New**: 이번 연구는 인공지능(AI) 도구와 함께 작업하는 작가들이 자신의 진정한 목소리를 보존하고자 하는 방법에 대한 심도 있는 분석을 제공합니다. 특히, 개인화된 AI 지원 도구가 작가의 성장과 진정성을 지원하는 데 중요한 역할을 할 수 있음을 강조합니다. 기존 연구에서 다루지 않았던 작가 중심의 진정성 개념을 탐구하여, AI와의 협동 작업이作品의 진정성에 미치는 영향을 평가합니다.

- **Technical Details**: 연구는 19명의 전문 작가와의 반구조화된 인터뷰를 통해 진행되었으며, 이 과정에서 작가들은 비개인화된 AI 도구와 개인화된 AI 도구로 공동 작성을 수행했습니다. 그 결과로 얻어진 데이터는 30명의 독자들로부터 수집한 의견과 함께 분석되었으며, AI 도구와 협력 시 진정성이 어떻게 인식되는지를 탐구합니다. 작가들의 내부 경험을 중심으로 진정성 개념을 정의하고, AI 도구가 이러한 진정성 개념을 지원하는 방안을 제시합니다.

- **Performance Highlights**: 연구 결과, 독자들은 AI 도움을 받은 저작물에 대해 긍정적인 태도를 보였으며, AI와의 공동 저작을 통해 작가들이 새롭게 기술을 활용하는 것에 대한 관심을 드러냈습니다. 또한, 작가들은 AI와의 공동 저작이 그들의 진정한 목소리를 보존하는 데 기여할 수 있다고 인식하면서도, 이러한 도구가 작가의 표현 방법을 전 좀 더 나아가야 한다고 주장했습니다. 이 연구는 AI 작문 도구의 설계 및 사용에 대한 중요한 시사점을 제공합니다.



### Training Physics-Driven Deep Learning Reconstruction without Raw Data Access for Equitable Fast MRI (https://arxiv.org/abs/2411.13022)
- **What's New**: 이 논문에서는 Compressibility-inspired Unsupervised Learning via Parallel Imaging Fidelity (CUPID)라는 새로운 방법을 제안하여, 고속 자기공명영상(MRI) 재건을 위한 PD-DL 훈련을 가능하게 합니다. CUPID는 전문 MRI 센터 외에서 사용할 수 없는 원시 k-공간 데이터에 대한 접근성을 필요로 하지 않으며, 임상에서 생성된 재구성 이미지만을 사용합니다.

- **Technical Details**: CUPID는 DICOM 형식으로 획득된 임상 이미지에서 훈련되며, Noise 및 Aliasing 아티팩트가 포함될 수 있는 저어 샘플링된 데이터를 활용합니다. 이 방법은 Parallel Imaging 알고리즘과의 일관성을 보장하고, 출력 이미지의 Compressibility를 평가함으로써 unsupervised 조건에서 훈련을 수행합니다.

- **Performance Highlights**: CUPID는 기존의 감시(supervised) 및 자기감시(self-supervised) PD-DL 훈련 전략과 비교할 때 동등한 결과를 달성하며, 전통적인 압축 감지(compressed sensing, CS) 기법 및 최신 생성 모델보다 뛰어난 성능을 보여줍니다. 이 방법은 농촌 및 소외 지역에서도 고속 MRI 접근성을 높이는 데 기여할 수 있습니다.



### Evaluating LLMs Capabilities Towards Understanding Social Dynamics (https://arxiv.org/abs/2411.13008)
Comments:
          To appear in ASONAM 24 proceedings

- **What's New**: 이번 연구는 Generative Models(생성 모델)인 Llama와 ChatGPT가 소셜 미디어의 동적 언어 이해에 대해 얼마나 잘 작동하는지를 분석합니다. 특히 사이버 불링(cyberbullying)과 반사이버 불링(anti-cyberbullying) 메시지의 이해 능력을 평가합니다. 이 연구는 LLM(대형 언어 모델)이 소셜 상호작용의 방향성(directionality) 및 언어 이해에 있어 강점과 약점을 보여준다는 것을 발견했습니다.

- **Technical Details**: 연구의 평가 프레임워크는 LLMs의 다양한 기능 개선 기법을 고려합니다. 세 가지 주요 수준의 상호작용 – 언어, 방향성 및 사이버불링/반사이버불링 메시지의 탐지 – 를 이해하기 위해 LLM의 성능을 비교·분석하였습니다. 또한, LLM의 세부 조정(fine-tuning) 및 프롬프트 엔지니어링(prompt engineering)이 특정 작업에서 긍정적인 영향을 미칠 수 있음을 나타냅니다.

- **Performance Highlights**: 사이비 불링 이해와 방향성 추적에서 LLM들은 일부 매우 긍정적인 결과를 보였으나, 적절한 패러프레이징(paraphrasing) 및 사이버 불링 탐지 측면에서는 혼합된 결과를 나타냈습니다. 궁극적으로, 이러한 연구 결과는 LLM의 발전뿐만 아니라 디지털 환경에서 인간 상호작용의 이해에 기여할 수 있는 기초 자료로 작용할 것입니다.



### Automating Sonologists USG Commands with AI and Voice Interfac (https://arxiv.org/abs/2411.13006)
- **What's New**: 이번 연구는 AI 기반의 첨단 초음파 이미징 시스템을 소개하며, 실시간 이미지 처리(real-time image processing), 장기 추적(organ tracking), 음성 명령(voice commands)을 통해 임상에서의 진단 효율성과 정확성을 향상시키기 위한 시스템입니다. 기존의 초음파 진단은 시간 소모가 크고, 사용자 상호작용에 따라 주관적인 결과가 발생하는 문제가 있었습니다.

- **Technical Details**: 이 시스템은 컴퓨터 비전(computer vision)과 딥러닝(deep learning) 알고리즘을 활용하며, 특히 Detectron2의 Mask R-CNN 모델을 사용해 장기 및 주요 랜드마크의 의미론적 분할(semantic segmentation)을 수행합니다. 자동화된 이미지 프로세싱을 통해 최소한의 인적 개입으로도 고급 정보를 추출할 수 있어 진단 정확성이 향상됩니다. 또한, 음성 인식 기능을 포함하여 사용자가 환자를 관찰하는 동안 손대지 않고 시스템을 조작할 수 있도록 합니다.

- **Performance Highlights**: 간 섬유증 감지를 위해 최적화된 간 조직병리 모듈은 98.6%의 인상적인 정확도를 달성하였습니다. 더불어 장기 분할 모듈은 50%에서 95% 사이의 출력 신뢰 수준(confidence levels)을 제공하여 장기 탐지의 효과성을 보여줍니다. 이러한 성과는 AI 기술이 임상 진단 과정에서 어떤 긍정적인 영향을 미칠 수 있는지를 잘 나타냅니다.



### LaVida Drive: Vision-Text Interaction VLM for Autonomous Driving with Token Selection, Recovery and Enhancemen (https://arxiv.org/abs/2411.12980)
- **What's New**: 최근 비주얼 언어 모델(Visual Language Models, VLMs)의 발전으로 자율주행 분야에서 시각적 질문 응답(Visual Question Answering, VQA)이 중요해졌습니다. 그러나 기존 방법들은 정적 이미지나 비디오에 주로 초점을 맞추며, 동적인 driving 환경에서는 효과적으로 Spatial과 Temporal 정보를 통합하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 우리는 LaVida Drive라는 새롭고 효율적인 VQA 프레임워크를 소개합니다.

- **Technical Details**: LaVida Drive는 시각적 인식의 세밀함을 유지하면서 높은 해상도의 입력을 통합하고, Temporal 데이터를 통합합니다. 이 프레임워크는 두 가지 모듈로 구성됩니다: 
1. Query-aware Token Selection 모듈은 입력 쿼리와의 의미적 정렬에 따라 가장 관련있는 비주얼 토큰을 동적으로 선택합니다.
2. Spatial-Temporal Token Recovery and Enhancement 모듈은 Spatial과 Temporal 정보 간의 원활한 상호작용을 보장하여 컨텍스트 연속성을 유지합니다.

- **Performance Highlights**: LaVida Drive는 다양한 자율주행 VQA 벤치마크에서 뛰어난 성능을 보여주었습니다. 실험 결과, LaVida Drive는 비주얼 토큰을 50%에서 84%까지 감소시켜 인퍼런스 효율성을 향상시키면서도 전반적인 성능을 유지했습니다. 주요 기여로는 높은 해상도의 Spatial 입력에 Temporal 데이터를 매끄럽게 통합한 혁신적 VQA 프레임워크와, 질문 응답을 위한 Key 정보 추출을 구현한 Query-aware Token Selection 메커니즘이 포함됩니다.



### Shrinking POMCP: A Framework for Real-Time UAV Search and Rescu (https://arxiv.org/abs/2411.12967)
Comments:
          Accepted to the The 3rd International Conference on Assured Autonomy

- **What's New**: 본 논문에서는 UAV(무인 항공기) 기반의 수색 및 구조(SAR) 작업을 위한 온라인 경로 계획자의 혁신적인 접근 방식을 소개합니다. 이를 통해 도심 환경에서의 SAR 작업 효율성을 향상시키는데 중점을 두었습니다. 특히, 'Shrinking POMCP'라는 새로운 방법론을 제안하여 제한된 시간 자원 내에서 보다 효과적인 의사 결정을 가능하게 하였습니다.

- **Technical Details**: 이 연구에서 제시된 접근법은 POMDP(부분 관찰 마르코프 결정 프로세스)를 기반으로 하여 UAV의 경로 계획 문제를 형식화합니다. POMCP(부분 관찰 몬테카를로 계획)는 온라인 POMDP 솔버로, 몬테카를로 트리 탐색(MCTS) 알고리즘을 확장하여 POMDP의 복잡성을 관리합니다. 이 방법은 다양한 신뢰도 유형에 대한 성능 평가를 포함하여 3D AirSim-ROS2 시뮬레이터와 2D 시뮬레이터를 통해 실험됩니다.

- **Performance Highlights**: 실험 결과는 병행하여 수행된 기존 방법들과 비교했을 때, 제안한 Shrinking POMCP 솔루션이 검색 시간을 현저히 단축시켰음을 보여줍니다. 이를 통해 UAV 지원 수색 및 구조 작업의 효율성을 높일 수 있는 잠재력을 입증하였습니다. 최적화된 경로 계획과 의사 결정을 통해 SAR 작전의 성공 확률이 크게 향상될 수 있음을 강조합니다.



### Enhancing Thermal MOT: A Novel Box Association Method Leveraging Thermal Identity and Motion Similarity (https://arxiv.org/abs/2411.12943)
Comments:
          Workshop on Towards a Complete Analysis of People, part of the European Conference on Computer Vision (ECCV) 2024

- **What's New**: 이번 논문은 열화상 이미지에서의 다중 객체 추적(MOT) 성능을 개선하기 위한 혁신적인 방법을 소개합니다. 저자들은 열 객체의 특성과 운동 유사성을 결합한 새로운 박스 연관(Box Association) 방법을 개발하였습니다. 이러한 방법은 열 특성의 희소성과 동적 객체 추적을 통합하여 MOT의 정확성과 견고성을 향상시킵니다.

- **Technical Details**: 이 논문은 전통적인 운동 연관(Motion Association) 방법에 의존하는 대신, 두 단계 MOT 모델의 연관 과정에서 열상 정체성(Thermal Identity)를 활용하는 새로운 박스 연관 알고리즘을 제안합니다. 이를 통해 알고리즘은 열 데이터의 특성과 운동 데이터를 함께 적용하여 보다 정확한 박스 연관을 달성합니다. 또한 다양한 도시 환경에서 캡처된 열 및 RGB 이미지의 대규모 데이터셋을 새롭게 구축하여 실험과 평가의 기초 자료로 제공합니다.

- **Performance Highlights**: 저자들은 제안한 방법이 기존 방법들에 비해 우수한 성능을 보이며, 다양한 조건에서 추적의 정확성과 견고성을 크게 향상시킨다고 보고합니다. 새로운 데이터셋과 소스 코드는 공개되어 연구자들이 이 방법을 재현하고 확장할 수 있게 합니다. 또한, 이 연구는 모든 센서 모달리티에 대한 MOT 알고리즘의 연구와 개발을 촉진하는 효과를 기대하고 있습니다.



### Loss-to-Loss Prediction: Scaling Laws for All Datasets (https://arxiv.org/abs/2411.12925)
- **What's New**: 이번 연구는 다른 데이터 분포에서의 손실 예측을 위한 새로운 방법론인 loss-to-loss prediction을 제시합니다. 기존의 scaling laws는 단일 데이터 분포의 훈련 손실 예측에 유용하지만, 이 연구는 다양한 pre-training 데이터셋 간 그리고 모델 간의 손실 관계를 탐구합니다. 또한, 연구 결과는 모델이 훈련된 데이터셋 간의 손실 변환을 통해 scaling laws를 효과적으로 활용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 논문에서는 세 가지 유형의 loss-to-loss 관계를 살펴봅니다. 첫 번째는 train-to-train 관계로, 서로 다른 데이터셋에서 훈련된 모델 간의 손실을 비교합니다. 두 번째는 train-to-test 관계로, 한 데이터셋에서 훈련된 모델의 성능을 다른 데이터셋에서 평가하는 방식입니다. 마지막으로 test-to-test 관계는 서로 다른 데이터셋에서 훈련된 모델의 테스트 손실을 비교합니다. 이 모든 관계에서 shifted power law가 관찰되었습니다.

- **Performance Highlights**: 연구 결과는 다양한 데이터셋과 다운스트림 작업에서 예측 가능한 손실 간의 관계를 입증했습니다. 특히, 적은 수의 샘플로부터 새로운 데이터셋에 대한 만약의 성능을 예측할 수 있는 전이 성능이 향상되었음을 보여줍니다. 또한, 여러 쌍의 데이터셋을 사용하면 새로운 데이터셋에서의 성능 예측이 더 정확해질 수 있음을 발견했습니다.



### Human-In-the-Loop Software Development Agents (https://arxiv.org/abs/2411.12924)
- **What's New**: 최근 LLM(대형 언어 모델)을 기반으로 한 다중 에이전트 패러다임이 소프트웨어 개발 작업을 자동으로 해결하는 데 도입되었습니다. 하지만 기존 연구는 역사적 벤치마크 데이터셋에만 의존하고, 자동화된 소프트웨어 개발 과정의 각 단계에서 인간 피드백을 고려하지 않고 있습니다. 본 논문에서는 HULA(휴먼-인-더-루프 LLM 기반 에이전트) 프레임워크를 도입하여 소프트웨어 개발 엔지니어가 LLM을 개선하고 유도할 수 있도록 합니다.

- **Technical Details**: HULA 프레임워크는 AI 계획 에이전트, AI 코딩 에이전트, 인간 에이전트 등 세 가지 에이전트로 구성됩니다. 이들은 협력하여 JIRA 이슈를 해결하는 공통 목표를 달성합니다. HULA의 효과를 평가하기 위해 세 가지 단계의 평가를 수행하였으며, 이는 벤치마크 데이터셋에 대한 오프라인 평가, 인간 피드백이 포함된 온라인 평가, 그리고 HULA 사용에 대한 실무자의 인식을 조사하는 것입니다.

- **Performance Highlights**: HULA 프레임워크는 총 663개의 실제 JIRA 이슈에 대해 계획을 성공적으로 생성했으며, 이 중 433개의 계획이 실무자에 의해 승인되어 82%의 승인률을 기록했습니다. 코드 작성 단계에서는 376개 이슈 중 95개에 대해 풀 리퀘스트가 생성되었고, 이 중 56개 풀 리퀘스트가 성공적으로 병합되었습니다. 조사 결과 대다수 참가자들은 생성된 코드의 가독성과 수정 용이성에 동의하였으나, 코드가 작업을 완전히 해결하기 위해 인간의 개입이 필요하다고 인식하였습니다.



### A Comparative Study of Text Retrieval Models on DaReCzech (https://arxiv.org/abs/2411.12921)
- **What's New**: 이 논문은 체코어 데이터셋 DaReCzech에서 7개의 상용 문서 검색 모델의 성능을 포괄적으로 평가한 결과를 제시합니다. 모델들은 Splade, Plaid, Plaid-X, SimCSE, Contriever, OpenAI ADA, Gemma2로 구성되며, 체코어 검색 접근 방식의 품질을 추정하는 것이 주요 목적입니다. 실험 결과, Gemma2가 가장 높은 정밀도와 재현율을 보였으며, SPLADE와 PLAID 모델은 효율성과 성능의 균형을 잘 이루었습니다.

- **Technical Details**: 제안된 연구에서는 다양한 정보 검색(IR) 모델들로부터 인덱스 크기, 검색 속도 및 메모리 사용량 등을 분석합니다. 각각의 모델을 체코 문서와 쿼리에 대해 평가하며, 베이스라인 모델인 BM25를 사용하여 다른 모델의 효과를 평가합니다. 모델들이 체코어 데이터셋과 영어 번역에서 어떻게 성능을 발휘하는지 비교 분석하였고, 특히 지수의 크기와 검색 속도가 큰 데이터셋에서의 스케일링에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험을 통해 저자들은 각 모델 간의 성능 차이를 명확히 식별했습니다. Gemma2는 정밀도와 재현율 모두에서 뛰어난 결과를 보였으나, Contriever는 저조한 성능을 나타냈습니다. SPLADE와 PLAID 모델은 효율성과 성능을 모두 고려할 수 있는 좋은 선택으로 밝혀졌습니다. 이번 연구는 체코어의 정보 검색 방법론의 심층 비교를 제공하는 첫 번째 사례로서, 향후 연구에 중요한 이정표가 될 것입니다.



### Enhancing Deep Learning-Driven Multi-Coil MRI Reconstruction via Self-Supervised Denoising (https://arxiv.org/abs/2411.12919)
- **What's New**: 이 논문에서는 자가 감독된 잡음 제거(self-supervised denoising)를 딥러닝 기반 재구성 방법의 전처리 단계로 통합했을 때의 효과를 연구하였습니다. 고무적인 점은, 저소음 참조 데이터의 필요성을 줄이고도 MRI 재구성의 품질을 향상시킬 수 있다는 것입니다. 연구에서는 가우시안 잡음으로 손상된 k-공간 데이터를 사용하여 진단 정확도를 높일 수 있는 가능성을 보여었습니다.

- **Technical Details**: 다양한 MRI 데이터의 특징을 반영하기 위해 일반화된 스타인 비편향 위험 추정(Generalized Stein's Unbiased Risk Estimate, GSURE)을 기반으로 하는 잡음 제거 파이프라인을 개발하였습니다. 이 과정에서 Diffusion Probabilistic Models (DPMs)와 Model-Based Deep Learning (MoDL)을 활용하여 자가 감독적 잡음 제거가 DL 네트워크의 학습 성과에 미치는 영향을 평가하였습니다. 특히, 자기공명영상(MRI) 재구성을 위한 피험자 데이터를 포함하여, 다양한 신호 대 잡음비(signal-to-noise ratio, SNR) 조건에서 성능을 시험했습니다.

- **Performance Highlights**: 실험 결과, 자가 감독된 잡음 제거를 통해 MRI 재구성의 품질과 효율성이 크게 향상됨을 확인하였습니다. 예를 들어, DPM을 사용하는 경우, 잡음이 제거된 이미지를 이용해 DL 네트워크를 훈련했을 때, 정상화된 평균 제곱근 오차(normalized root mean squared error, NRMSE)와 구조적 유사도 지수(structural similarity index measure, SSIM), 피크 신호 대 잡음비(peak signal-to-noise ratio, PSNR)가 모든 SNR 수준에서 유의미하게 개선되었습니다. 이러한 결과는 다양한 조건에서도 자가 감독된 잡음 제거가 DL 기반 MRI 재구성 방법의 유효성을 높일 수 있음을 뒷받침합니다.



### MLDGG: Meta-Learning for Domain Generalization on Graphs (https://arxiv.org/abs/2411.12913)
Comments:
          Accepted in KDD 2025 (research track)

- **What's New**: 본 논문에서는 MLDGG라는 새로운 크로스-멀티 도메인 메타 학습 프레임워크를 제안합니다. 이 프레임워크는 소스 도메인에서 샘플링된 그래프에서 이전 지식을 습득하고, 이러한 지식을 바탕으로 훈련 중 접근할 수 없는 타겟 도메인에 일반화하는 것을 목표로 합니다. 여기에는 구조 학습과 의미 식별을 통해 도메인 간의 적응력을 높이는 방식이 포함되어 있습니다.

- **Technical Details**: MLDGG는 두 가지 주요 구성 요소로 이루어져 있습니다: 구조 학습자(structure learner)와 표현 학습자(representation learner). 구조 학습자는 관련 없는 엣지의 부정적 영향을 줄이며, 다양한 도메인 간의 공유 구조 지식을 포착하여 GNN이 학습한 표현의 종합성을 향상시킵니다. 표현 학습자는 노드 임베딩에서 도메인 불변 의미와 도메인 특화 변동 정보를 분리하여 일반화 능력을 높입니다.

- **Performance Highlights**: 실험 결과, MLDGG는 세 가지 다양한 분포 이동(distribution shift) 환경에서 기존의 최첨단 방법들을 초월하는 성능을 보여주었습니다. 이는 MLDGG가 노드 예측 과제에 대해 제공되는 다양한 도메인에서 강력한 일반화 능력을 유지함을 입증합니다. 이러한 결과는 제안된 메타 학습 접근 방식의 효과성을 강조합니다.



### Advancing Large Language Models for Spatiotemporal and Semantic Association Mining of Similar Environmental Events (https://arxiv.org/abs/2411.12880)
- **What's New**: 이번 논문은 최신 검색 도구에서 필수적인 두 가지 작업인 Retrieval(검색)과 Recommendation(추천)을 개선하기 위한 새로운 Retrieval-Reranking(재정렬) 프레임워크를 소개합니다. 이 프레임워크는 Large Language Models (LLMs)을 활용하여 뉴스 기사와 웹 포스트에 설명된 비정상적인 기후 및 환경 사건을 더 효과적으로 검색하고 추천할 수 있도록 설계되었습니다. 전통적인 수작업 큐레이션 방법의 고비용과 비확장성 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방안은 고급 자연어 처리 (Natural Language Processing) 기법을 사용하여 시공간(spatiotemporal) 사건을 분석하는 최적화된 솔루션을 탐구합니다. 또한, 공간적 근접성(spatial proximity), 시간적 연관성(temporal association), 의미 유사성(semantic similarity), 카테고리 기반 유사성(category-instructed similarity)을 포함한 다각적 기준을 통합한 Geo-Time Re-ranking (GT-R) 전략을 제안합니다. 이 모델은 4000개 Local Environmental Observer (LEO) Network 사건 데이터셋에 적용되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 다수의 최신 Dense Retrieval 모델들 사이에서 비슷한 사건 추천에 있어 최고의 성능을 달성했습니다. 이 검색 및 추천 파이프라인은 지리적(geospatial) 및 시간적(temporal) 데이터와 관련된 다양한 검색 작업에도 적용될 수 있습니다. 우리는 관련 사건을 연결함으로써 대중이 기후 변화와 그로 인한 다양한 커뮤니티에 미치는 영향을 더 잘 이해할 수 있도록 지원할 수 있기를 희망합니다.



### The Illusion of Empathy: How AI Chatbots Shape Conversation Perception (https://arxiv.org/abs/2411.12877)
- **What's New**: 이 연구는 AI 챗봇의 공감을 탐구하며, 사용자가 인식하는 챗봇의 정체성과 공감이 대화 품질에 미치는 영향을 검토합니다. GPT 기반 챗봇이 대화 품질에서 더 높은 평가를 받았지만, 사용자는 여전히 이들을 인간보다 덜 공감적이라고 평가한다는 점이 주목할 만합니다. 연구 결과는 챗봇의 공감 표현과 사용자가 느끼는 대화 만족도가 상호작용에서 중요하다는 점을 강조합니다.

- **Technical Details**: 연구에는 GPT-4o와 같은 대규모 언어 모델(LLM)과 155개의 대화가 포함되어 있으며, 챗봇과 인간 간의 대화에서의 공감 표현을 분석하여 이들 간의 차이를 규명하고자 했습니다. 공감 모델은 인간 간 대화에서 훈련된 5개 모델 중 3개에서 챗봇과 인간 간의 공감 언어에서 유의미한 차이를 발견하지 못했습니다. 이러한 결과는 사용자가 공감적 언어를 해석하는 방식에 대한 통찰을 제공합니다.

- **Performance Highlights**: 연구 결과, 챗봇은 전체 대화 품질에서 높은 평가를 받았지만 공감 수준은 인간 파트너보다 낮았다는 점이 두드러집니다. 또한, 사용자 자기보고와 LLM의 주석이 일치하여 챗봇의 공감 부족을 강화합니다. 이러한 발견은 감정 언어의 삽입 이상의 접근이 필요함을 강조하며, 사용자 경험에서의 복잡한 공감 해석을 반영해야 함을 시사합니다.



### Puppet-CNN: Input-Adaptive Convolutional Neural Networks with Model Compression using Ordinary Differential Equation (https://arxiv.org/abs/2411.12876)
- **What's New**: 이 논문에서는 CNN의 전통적인 한계를 극복하기 위해 새로운 CNN 프레임워크인 Puppet-CNN을 제안합니다. 이 프레임워크는 puppet 모듈과 puppeteer 모듈의 두 가지 모듈로 구성되어 있으며, puppeteer 모듈이 입력 데이터의 복잡성에 따라 커널을 생성합니다. 이는 네트워크의 깊이에 따라 동적으로 커널 파라미터를 조정하여 효율성과 성능을 획기적으로 개선합니다.

- **Technical Details**: Puppet-CNN의 핵심은 Ordinary Differential Equation (ODE)을 사용하여 puppeteer 모듈이 puppet 모듈의 커널 파라미터를 생성하는 것입니다. 이렇게 하면 더 많은 파라미터를 저장할 필요 없이 작고 효율적인 puppeteer ODE 모듈을 통해 깊은 puppet 네트워크를 최적화할 수 있습니다. 다양한 데이터 샘플에 대해 커널과 깊이를 다르게 설정할 수 있어 유연성이 높아지는 장점이 있습니다.

- **Performance Highlights**: 실험 결과, Puppet-CNN은 전통적인 CNN보다 성능과 효율성 모두에서 우수한 결과를 보여줍니다. 모델 크기는 10배 이상 줄일 수 있으며, 이는 다양한 데이터 복잡성을 처리하는 데 있어 훨씬 효과적인 접근법을 제공합니다. 이는 특히 많은 파라미터를 요구하는 전통적인 CNN의 한계를 극복하는 데 도움을 줍니다.



### From Text to Pose to Image: Improving Diffusion Model Control and Quality (https://arxiv.org/abs/2411.12872)
Comments:
          Published at the NeurIPS 2024 Workshop on Compositional Learning: Perspectives, Methods, and Paths Forward

- **What's New**: 최근 2년 동안 텍스트-투-이미지(diffusion models)를 이용한 이미지 생성 모델의 품질과 사용이 증가함에 따라, 출력 제어의 필요성이 대두되고 있습니다. 본 논문에서는 텍스트-투-포즈(text-to-pose, T2P) 생성 모델과 새로운 샘플링 알고리즘을 통해 이미지 생성의 두 가지 주요 문제를 해결했습니다. 특히, 본 기술은 인체 포즈의 높은 충실도를 위한 새로운 포즈 어댑터를 포함하고 있어, 텍스트-투-포즈-투-이미지(generative text-to-pose-to-image) 프레임워크를 가능하게 합니다.

- **Technical Details**: T2P 모델에서는 인체 포즈를 18개의 신체 부위, 42개의 손, 68개의 얼굴 포인트로 설명하고, CLIP을 기반으로한 메트릭을 통해 생성된 포즈의 품질을 평가합니다. 포즈 생성을 위해 텍스트 특징을 조건으로 하는 오토리그레시브(decoder-only transformer architecture)를 활용하였으며, 가우시안 혼합 모델(GMM)을 사용하여 포즈의 연속성을 학습합니다. 실험에서는 4M (포즈, 프롬프트) 쌍을 학습 데이터로 활용하여 T2P 모델의 성능을 검증하였습니다.

- **Performance Highlights**: T2P 모델은 COCO-Pose 벤치마크 데이터셋에서 KNN 방식보다 78%의 높은 성능을 보여주며, 텍스트 프롬프트에 대한 정렬 능력을 입증했습니다. 또한 새로운 포즈 어댑터는 얼굴과 손에 대한 포인트를 포함하여 기존 SOTA(SDXL-Tencent) 모델보다 향상된 포즈 정확도를 보여주었습니다. 그러나 이미지 품질은 여전히 기본 SDXL 모델에는 미치지 못하는 결과를 나타냈습니다.



### mDAE : modified Denoising AutoEncoder for missing data imputation (https://arxiv.org/abs/2411.12847)
- **What's New**: 이번 논문에서는 Denoising AutoEncoder (DAE)를 기반으로 한 새로운 기법인 mDAE를 제안합니다. 이 방법은 손실 함수(loss function)의 수정과 하이퍼파라미터(hyper-parameter) 선택 절차를 간소화하여 결측치(imputation) 문제를 해결합니다. 여러 UCI Machine Learning Repository 데이터셋에 대한 연구를 통해 mDAE 방법이 Root Mean Squared Error (RMSE) 측면에서 향상된 성능을 보여주는 것을 확인했습니다.

- **Technical Details**: mDAE는 결측치를 수치적 데이터에서 보완하는 데 특화된 Denoising AutoEncoder의 수정형입니다. AutoEncoder는 레이블이 없는 데이터를 효율적으로 표현하고 복원하는 인공 신경망입니다. 원래 DAE는 노이즈가 있는 데이터로부터 원본 데이터를 회복하도록 설계되었으며, 본 논문에서는 결측치를 노이즈로 간주하여 이를 처리하는 방식을 제안합니다.

- **Performance Highlights**: 제안된 mDAE 방법은 8가지 다른 imputation 방법과 비교했을 때, Mean Distance to Best (MDB) 기준에서 지속적으로 상위에 위치하였으며, 특히 SoftImput과 missForest와 함께 최고 방법으로 평가되었습니다. 이 연구를 통해 mDAE가 주어진 데이터셋에서 효과적인 결측치 보완 방법임을 입증했습니다. 결과는 GitHub를 통해 재현 가능하고 다른 데이터셋과 방법으로 일반화 가능합니다.



### Reward Modeling with Ordinal Feedback: Wisdom of the Crowd (https://arxiv.org/abs/2411.12843)
- **What's New**: 본 논문에서는 인간의 선호도를 반영하여 보상 모델을 학습하는 새로운 프레임워크를 제안합니다. 기존의 이진 피드백을 넘어, 다양한 수준의 순서형 피드백(ordinal feedback)을 활용하여 데이터로부터 중요한 정보를 더 효과적으로 추출할 수 있습니다. 특히 이 연구는 선호 간의 미세한 차이를 반영하는 방법을 제시합니다.

- **Technical Details**: 제안된 프레임워크에서는 인간의 선호 피드백을 특정 응답이 다른 응답보다 더 낫다는 확률과 연결짓습니다. 우리는 'wisdom of the crowd'라는 사회학적 개념을 기반으로 한 마진 편향 제로(marginal unbiasedness) 조건을 도입하여 이론적 근거를 마련합니다. 또한, 우리는 이 순서형 피드백 모델의 통계적 이점을 검증하며, Rademacher 복잡성을 줄일 수 있는 방법을 제시합니다.

- **Performance Highlights**: 수치 실험의 결과, 미세한 피드백을 포함한 경우 보상 모델 학습이 개선된다는 것을 확인하였습니다. 4개의 서로 다른 순서형 피드백 시스템을 설정하여 이론적 발견을 검증하였고, 특히 특정 비율의 동점(tied) 샘플을 포함하는 것이 RM 학습을 촉진하는 것으로 나타났습니다. 이러한 결과는 보상 모델의 성능을 크게 향상시킬 가능성을 보여줍니다.



### Efficient Medicinal Image Transmission and Resolution Enhancement via GAN (https://arxiv.org/abs/2411.12833)
- **What's New**: 본 논문은 X-ray 이미지를 위한 새로운 접근법을 제시합니다. Real-ESRGAN의 최적화된 네트워크 전송을 통해 이미지 품질을 향상시키고, 서버 부담 및 전송 대역폭을 줄이는 방법을 소개합니다. 특히 B/W X-ray 이미지의 노이즈와 해상도 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 X-ray 이미지를 낮은 해상도로 전처리한 후, 최종 수신자가 Real-ESRGAN을 사용하여 이미지를 업스케일링합니다. Residual-in-Residual Dense Blocks (RRDB)와 지각(Perceptual) 및 적대적 손실(Adversarial Loss) 함수를 통합하여 고품질의 저노이즈 이미지를 얻습니다. 이 과정은 전송 품질에 관계없이 주요 진단 정보를 보존하는 것을 목표로 합니다.

- **Performance Highlights**: 비교 평가는 제안된 방법이 기존의 CNN 기반 및 ESRGAN 모델보다 우수한 노이즈 감소 및 세부 정보 선명도를 제공함을 보여주었습니다. Peak Signal-to-Noise Ratio (PSNR) 및 Structural Similarity Index (SSIM)와 같은 계량적 지표와 정성적 평가 모두 이러한 이점을 확인했으며, Real-ESRGAN의 활용 가능성을 보여주고 있습니다.



### Probing the Capacity of Language Model Agents to Operationalize Disparate Experiential Context Despite Distraction (https://arxiv.org/abs/2411.12828)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM) 에이전트의 새로운 연구 결과를 제시하고 있습니다. 특히 OEDD(Operationalize Experience Despite Distraction)라는 새로운 데이터셋을 소개하며, 이는 다양한 경험 정보를 바탕으로 distractor가 있는 상황에서 에이전트가 의사 결정을 내리는 과정을 평가합니다. 이는 LLM의 경험 기반 추론 능력을 향상시키기 위한 기초 자료로 사용될 수 있습니다.

- **Technical Details**: OEDD 데이터셋은 인간 주석자에 의해 검증된 시나리오로 구성되어 있으며, 에이전트는 주어진 입력 프롬프트에서 여러 개의 경험적 정보를 바탕으로 분석해야 합니다. 연구에서는 최첨단 LLM인 GPT-3.5 Turbo, GPT-4o 및 Gemini 1.5 Pro를 평가하고, 최소한의 체인 오브 씽킹(chains of thought) 프롬프트 전략을 적용하였습니다. 평가 과정에서는 1,615개 이상의 토큰을 포함하는 입력 컨텍스트와 두 개의 상반된 환경 전제에서 결정적인 결론을 도출해야 하며, 이후에 주의 분산 요소인 trivial한 정보를 갖는 distractor가 등장합니다.

- **Performance Highlights**: 이 실험 결과, 모든 LLM들은 두 가지 행동 중 더 나은 선택을 할 때 무작위 선택보다 성능이 떨어지는 것으로 나타났습니다. 이로 인해, 복잡한 경험적 정보가 존재하는 상황에서 LLM의 결정 능력을 향상시킬 필요가 확인되었습니다. 연구팀은 코드와 테스트 데이터셋을 공개하여, 다른 연구자들이 이를 활용할 수 있도록 하고 있습니다.



### Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models (https://arxiv.org/abs/2411.12790)
- **What's New**: 이번 논문에서는 이미지 내 다수의 상호작용하는 개체들에 대한 정밀한 지식 수정을 목표로 하는 새로운 비주얼 기반의 세분화된 지식 편집 과제를 제안합니다. 기존의 multimodal knowledge editing 방법들이 주로 텍스트 중심, 조잡한 시나리오에 집중했던 반면, 우리는 Fine-Grained Visual Knowledge Editing (FGVEdit) 벤치마크를 통해 이 과제를 평가합니다. 이로써 기존 방법들이 갖는 한계를 극복하고 보다 정교한 지식 수정을 가능하게 하는 접근 방식이 마련되었습니다.

- **Technical Details**: 우리는 Multimodal Scope Classifier-based Knowledge Editor (MSCKE) 프레임워크를 제안하며, 이는 비주얼 및 텍스트 정보를 통합하는 멀티모달 범위 분류기를 활용하여 이미지 내 특정 개체와 관련된 지식을 정확하게 식별하고 업데이트합니다. 이를 위해 기존의 텍스트 기반 편집 방법 SERAC을 바탕으로 하여 텍스트 전용 범위 분류기를 멀티모달 범위 분류기로 교체했습니다. 이 접근 방식은 정밀한 지식 수정을 보장하며 관계가 없는 정보는 보존하는 데 중점을 두고 있습니다.

- **Performance Highlights**: FGVEdit 벤치마크에 대한 광범위한 실험 결과, MSCKE가 기존 방법들보다 뛰어난 성능을 보여주었습니다. 특히, 이미지 내 상호작용하는 다중 개체에 대한 복잡한 지식 편집 도전 과제를 해결하는 데 있어 MSCKE의 유효성이 입증되었습니다. 이러한 성과는 멀티모달 지식 편집의 정확성과 관련성을 높이는 데 기여하고 있습니다.



### Visual Cue Enhancement and Dual Low-Rank Adaptation for Efficient Visual Instruction Fine-Tuning (https://arxiv.org/abs/2411.12787)
- **What's New**: 이 논문에서는 새로운 접근법인 Vision Cue Enhancement (VCE)와 Dual Low-Rank Adaptation (Dual-LoRA)을 제안하여 다중 모달 대형 언어 모델(MLLMs)의 미세 조정 프레임워크를 개선합니다. VCE는 고수준 시각적 특징에 의존하는 기존 방법의 한계를 극복하기 위해 다층 시각적 단서를 통합하여 모델의 세밀한 시각적 정보 포착 능력을 향상시키고 있습니다. Dual-LoRA는 기술적 및 작업적 학습 공간을 분리하여 다양한 작업 간의 효율적인 조정을 목표로 합니다.

- **Technical Details**: 저자들은 Vision Cue Enhancement (VCE) 모듈을 통해 다층 시각적 특성 맵을 활용하여 각 패치를 개선합니다. 이를 통해 더 지역적인 시각적 특성이 캡처되어 시각적 이해력이 개선됩니다. 또한, Dual Low-Rank Adaptation (Dual-LoRA)은 특정 기술 지식을 학습하는 기술 저차원 공간과 이전 지식을 활성화하는 작업 활성화 저차원 공간을 결합하여 데이터 충돌을 완화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다운스트림 작업 및 일반 MLLM 벤치마크에서 우수한 성능을 나타냅니다. VCE 모듈은 고급 비주얼 기능을 더욱 풍부하게 만들어 다중 작업에서의 효과적인 비주얼 지침 미세 조정을 가능하게 하며, Dual-LoRA를 통해 각 작업에 맞는 높은 세부 사항을 유지하면서도 데이터 충돌 문제를 해결합니다.



### Lucia: A Temporal Computing Platform for Contextual Intelligenc (https://arxiv.org/abs/2411.12778)
- **What's New**: 이 논문은 기계 인지의 새로운 길을 여는 Temporal Computing 플랫폼인 Lucia를 소개합니다. Lucia는 사용자의 일상 활동을 연속적으로 기록하고 해석해, 사용자 맞춤형 기억력을 구축하는 오픈 소스 장치입니다. 이 플랫폼은 가볍고 착용하기 편리하며, 기존의 기기들과 비교하여 모든 날 동안 데이터를 접근할 수 있는 실시간 기능을 강조합니다. 또한, 이 기술은 인공지능의 발전을 통해 인간의 인지 능력을 향상시키려는 목표를 가지고 있습니다.

- **Technical Details**: Lucia는 다양한 센서들이 통합된 첨단 웨어러블 플랫폼으로, 다중 모드 데이터를 캡처하는 데 중점을 둡니다. 이 장치는 12MP 해상도의 카메라, 저전력 Inertial Measurement Unit (IMU), 두 개의 마이크로폰 배열, 그리고 근접 센서를 포함하고 있습니다. 이러한 센서들은 정밀하게 보정되어 있으며, 실시간 데이터 액세스를 통해 Temporal Computing의 요구를 충족시키는데 필수적입니다. 또한, 이 장치는 사용자의 편안함을 고려하여 44그램의 경량 설계를 갖추고 있습니다.

- **Performance Highlights**: Lucia는 고급 AI 모델과 Temporal Computing 원칙을 활용해 시간적 경험을 기반으로 사용자에게 통찰력을 제공하는 장치입니다. 이 장치는 기본적으로 사용자의 활동을 기록하고, 기억 회상, 의사 결정을 향상시키며 개인 생산성을 높이는 기능을 갖추고 있습니다. 모든 날 동안 착용 가능하고 인지 적응을 통해 개인화된 지원을 제공함으로써 디지털 세계와의 상호 작용을 대폭 개선합니다. 이 프로젝트는 연구 기관과 협력하여 개인화된 AI 기술의 혁신을 촉진하고 있습니다.



### Revisiting Fake News Detection: Towards Temporality-aware Evaluation by Leveraging Engagement Earliness (https://arxiv.org/abs/2411.12775)
Comments:
          WSDM 2025

- **What's New**: 사회적 그래프 기반의 가짜 뉴스 탐지에서 모델이 과거 정보만을 사용하여 훈련되는 현실적인 평가 방식을 제안합니다. 기존 접근 방식들은 미래 데이터에 접근이 가능하여 발생하는 정보 유출로 인해 평가가 비현실적임을 지적하고, 이에 따라 ‘DAWN’이라는 새로운 방법을 개발하였습니다. 이 방법은 사회적 네트워크에서의 사용자 참여 초기 시점을 활용하여 가짜 뉴스 탐지 성능을 향상시킵니다.

- **Technical Details**: DAWN는 그래프 구조적 학습(Graph Structure Learning, GSL) 프레임워크를 사용하고, 사용자와 트윗의 참여 초기 시점에 기반한 특성 표현을 통해 노이즈 엣지의 가중치를 낮추는 방식으로 작동합니다. 이를 통해 더 정확한 구별 능력을 발휘할 수 있도록 설계되었습니다. 또한, 본 연구에서는 참여의 빠름(earliness)과 뉴스의 진위(label consistency)와의 관계를 심층적으로 분석하여, 정보의 진실성을 연결하는 중요한 지표를 제시하였습니다.

- **Performance Highlights**: DAWN은 두 가지 실제 가짜 뉴스 데이터셋에서 기존 방법들에 비해 높은 성능을 기록하였으며, 특히 ‘temporality-aware’ 훈련과 평가 설정 하에서도 튼튼한 성능을 보였습니다. 실험 결과, DAWN은 가짜 뉴스 탐지 분야에서 향상된 성능을 나타내어 강인성을 증명하였습니다. 실험을 통해 사용자 참여 초기정보를 활용하는 것이 가짜 뉴스 탐지에서 얼마나 효과적인지 확인하였습니다.



### CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization (https://arxiv.org/abs/2411.12768)
- **What's New**: 최근 연구 결과에 따르면, Large Language Models (LLMs)는 '백도어 공격(backdoor attacks)'에 취약하다는 것이 밝혀졌습니다. 이러한 공격은 적대자가 숨겨진 트리거를 삽입하여 모델의 응답을 조작하는 방식으로 이루어집니다. 본 논문에서는 Internal Consistency Regularization (CROW)라는 새로운 방어 기법을 소개하며, 이는 모델 훈련 중 일관성을 촉진하여 백도어 트리거로 인한 레이어 간 불일치를 해결합니다.

- **Technical Details**: CROW는 적대적 섭동(adversarial perturbations)과 정규화를 통해 내부 일관성(internal consistency)을 강화하여 백도어의 영향을 중화합니다. 이 방식은 클린 데이터(clean data) 세트만을 기반으로 하며, 클린 참조 모델이나 사전 트리거 지식이 필요하지 않아 다양한 LLM 아키텍처에서의 실용성을 높입니다. 또한, CROW는 레이어 간의 일관성을 정의하고 백도어가 이 일관성을 어떻게 교란시키는지를 명확히 하여 새로운 방어 기법의 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, CROW는 Llama-2, CodeLlama, Mistral-7B 등 다양한 모델에서 공격 성공률(ASR)을 일관되게 감소시키는 것으로 나타났습니다. 기존 방어 기법인 파인튜닝(finetuning), 가지치기(pruning), 양자화(quantization)와 비교할 때 성능이 월등히 개선되었으며, 모델의 생성 능력을 유지합니다. CROW는 100개의 클린 샘플로 4분 이내에 훈련을 완료할 수 있어 계산적으로도 효율적입니다.



### Suicide Risk Assessment on Social Media with Semi-Supervised Learning (https://arxiv.org/abs/2411.12767)
Comments:
          Accepted for publication in the 2024 IEEE International Conference on Big Data

- **What's New**: 이 논문에서는 자살 위험 평가를 위한 새로운 반지도 학습 프레임워크를 제안하고 있습니다. 기존 연구의 한계를 극복하기 위해 500개의 라벨이 있는 데이터와 1,500개의 라벨이 없는 데이터를 활용하며, 데이터의 불균형 문제를 해결하기 위해 새로운 pseudo-label 획득 과정을 도입했습니다. 또한, RoBERTa 모델을 기본 구조로 사용하여 성능을 극대화했습니다.

- **Technical Details**: 연구에서 사용되는 데이터는 Reddit의 r/SuicideWatch 서브레딧에서 수집된 500개의 라벨이 있는 사례와 1,500개의 라벨이 없는 사례로 구성됩니다. 데이터셋은 네 가지 자살 위험 수준으로 분류되며, 반지도 학습(SSL) 알고리즘이 적용됩니다. 특히, Stratified Confidence Sampling (SCS) 알고리즘을 사용하여 자살 위험의 낮은 클래스에 대한 pseudo-label 정확성을 보장합니다.

- **Performance Highlights**: 반지도 학습 방법론을 통해, 라벨이 없는 데이터를 활용하여 자살 위험 수준을 식별하는 데 있어 기존의 감독형 접근법보다 향상된 예측 성능을 입증했습니다. 모델의 성능은 다양한 심리적 표현의 정도를 식별하는 데 매우 효과적이며, 자살 예방을 위한 자동화된 시스템 구축에 기여할 것으로 기대됩니다.



### SEFD: Semantic-Enhanced Framework for Detecting LLM-Generated Tex (https://arxiv.org/abs/2411.12764)
- **What's New**: 본 논문은 대용량 언어 모델(LLM)이 생성한 텍스트를 탐지하기 위한 새로운 접근법인 개선된 의미 기반 프레임워크(SEFD)를 소개합니다. 이 프레임워크는 기존 탐지 방법론에 비해 더 robust한 성능을 제공하며, 특히 paraphrasing 기법을 통한 텍스트 변형에 효과적으로 대응합니다. 텍스트의 의미를 충분히 활용하기 위해 검색 기반 메커니즘을 통합한 것이 특징입니다.

- **Technical Details**: SEFD 프레임워크는 초기 탐지 단계, 의미 유사성 계산, 의미가 강화된 탐지의 세 가지 주요 단계를 포함합니다. 초기 탐지를 위해 기존 탐지기를 사용하고, BERT 기반 모델을 통해 검색 풀 내 텍스트와의 의미 유사도를 평가합니다. 마지막으로, 탐지 점수와 유사성 점수를 통합하여 최종 점수를 도출하며, 이 과정에서 검색 풀도 지속적으로 업데이트됩니다.

- **Performance Highlights**: 본 연구에서는 SEFD 프레임워크가 실제 환경에서 흔히 발생할 수 있는 순차 텍스트 시나리오에서의 효율성을 입증합니다. 다양한 LLM 생성 텍스트와 탐지 방법을 사용한 실험을 통해, paraphrasing 상황에서도 탐지 정확성이 크게 향상됨을 보여주었습니다. 이로 인해 데이터의 무결성을 보장하고 신뢰할 수 있는 정보 제공에 기여할 것으로 기대됩니다.



### Education in the Era of Neurosymbolic AI (https://arxiv.org/abs/2411.12763)
- **What's New**: 이 논문에서는 신경상징 인공지능(Neurosymbolic AI, NAI)이 교육 분야에 혁신을 가져올 수 있는 가능성을 탐구하고 있습니다. NAI를 활용함으로써 기존의 교육 기술들이 가지는 개인화된 학습 경험의 한계를 극복하고, 학습자의 이해도를 세밀하게 진단하여 맞춤형 교육 콘텐츠를 제공할 수 있다고 주장합니다. 특히, 교육적 에이전트(pedagogical agents)를 활용한 하이브리드 NAI 아키텍처가 중요하다고 강조됩니다.

- **Technical Details**: NAI는 상징적 인공지능과 뉴럴 시스템의 강점을 결합한 하이브리드 접근 방식으로, 복잡한 인간 개념의 이해를 돕습니다. 이 시스템은 지식 그래프(knowledge graphs)와 대규모 언어 모델(large language models, LLMs)을 통합하여 교육적 맥락에서 정보의 정확한 구성과 관계를 인식하게 합니다. 교육적 에이전트(PAs)는 이러한 NAI 구조에서 중요한 구성 요소로, 학습자의 반응을 동적으로 인식하고 개인화된 커리큘럼을 제공하는 데 핵심적인 역할을 할 것입니다.

- **Performance Highlights**: NAI 시스템은 학습자의 성과와 지식의 격차를 인식하여 맞춤형 교육을 가능하게 할 것입니다. 이를 통해 다양한 학습자의 요구에 맞춘 적응형 콘텐츠를 제공하며, 특수 교육과 학습 장애에 대한 낙인을 줄이는 데 기여할 수 있습니다. 결론적으로, NAI 시대의 교육은 접근성과 형평성을 향상시키고, 실제 세계 기술과의 정합성을 높이는 데 중요한 역할을 할 것으로 기대됩니다.



### Playing Language Game with LLMs Leads to Jailbreaking (https://arxiv.org/abs/2411.12762)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)의 안전성을 우회하는 새로운 jailbreak 공격 방법을 제안합니다. 연구자들은 mismatched generalization(불일치 일반화)이라는 개념을 기반으로 한 자연어 게임(natural language games)과 맞춤형 언어 게임(custom language games)이라는 두 가지 혁신적인 방법을 개발했습니다. 이 방법들은 LLM의 안전 메커니즘을 효과적으로 우회할 수 있어, 높은 공격 성공률을 기록했습니다.

- **Technical Details**: 제안된 jailbreak 방법은 자연어를 변형하여 공격자가 모델의 안전 제어를 피할 수 있도록 하는 언어 게임을 활용합니다. 특정 규칙을 적용하여 입력을 변형함으로써, LLM이 해로운 요청을 악의적인 것으로 인식하지 못하게 합니다. 연구에서는 GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet에 대한 실험을 통해 각각 93%, 89%, 83%의 성공률을 기록했습니다.

- **Performance Highlights**: 실험 결과, 제안된 자연어 게임 및 맞춤형 언어 게임이 LLM의 안전 정렬을 효과적으로 우회할 수 있다는 것을 입증했습니다. 또한, Llama-3.1-70B 모델의 미세 조정을 통해 상관된 안전 정렬이 다른 도메인에서 효과적으로 일반화되지 못하는 한계도 발견했습니다. 이는 현재의 안전 정렬 기법이 LLM에 내재된 안전 지식을 효과적으로 일반화하지 못하고 있음을 시사합니다.



### AI-Empowered Human Research Integrating Brain Science and Social Sciences Insights (https://arxiv.org/abs/2411.12761)
Comments:
          Accepted to IEIR 2024, 10 pages, 4 figures

- **What's New**: 이 논문은 인공지능(AI)이 과학 연구, 특히 뇌 과학과 사회 과학 분야에서 어떻게 혁신적으로 역할을 수행하는지 탐구합니다. 저자들은 인간 연구의 기본 요소를 분석하고, 인공지능과의 협력 연구로의 전환이 시급하다고 주장합니다. 특히 AI가 연구 도구에서 연구 참여자로 변화하는 과정과 이와 관련된 새로운 연구 패러다임을 제안하고 있습니다.

- **Technical Details**: 저자들은 AI-Brain Science Research Paradigm과 AI-Social Sciences Research Paradigm이라는 두 가지 혁신적인 연구 패러다임을 제안합니다. 이러한 패러다임 아래에서는 AI가 연구 도구(ART), 연구 보조자(ARA), 연구 참여자(ARP)로서 활동합니다. 또한 이 논문에서는 인간-AI 공동 연구의 수행 방법을 제시하였으며, 실증 연구와 설문 조사를 통해 AI가 연구 과정에서 창의적이고 비판적 사고에 미치는 영향을 평가할 것을 권장합니다.

- **Performance Highlights**: 이 연구의 주요 기여는 인간 연구 과정에서의 인지 및 사회적 상호작용의 기초 요소를 재검토하여, 인간-AI 공동 연구의 발전을 위한 통찰을 제시했다는 점입니다. 뿐만 아니라, 두 가지 혁신적인 연구 패러다임을 제안하고, 인간-AI 공동 연구에 있어 세 가지 협력 모델을 제시함으로써 AI가 진정한 연구 협력자로 자리매김할 수 있는 방법을 모색하고 있습니다.



### A Novel Approach to Eliminating Hallucinations in Large Language Model-Assisted Causal Discovery (https://arxiv.org/abs/2411.12759)
- **What's New**: 이 논문은 대규모 언어 모델(LLM) 사용 시 발생하는 환각(hallucinations)에 대한 최초의 조사를 제공합니다. 인과 발견(causal discovery)에서 인간 전문가 대신 LLM을 사용하는 증가하는 추세에 따라, 최적의 모델 선택의 중요성이 증가하고 있음을 강조합니다. 기존 LLM의 환각 문제를 다루며, 이를 감소시키기 위한 새로운 방법을 제안하고 있습니다.

- **Technical Details**: 연구에서는 Retrieval Augmented Generation (RAG) 방식을 사용하여 고품질 데이터(quality data)가 있을 때 환각을 줄이는 방법을 제안합니다. 또한, 여러 LLM을 사용하는 새로운 방법론을 도입하여 이들이 논쟁(debate)을 통해 인과 그래프(causal graphs)에서의 엣지를 감시(audit)하는 방식을 소개합니다. 이 방법은 RAG와 유사한 수준으로 환각을 감소시키는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 제안된 방법은 기존의 LLM 사용 시 발생하는 환각을 효과적으로 감소시키며, 인과 발견의 정확도를 높이는 데 기여할 수 있습니다. 연구 결과는 LLM의 선택이 인과 발견의 결과에 중요한 영향을 미친다는 것을 입증하며, 제시된 접근 방식들이 실제 적용 가능성을 보여줍니다.



### An exploration of the effect of quantisation on energy consumption and inference time of StarCoder2 (https://arxiv.org/abs/2411.12758)
- **What's New**: 이 연구는 코드 대형 언어 모델(code Large Language Models, LLM)의 추론(inference) 과정에서 에너지 소비를 줄이기 위한 양자화(quantization) 및 가지치기(pruning) 전략을 조사합니다. StarCoder2를 사용한 결과, 양자화는 처리량(throughput)이 낮아져 더 많은 에너지가 필요하고, 정확도(accuracy)가 떨어질 수 있음을 확인했습니다. 반면, 가지치기는 에너지를 줄일 수 있으나 성능을 저하시키는 한계가 있습니다. 이러한 결과는 LLM 모델 압축의 어려움과 트레이드오프(trade-off)를 강조합니다.

- **Technical Details**: AI의 에너지 소비를 줄이는 것은 중요한 문제로, 특히 추론 단계에서 사용자 인터랙션이 증가함에 따라 그 영향력이 커집니다. 양자화는 모델의 무게를 낮은 비트 포맷으로 변환하여 메모리를 감소시키고, 이는 사용자가 개인 장치에서 대형 모델을 실행할 수 있도록 돕습니다. 이 연구에서는 양자화와 가지치기를 통해 에너지 소비를 줄이는 방법을 탐구하며, 양자화가 성능 손실 없이도 모델을 작게 만들 수 있음에 유의합니다.

- **Performance Highlights**: 연구 결과, 양자화와 가지치기 모두 LLM의 에너지 소비를 감소시킬 수 있지만 각기 다른 성능적 영향을 미칩니다. 특히 양자화의 경우, QLoRA를 통해 양자화된 모델이 ChatGPT와 유사한 성능을 제공할 수 있게 되었습니다. 하지만 가지치기는 성능 저하를 초래할 수 있는 가능성이 높아, 향후 연구에서는 하드웨어 최적화 양자화 방법을 탐색하여 성능 손실을 최소화하는 방향으로 나아갈 필요성이 강조됩니다.



### FedCL-Ensemble Learning: A Framework of Federated Continual Learning with Ensemble Transfer Learning Enhanced for Alzheimer's MRI Classifications while Preserving Privacy (https://arxiv.org/abs/2411.12756)
Comments:
          6 pages, 4 figures

- **What's New**: 이 연구는 알츠하이머 질병(Alzheimer's disease)의 분류(classification)를 위한 새로운 접근법을 제안합니다. 고급 딥 러닝 기술(deep learning techniques)을 활용하고, 안전한 데이터 처리 방법을 결합하여 향상된 성능을 보이는 모델을 개발했습니다. 특히, 전이 학습 모델(transfer learning models)을 사용하여 의료 이미지 데이터에서 고급 특징을 추출하는 데 중점을 두었습니다.

- **Technical Details**: 주요 기술로는 ResNet, ImageNet, VNet과 같은 전이 학습 모델을 사용하여 알츠하이머 관련 미세한 패턴을 감지할 수 있습니다. 이 모델은 데이터 소스의 다양성에 대해 강력한 특징을 추출할 수 있도록 조정되었습니다. 또한, 페더레이션 학습(federated learning) 접근법을 통합하여 데이터 개인 정보 보호를 보장하면서 분산된 모델의 이점을 최대한 활용합니다.

- **Performance Highlights**: 실험 결과는 알츠하이머 분류의 정확성을 향상시키는 데 기여하며, 안전하고 협력적인 건강 관리 데이터 분석을 위한 프레임워크도 제공합니다. 추가적으로, 데이터 전송 시 기밀성과 무결성을 보장하기 위해 암호 기반 encryption 메커니즘도 적용되었습니다. 이러한 결과는 예측 성능을 개선하고 환자 데이터를 공유하지 않으면서도 모델의 강력한 학습을 가능하게 합니다.



### A Survey of Financial AI: Architectures, Advances and Open Challenges (https://arxiv.org/abs/2411.12747)
Comments:
          Full list of papers and summary slides are available at: this https URL

- **What's New**: 본 보고서는 인공지능(AI)을 활용한 금융 시장 예측, 포트폴리오 최적화, 자동 거래에서의 최근 발전을 체계적으로 분석하였습니다. 특히, 금융 타임 시리즈를 위한 foundation models, 시장 관계 모델링을 위한 graph-based architectures, 포트폴리오 최적화를 위한 hierarchical frameworks와 같은 주요 혁신을 점검하였고, 이로 인해 모델의 복잡성과 실제 제약 조건 간의 중요한 trade-offs를 조명하였습니다.

- **Technical Details**: 이 보고서는 금융 AI 응용 프로그램의 다양한 기술적 접근 방식을 포괄적으로 정리하였습니다. 이를 통해 deep learning 모델의 구조적 혁신, 훈련 및 최적화 분야의 발전, 그리고 실제 구현 및 확장성의 개선과 같은 세 가지 주요 방향을 제시하였습니다. 또한, 예측적 작업과 의사결정 작업 간의 수학적 기초를 통합하는 통일된 공식 프레임워크를 제공하여 예측 및 결정 문제를 포괄해 다루었습니다.

- **Performance Highlights**: 보고서는 실제 금융 애플리케이션과 학술 문헌을 철저히 분석하여, AI 아키텍처와 방법론의 혁신이 금융 모델링에 대한 보다 정교한 접근을 가능하게 한다는 것을 보여주었습니다. 또한 이 연구는 금융 결정 및 산업 관행에 대한 심도 있는 논의를 포함하여, 실제 구현 고려 사항을 강조하면서 이론적 진전을 다루고 있습니다. 미래 연구를 위한 주요 방향과 도전 과제가 제시되어 있으며, 이는 학계와 산업 간의 간극을 메우는 데 기여할 것입니다.



### A Review of Reinforcement Learning in Financial Applications (https://arxiv.org/abs/2411.12746)
- **What's New**: 최근 금융 분야에서 강화 학습(Reinforcement Learning, RL)을 활용한 연구가 증가하고 있습니다. 이 조사 논문은 RL의 금융 애플리케이션에 대한 포괄적인 연구를 제공하며, 기초 분석을 통해 기존 문헌에서 나타나는 공통 주제를 탐구합니다. RL의 성능에 영향을 미치는 주요 요인들과 적용 시의 도전 과제를 식별하고, 이를 극복하기 위한 최근 발전을 논의합니다.

- **Technical Details**: 이번 조사에서는 모델 프리(model-free) 및 모델 기반(model-based) RL 알고리즘을 소개합니다. 특히, 액터 전용(actor-only), 비평가 전용(critic-only), 액터-비평가(actor-critic) 방법론을 다룹니다. 다양한 알고리즘인 정책 기울기(Policy Gradient, PG), 근접 정책 최적화(Proximal Policy Optimization, PPO)가 포함되며, 이러한 알고리즘들은 금융 데이터 분석에 활용될 수 있는 능력을 제공합니다.

- **Performance Highlights**: RL은 금융 시장의 고유한 데이터 복잡성에 적합하며, 기존 방법에 비해 우수한 성능을 발휘할 수 있는 잠재력을 지니고 있습니다. 분석된 연구에 따르면, RL은 특히 포트폴리오 관리 및 최적 실행과 같은 핵심 영역에서 향상된 결과를 보였습니다. 하지만, 금융 데이터의 잡음 및 비정상성, 분포의 문제 등은 RL의 활용에 있어 여전히 해결해야 할 도전 과제로 남아 있습니다.



### SRA-MCTS: Self-driven Reasoning Augmentation with Monte Carlo Tree Search for Enhanced Code Generation (https://arxiv.org/abs/2411.11053)
- **What's New**: 이 논문에서는 복잡한 문제를 해결하기 위한 새로운 데이터 생성 프로세스인 SRA-MCTS(Reasoning-Augmented Monte Carlo Tree Search)를 제안하고 있습니다. 이 방법은 모델이 자율적으로 고급 중간 추론 경로를 생성하도록 세심하게 안내합니다. 결과적으로, 이 프로세스는 지속적인 개선을 위한 긍정적인 피드백 루프를 생성하여 자연어 추론 경로를 실행 가능한 코드로 변환합니다.

- **Technical Details**: SRA-MCTS는 세 가지 주요 단계로 구성된 데이터 생성 파이프라인을 제안합니다: 계획 생성, 계획을 코드로 변환, 모델 학습입니다. 이 과정의 핵심은 Monte Carlo Tree Search(MCTS)의 가지 선택 메커니즘을 활용하여 문제의 최적 해결책을 선택하도록 하는 것입니다. 이를 통해 다양한 솔루션을 생성하고 각 단계에서 올바른 프로세스를 선택하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, SRA-MCTS를 통해 생성된 데이터로 미세 조정된 모델은 기존 모델 구조 및 CoT를 기반으로 한 미세 조정 모델보다 우수한 성능을 보였습니다. 특히, 작은 모델에서도 자가 개선의 잠재력이 크다는 것을 발견했습니다. CoT가 성능 저하를 보이는 경우에도 SRA-MCTS의 접근 방식은 더욱 개선된 다양성 메트릭을 보여주었습니다.



New uploads on arXiv(cs.LG)

### Promoting User Data Autonomy During the Dissolution of a Monopolistic Firm (https://arxiv.org/abs/2411.13546)
Comments:
          This paper appeared at the 2nd Workshop on Regulatable ML at NeurIPS 2024

- **What's New**: 이번 논문은 소비자 제품에 AI를 배포하는 데 있어, 대형 신경망인 foundation models에 의존함으로써 발생할 수 있는 독점적 행동의 위험과 그에 대한 해결책으로서의 분해(dissolution)에 대해 다룹니다. 저자들은 독점적인 AI 기업을 반독점적으로 해결하기 위해 여러 작은 자회사로 나누는 방법을 논의하며, 데이터 자율성을 보장하기 위한 Conscious Data Contribution 프레임워크를 제안합니다. 이 연구는 텍스트 기반 시뮬레이션을 통해 기업 분할시의 사용자 데이터 자율성 확보 가능성을 탐구합니다.

- **Technical Details**: 저자들은 AI 기업의 분해 시나리오를 설정하면서, 독점 기업 F가 다수의 후속 기업 S1, S2, ..., Sk로 나뉘는 과정을 설명합니다. 이러한 상황에서 데이터셋 U 및 이를 기반으로 한 모델들이 어떻게 영향을 받을지를 고민하며, 사용자 데이터의 소유권과 데이터를 공유할 권리(respective data consent)를 강조합니다. 이 과정에서 Conscious Data Contribution (CDC)와 Machine Unlearning 개념을 활용하여 사용자들이 자신의 데이터를 어떻게 관리할 수 있는지에 대한 원칙을 세웁니다.

- **Performance Highlights**: 시뮬레이션 연구를 통해 저자들은 fine-tuning을 이용한 machine unlearning이 사용자의 요청에 따라 데이터를 효과적으로 제거하는 방법으로 유용하게 작용할 수 있음을 발견했습니다. 특히, 이 과정에서 발생하는 "catastrophic forgetting" 현상이 모델의 특정 정보를 지우는 데 도움이 될 수 있음을 강조하며, 이는 사용자 중심의 데이터 사용 방식을 강화합니다. 이러한 접근은 AI 시장에서 소규모 기업들이 경쟁할 수 있는 토대를 마련하는 데 중요한 역할을 할 것으로 기대됩니다.



### Metacognition for Unknown Situations and Environments (MUSE) (https://arxiv.org/abs/2411.13537)
- **What's New**: 이 연구는 자율 에이전트가 새로운 환경에서의 적응력을 향상시키기 위해 메타인지(metacognition)를 통합하는 새로운 프레임워크인 MUSE(Metacognition for Unknown Situations and Environments)를 제안합니다. 메타인지는 자신의 인지 과정에 대한 인식과 조절을 포함하며, 이는 AI 에이전트가 낯선 도전에 대처하는 데 필요한 인지적 유연성을 제공합니다. MUSE 프레임워크는 두 가지 초기 구현(세계 모델링 및 대형 언어 모델(LLMs) 기반)을 소개하여 메타인지 사이클의 구현을 보여줍니다.

- **Technical Details**: MUSE 프레임워크는 자기 인식(self-awareness) 및 자기 조절(self-regulation) 메커니즘을 통합하여 에이전트가 주어진 작업을 성공적으로 수행할 확률을 예측하고 iteratively 전략을 선택할 수 있도록 설계되었습니다. 이 시스템은 과거 경험에 기반하여 내부 모델을 지속적으로 학습하여 동작 계획의 성공 가능성을 평가합니다. MUSE 에이전트는 Meta-World 및 ALFWorld 환경에서의 실험을 통해 기존 강화 학습 접근법보다 새로운 시나리오 처리에서 상당한 개선을 보여주었습니다.

- **Performance Highlights**: MUSE 프레임워크는 에이전트의 적응성을 높이기 위해 주요 평가 지표로 역량(competence)을 우선시합니다. 이 접근은 에이전트가 낯선 상황에 만나고도 고립되지 않도록 하여 효과적인 탐색을 촉진합니다. 결과적으로 MUSE 에이전트는 과제 성공 가능성을 극대화하는 행동 계획을 식별하고 온라인 학습 및 적응에서 더 안전하고 효과적으로 작동할 수 있게 됩니다.



### Advancing Heatwave Forecasting via Distribution Informed-Graph Neural Networks (DI-GNNs): Integrating Extreme Value Theory with GNNs (https://arxiv.org/abs/2411.13496)
Comments:
          23 pages, 13 figures, pdf format

- **What's New**: 이 논문은 기후 변화로 인해 빈도와 심각성이 증가한 극단적인 더위인 열파(heatwave)의 효과적인 예측을 위해 Distribution-Informed Graph Neural Network (DI-GNN)라는 새로운 프레임워크를 소개합니다. DI-GNN은 Extreme Value Theory (EVT)의 원리를 그래프 신경망 아키텍처에 통합하여 열파와 같은 고유한 및 드물게 발생하는 사건을 더 잘 탐지할 수 있게 합니다. 이 모델은 Generalized Pareto Distribution (GPD)에서 파생된 기술적 설명자를 특징 공간, 인접 행렬, 손실 함수에 통합하여 열파 발생의 복잡성을 이해하는 데 도움을 줍니다.

- **Technical Details**: DI-GNN은 기후 분포의 꼬리(tail)를 우선시 함으로써 기존 예측 모델의 한계를 극복하고, 특히 불균형 데이터세트에서 전통적인 정확도와 같은 지표가 오해를 일으킬 수 있는 상황에서 효과적으로 작동합니다. 본 연구는 브리티시컬럼비아의 기상 데이터에 대한 실험 평가를 통해 DI-GNN의 우수한 성능을 입증하며, 균형 정확도(balanced accuracy), 재현율(recall), 정밀도(precision)에서 개별 모델보다 탁월한 성과를 나타냅니다. 이러한 접근 방식은 극단적인 기후 이벤트를 이해하고 예측하기 위한 데이터 기반 방법론의 발전을 나타냅니다.

- **Performance Highlights**: DI-GNN은 높은 AUC와 평균 정밀도 점수를 달성하며, 열파 이벤트를 구별하는 데 있어 강력한 견고성을 보입니다. 이 모델은 다른 기후 극단적 이벤트인 한파, 홍수, 가뭄 및 대기 차단 현상과 같은 예측에도 적용될 수 있는 가능성을 보여줍니다. DI-GNN의 발전은 정책 입안자와 응급 대응 팀에게 재해 대비 및 인프라 복원력을 강화하기 위한 실행 가능한 통찰력을 제공합니다.



### SynEHRgy: Synthesizing Mixed-Type Structured Electronic Health Records using Decoder-Only Transformers (https://arxiv.org/abs/2411.13428)
- **What's New**: 본 논문에서는 전자 건강 기록(Electronic Health Records, EHR)의 합성 데이터를 생성하기 위한 새로운 토큰화 전략을 제안합니다. 이 전략은 covariates, ICD 코드, 불규칙하게 샘플링된 시간 시계열 데이터와 같은 다양한 데이터 유형을 포함한 구조화된 EHR 데이터에 최적화되어 있습니다. 저자들은 GPT와 유사한 디코더 전용 변환 모델을 사용하여 고품질의 합성 EHR을 생성하는 방법을 입증합니다.

- **Technical Details**: 제안된 방법론에서는 구조화된 EHR 데이터 생성에 있어서의 토큰화 전략을 혁신적으로 재구성하고, 여러 환자의 방문 데이터를 아우르는 복합형 구조를 다룹니다. 이 연구는 MIMIC-III 데이터셋을 활용하여 생성된 데이터의 품질(quality)을 주목하며, 특히 높은 누락률(missingness)과 불규칙한 시간 포인트로 인한 도전 과제를 해결하는 데 중점을 두고 있습니다. 또한, 비슷한 최첨단 모델들과 비교하여 데이터를 다룰 때의 유용성(utility)과 개인 정보 보호(privateness) 측면에서의 평가 결과도 제시됩니다.

- **Performance Highlights**: SynEHRgy라는 프레임워크는 여러 방문 데이터에 걸쳐 구조화된 EHR 데이터를 성공적으로 생성할 수 있는 방법을 제시하며, 특히 불규칙하게 샘플링된 시간 시계열 데이터 생성에서 뛰어난 성능을 발휘합니다. 제안된 접근 방식은 최신 모델들과 비교하여 생성된 데이터의 신뢰성(fidelity), 유용성(utility), 개인 정보 보호(privateness) 측면에서 우수한 결과를 가져왔습니다. 이 연구는 EHR 데이터 생성의 새로운 가능성을 제시하며, 의료 데이터의 수집과 활용에 있어 장기적 영향을 미칠 것으로 기대됩니다.



### A Survey On Enhancing Reinforcement Learning in Complex Environments: Insights from Human and LLM Feedback (https://arxiv.org/abs/2411.13410)
- **What's New**: 이 논문은 강화학습(Reinforcement Learning, RL) 분야의 최신 동향 및 도전 과제를 다룹니다. 특히, 대규모 관측 공간에서의 학습 효율성을 높이기 위해 인공지능과 인간 피드백의 융합 방식을 탐구합니다. 다양한 형태의 피드백을 통해 RL 에이전트의 성능 개선 가능성을 제시하며, 이전 연구를 기반으로 한 메타분석을 통해 이를 명확히 하고자 합니다.

- **Technical Details**: 강화학습에서는 에이전트가 환경과 상호작용하면서 결정을 내리고 학습을 진행하는 과정에서 샘플 비효율(sample inefficiency) 문제가 발생합니다. 이 문제는 차원의 저주(curse of dimensionality)로 이어져 복잡한 환경 속에서 RL 에이전트의 의사결정 및 주의(attention) 조절을 어렵게 만듭니다. 논문에서는 인간이나 대형 언어 모델(LLM)의 피드백을 통해 이 문제를 해결하고자 하며, 피드백의 형태는 자연어, 비자연어 및 다른 정보 제공 방법을 포함합니다.

- **Performance Highlights**: 인간의 자연어 피드백을 활용한 연구 결과, RL 에이전트는 더 나은 결과를 얻을 수 있음을 보여줍니다. 예를 들어, 특정 환경에서 자연어 명령을 통해 행동을 조정하거나, 사전 훈련된 언어 모델을 통해 명령을 생성하여 Zero-shot transfer가 가능하다는 점을 강조합니다. 이러한 접근법은 RL 에이전트의 학습 속도를 증가시키고 자원을 보다 효율적으로 사용할 수 있게 합니다.



### ODTE -- An ensemble of multi-class SVM-based oblique decision trees (https://arxiv.org/abs/2411.13376)
Comments:
          29 pages

- **What's New**: 본 논문에서는 오프라인 결정 트리(Oblique Decision Trees)를 기반 분류기로 사용하는 새로운 앙상블 방법인 ODTE를 제안합니다. 또한, 지원 벡터 머신(Support Vector Machines)을 활용하여 결정 노드 내에서 하이퍼플레인을 정의하는 STree라는 기본 알고리즘을 소개합니다. 이 방법은 다중 클래스 전략을 포함하여 비 이진 분류 작업을 직접 처리할 수 있도록 합니다.

- **Technical Details**: 오프라인 결정 트리는 결정을 내리는 노드에서 단일 최고의 성능을 보이는 모델인 SVM만을 유지하며, 이는 n-ary(classification task에서 불순도(impurity)를 최소화하는 데 중점을 둡니다. ODTE의 성능 비교 실험은 49개의 데이터셋을 사용하였고, 하이퍼파라미터(optimization)를 조정하여 의미 있는 성능 향상이 이루어졌습니다. STree를 통해 학습된 오프라인 결정 트리는 다른 알고리즘보다 더 간결한 구조를 가지는 것으로 나타났습니다.

- **Performance Highlights**: ODTE는 다양한 최첨단 알고리즘과 비교했을 때 지속적으로 우수한 순위를 기록했습니다. 하이퍼파라미터 조정을 통해 성능이 크게 향상되었으며, 실험 결과는 ODTE가 기존 방법들보다 더 경쟁력 있다는 것을 보여주고 있습니다. 또한, STree를 통해 생성된 오프라인 결정 트리는 구조적으로 더 간결하여 효율성을 증대시킵니다.



### Predicting Wall Thickness Changes in Cold Forging Processes: An Integrated FEM and Neural Network approach (https://arxiv.org/abs/2411.13366)
- **What's New**: 이번 연구는 튜브의 nosing 과정 중 벽 두께 변화를 예측하는 새로운 접근 방식을 제시합니다. 이를 위해 nosing 과정에 대한 철저한 분석 과정을 거친 후, 다양한 프로세스 파라미터의 영향을 분석하기 위해 유한요소법(FEM) 시뮬레이션을 설정합니다. 하지만, 전통적인 FEM 시뮬레이션이 정확하지만 시간 소모가 크고 계산 집약적이라는 문제점이 있어, 이를 보완하기 위해 그래프 신경망(graph neural networks) 기반의 대체 모델을 제안합니다.

- **Technical Details**: 제안된 그래프 신경망 구조는 nosing 프로세스와 관련된 다양한 정보를 직접 통합하여 객체 간의 상호 작용을 모델링합니다. 이 모델은 강력한 시뮬레이션 데이터 생성 프레임워크를 제공하며, 전통적인 FEM 모델과 유사한 수준의 정확도를 유지하면서 실시간 예측을 가능하게 합니다. 또한, 두께 곡선 간 면적(area between thickness curves, ABTC)이라는 새로운 평가 지표도 도입하여 다양한 모델의 성능을 정량적으로 평가합니다.

- **Performance Highlights**: 제안된 접근 방식은 광범위한 실험을 통해 검증되어 매우 유망한 성능을 보여줍니다. 실험 결과는 신경망 모델이 FEM 시뮬레이션을 대체할 수 있는 잠재력을 제시하며, 벽 두께 변화를 높은 정확도로 예측하면서 계산 비용을 줄일 수 있음을 입증합니다. 이 연구는 냉간 단조 공정 분야에 중요한 발전을 의미하며, 향후 제조 작업의 효율성과 정확성을 높이는 데 기여할 것으로 기대됩니다.



### Vertical Validation: Evaluating Implicit Generative Models for Graphs on Thin Support Regions (https://arxiv.org/abs/2411.13358)
Comments:
          Accepted to UAI 2024

- **What's New**: 이번 논문에서는 implicit graph generative models를 위한 새로운 평가 방법인 Vertical Validation (VV)을 제안합니다. 이 방법은 훈련과 테스트 과정에서 thin support regions를 체계적으로 생성하여, 기존 방법들이 포착하지 못했던 분포의 미지 영역에서의 성능을 평가할 수 있도록 합니다. 새로운 방법은 모델 선택뿐만 아니라 과적합 (overfitting) 감지에도 효과적입니다.

- **Technical Details**: VV 접근 방식은 표준 기계 학습의 train-test 절차를 일반화한 것으로, 샘플의 특징에 의존하여 분할을 진행합니다. 본 연구에서 사용된 주요 기술적 요소는 empirical CDF를 기반으로 샘플들을 단위 구간에 투영하는 방법과 중요도 가중치를 추정하는 kernel mean matching입니다. 이를 통해 모형이 실제로 예상되는 thin support regions에서의 성능을 어떻게 측정할 수 있는지 명확하게 설명합니다.

- **Performance Highlights**: 연구 결과에 따르면, VV 접근 방식은 thin support regions에서 모델 선택에 신뢰할 수 있는 신호를 제공합니다. 기존 방식들이 과적합 문제를 제대로 감지하지 못했던 반면, VV 방법은 이러한 문제를 보다 효과적으로 탐지할 수 있음을 보여주었습니다. 실험 결과, 이 새로운 평가 방식이 implicit generative models의 성능을 정확하게 измер하는데 중요하다는 것을 입증했습니다.



### Verifying Machine Unlearning with Explainable AI (https://arxiv.org/abs/2411.13332)
Comments:
          ICPRW2024

- **What's New**: 본 연구에서는 Explainable AI (XAI)를 활용하여 Machine Unlearning (MU)의 효과성을 검증하는 방법을 탐구합니다. 특히, 데이터 삭제를 위한 전통적인 ML 모델 재훈련 방법의 비효율성을 해소하기 위해 MU를 적용하여 특정 학습 패턴을 선택적으로 잊도록 하였습니다. 본 논문은 XAI를 통해 MU의 성능을 평가하는 새로운 지표인 Heatmap Coverage (HC)와 Attention Shift (AS)를 제안하며, 이는 MU 검증의 혁신적인 단계를 제공합니다.

- **Technical Details**: XAI는 ML 모델의 결정 과정에 대한 통찰력을 제공하여, 민감한 정보를 성공적으로 삭제했는지, 새로운 문제를 일으키지는 않았는지 평가합니다. 본 연구에서는 데이터 재레이블링, 모델 섭동 등 다양한 MU 기법을 탐색하며 각 기법의 도전 과제를 논의합니다. 이 과정에서 атриbuty 기반 XAI 기술을 활용하여 지역적 특성의 중요성 변화를 관찰하고 정량화합니다.

- **Performance Highlights**: 본 연구의 주요 기여는 MU 기법의 효과를 검증하기 위해 атриbuty 기반 XAI 메소드를 적용한 것입니다. 이를 통해 모델이 특정 데이터 삭제 후 주목하는 위치를 시각화하고 정성적으로 평가할 수 있습니다. 또한, MU 프로세스를 평가하기 위해 HC와 AS라는 새로운 XAI 기반 지표를 도입하여, 기존의 정확도 지표를 넘어서는 포괄적인 평가를 제공합니다.



### Transformers with Sparse Attention for Granger Causality (https://arxiv.org/abs/2411.13264)
- **What's New**: 이번 연구는 심층 학습 기반의 Sparse Attention Transformer 모델을 통해 다변량 시계열 데이터의 인과 관계를 분석하는 새로운 접근 방식을 제안합니다. 기존의 Granger Causality 방법은 고정된 지연 시간(lag) 길이를 가정하였지만, 본 방법은 변화하는 지연 의존성을 반영하여 모델이 가장 중요한 과거 시점을 선택하도록 합니다. 모델의 자기 주의(attention) 모듈을 수정함으로써 더 정교한 인과 관계 추론을 가능하게 만듭니다.

- **Technical Details**: Sparse Attention Transformer(SAT)는 시간 단계 간에 인과 관계를 확인하기 위해 두 가지 접근 방식을 조합한 새로운 구조입니다. 첫째, 시간에 따라 주의(attention)를 먼저 수행하고, 다음으로는 각 변수 간의 주의(attention)를 개별적으로 계산하여 Granger Causality 지수를 산출합니다. 이로 인해 모델이 시계열의 불규칙한 지연 효과를 더 효과적으로 처리할 수 있게 됩니다.

- **Performance Highlights**: SAT는 여러 합성 벤치마크 데이터셋을 통해 실험적으로 그 성능을 입증하였으며, 전통적인 벡터 자기회귀(Vector Autoregression) 기반의 Granger Causality 방법과 비교하여 유의미한 차별성을 나타냈습니다. 특히, 모델은 고정된 시간 지연을 요구하지 않아 무작위 지연이 존재하는 실제 환경에서도 효과적으로 작동함을 보여줍니다.



### Engagement-Driven Content Generation with Large Language Models (https://arxiv.org/abs/2411.13187)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 사회적 영향력, 특히 소셜 네트워크 내에서의 상호작용을 탐구합니다. 기존 연구들은 LLMs의 능력을 일대일 상호작용에서 주로 다루었으나, 이 연구는 넓은 연결 구조에서의 LLMs의 잠재력을 강조합니다. 연구 질문은 LLMs가 사용자 참여를 극대화하는 의미 있는 콘텐츠를 생성할 수 있는지를 검토합니다.

- **Technical Details**: 연구는 LLM 기반 콘텐츠 생성을 위한 파이프라인을 정의하고, 강화 학습(reinforcement learning)과 시뮬레이션된 피드백(simulated feedback)을 사용하여 보상을 설정합니다. 이 프레임워크는 LLM이 주제에 맞는 콘텐츠를 생성하고 최소한의 유창성(fluency) 기준을 충족하도록 요구합니다. 연구진은 공정 피드백 루프를 통해 LLM이 주제, 사회적 네트워크 구조 및 의견 분포에 적응하는 과정을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기초적인 전파 모델에 무관하며 완전히 적응적임을 나타냈습니다. LLM 에이전트는 생성한 콘텐츠의 감정을 자동으로 적응시키며, 실제 네트워크 내에서 생성된 콘텐츠와 동등한 수준의 참여를 생성하는 경향을 보였습니다. 이 연구는 복잡한 참여 과제에 맞게 쉽게 조정될 수 있는 가능성을 보여줍니다.



### A Unified Analysis for Finite Weight Averaging (https://arxiv.org/abs/2411.13169)
Comments:
          34 pages

- **What's New**: 이번 연구에서는 유한 가중치 평균(Finite Weight Averaging, FWA)을 활용하여 확률적 경량 경량화(Stochastic Gradient Descent, SGD) 및 최신 가중치 평균(LAtest Weight Averaging, LAWA)과의 이론적 차별성을 탐구합니다. FWA는 이전의 기법들과 비교했을 때 더 빠른 수렴(convergence) 속도를 보여주고, 일반화(generalization) 성능 또한 개선할 수 있는 가능성을 지니고 있습니다. 특히 이 연구는 FWA의 수렴 경계와 일반화 성능을 뒷받침하는 수학적 증거를 제공합니다.

- **Technical Details**: FWA는 특정한 수학적 정의와 함께 SGD와 LAWA를 연결하는 방법으로 제시됩니다. 연구에서는 FWA의 수렴 분석을 위해 로그 수렴 경계를 $	extmathcal{O}(rac{	extlog(T/2k)}{	extsqrt{T}})$로 정립하였으며, $k$는 최종 반복 횟수를 나타내는 상수로 정의됩니다. FWA는 SGD 경우와 비교하여 이론적으로 더 빠른 수렴 속도를 달성하였으며, 평균 포인트의 수에 따라 성능이 향상됨을 검증하였습니다.

- **Performance Highlights**: 실험 결과, FWA가 테스트 벤치마크에서 SGD보다 우수한 성능을 보여주는 것이 확인되었습니다. 특히, FWA는 가중치 평균에 있어 k의 크기가 커질수록 더 나은 수렴 성능을 보인다고 보고되었습니다. 결과적으로 FWA는 다양한 설정에서 SGD보다 양호한 일반화 성능을 나타내는 것으로 평가됩니다.



### Unlocking Historical Clinical Trial Data with ALIGN: A Compositional Large Language Model System for Medical Coding (https://arxiv.org/abs/2411.13163)
- **What's New**: ALIGN는 역사적 임상 시험 데이터를 재사용하는 과정에서의 상호 운용성 문제를 해결하기 위해 개발된 새로운 LLM 기반 시스템입니다. 이 시스템은 레이블이 없는 데이터에서의 자동 코딩을 통해 의료 연구와 약물 개발을 가속화하는 데 기여합니다. 세 가지 단계로 구성된 ALIGN은 다양한 후보 코드 생성, 코드의 자기 평가 및 신뢰성 향상을 위한 불확실성 추정을 포함합니다.

- **Technical Details**: ALIGN는 임상 시험에서의 약물 용어를 Anatomical Therapeutic Chemical (ATC) 및 Medical Dictionary for Regulatory Activities (MedDRA) 코드로 조화시키기 위해 사용됩니다. 22개의 면역학 시험에서 테스트한 결과, ALIGN은 LLM 기반 모델보다 월등한 성과를 보였습니다. 특히 MedDRA 코딩에서 다양한 수준에서 높은 정확도를 달성하였고, ATC 코딩에서도 모든 계층 수준에서 우수한 성능을 나타냈습니다.

- **Performance Highlights**: ALIGN의 불확실성 기반 지연(deferral) 전략은 일반적으로 드물게 나타나는 약물의 성능을 크게 향상시켰습니다. 전체적으로 ATC Level 4에서 72-73%의 정확도를 기록하며, 일반 약물에 대해서는 86-89%의 정확도를 보였고, 기존 기반선 모델을 7-22% 초과한 성과를 거두었습니다. 이 시스템은 비용 효율적으로 운영되며, $0.0007 및 $0.02의 비용으로 코드당 처리할 수 있어 임상 채택 장벽을 줄이는 데 기여하고 있습니다.



### Long-term Detection System for Six Kinds of Abnormal Behavior of the Elderly Living Alon (https://arxiv.org/abs/2411.13153)
Comments:
          20 pages, 3 figures

- **What's New**: 본 연구는 일본의 노인 인구가 증가함에 따라, 스마트 홈 환경에서 자동으로 이러한 건강 문제를 조기에 식별할 수 있는 센서 기반 탐지 시스템을 제안하고 있습니다. 이 시스템은 반 침대 상태, 집에 갇히는 상태, 잊어버림, 배회, 걷는 중 넘어짐 및 서 있는 중 넘어짐과 같은 여섯 가지 전형적인 이상 행동을 감지할 수 있도록 설계되었습니다. 다양한 공간 배치, 센서 배열 및 거주자의 특성에 따라 맞춤화할 수 있는 것을 강조하고 있습니다.

- **Technical Details**: 이 시스템은 시뮬레이터 기반으로 구축되어, 각 이상 행동의 발생 지속 시간에 따라 적절한 탐지 분류기를 훈련합니다. 예를 들어, 집에 갇혀 있는 경우는 일별, 넘어짐의 경우는 초 단위로 레이블을 생성합니다. 수치 평가를 통해, 이 시스템은 배회와 넘어짐을 탐지하는 데 있어 기존 방법과 비교할 만한 성능을 보여줍니다.

- **Performance Highlights**: 제안된 탐지 방법은 실제 사례에서 얻은 데이터를 기반으로 하지 않고, 여전히 유효한 성능을 보여줍니다. 감지된 기간이 실제 이상 기간과 약간이라도 겹칠 경우를 포함한 성능 지표에서, 각 이상 행동은 높은 민감도와 낮은 잘못된 경고 비율을 나타내었습니다. 예를 들어, 반 침대 상태에서는 민감도가 1.0까지 도달했습니다.



### Domain Adaptive Unfolded Graph Neural Networks (https://arxiv.org/abs/2411.13137)
- **What's New**: 이번 연구에서는 그래프 신경망(GNN) 아키텍쳐의 개선을 통해 그래프 도메인 적응(Graph Domain Adaptation, GDA)을 원활하게 할 수 있는 방법을 모색합니다. 특히, 최적화 문제를 기반으로 설계된 겹쳐진 GNN(UFNNs)을 다루며, 이들의 전이 과정에서 발생하는 하위 수준의 목표값 변화가 상위 수준 목표에도 영향을 미친다는 사실을 발견하였습니다. 이를 바탕으로 하위 수준 목표값을 감소시킬 수 있는 'Cascaded Propagation (CP)'이라는 효과적인 전략을 제안합니다.

- **Technical Details**: Cascaded Propagation(CP) 전략은 하위 문제의 출력을 다시 입력으로 주입하여 하위 수준의 목표값을 줄이도록 설계되었습니다. 연구에서는 APPNP, GPRGNN, ElasticGNN과 같은 세 가지 대표적인 UGNN 아키텍쳐와 함께 CP의 효율성을 평가합니다. 복잡한 GDA 과정에서 UGNN의 특징을 활용해 예측 성능을 개선하기 위한 여러 가지 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, CP가 통합된 UGNN들이 기존의 GDA 방법론에 비해 월등한 성능을 보이는 것으로 나타났습니다. 다섯 개의 실제 데이터셋을 대상으로 한 포괄적인 실험을 통해 이 신뢰할 수 있는 성능 향상을 확인하였습니다. 특히, GDA 작업의 다양한 UGNN 아키텍쳐와 데이터셋에서 일관된 성능 개선이 이루어졌습니다.



### Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders (https://arxiv.org/abs/2411.13117)
- **What's New**: 최근 연구들은 Sparse Autoencoders (SAEs)를 활용하여 신경망 표현에서 해석 가능한 특징을 밝혀내는 가능성을 보여주고 있습니다. 그러나 SAEs의 단순한 선형 비선형 인코딩 메커니즘은 정확한 희소 추론을 수행하는 데 제한이 있습니다. 본 논문에서는 희소 부호화(sparse coding)에 대한 관점을 통해 SAEs의 희소 추론과 학습을 조사하고, 인코딩과 디코딩 과정을 분리하여 다양한 희소 인코딩 전략을 비교합니다.

- **Technical Details**: SAEs는 입력 데이터를 희소한 잠재 공간으로 매핑하는 인코더와 이 잠재 표현에서 입력을 재구성하는 디코더로 구성됩니다. 희소 코딩 기술은 입력 데이터를 희소한 기저 벡터의 선형 조합으로 표현하는 것을 목표로 하며, reconstruction error를 최소화하면서 희소성을 유지합니다. 다만, SAEs는 효율적인 인코딩 기능을 학습하기 위해 경량의 인코더를 사용하며, 이 과정에서 최적의 희소성을 희생할 수 있습니다. 이러한 희소 추론의 한계를 'amortisation gap'으로 정의하였습니다.

- **Performance Highlights**: 이번 연구에서는 다양한 인코딩 방법을 합성 데이터셋에서 평가하였고, 진정한 희소 특징과의 정렬 및 올바른 희소 코드 추론 능력을 두 가지 차원에서 분석하였습니다. 실험 결과, 최소한의 계산 비용 증가로 인상적인 성능 향상을 달성할 수 있음을 확인하였으며, 이는 대형 언어 모델(LLMs)에서도 유사한 해석 가능성을 달성할 수 있음을 입증합니다. 이 연구는 신경망 표현을 이해하는 데 새로운 가능성을 열어주며, 대형 언어 모델의 활성화를 분석할 도구를 개선하는 데 중요한 함의를 제공합니다.



### Provably Efficient Action-Manipulation Attack Against Continuous Reinforcement Learning (https://arxiv.org/abs/2411.13116)
- **What's New**: 본 논문에서는 임의의 RL 기반 에이전트가 지속적인 액션 공간을 통해 특정 행동을 조작하는 공격을 조사합니다. 기존 연구들이 주로 이산 상태에 초점을 맞췄던 반면, 저자들은 사이버-물리적 시스템(CPS)과 같은 실제 응용 프로그램에 적합한 연속적 액션 조작에 대한 공격 방법을 제안합니다. 저자들은 Monte Carlo tree search 방법을 활용하여 LCBT(하한 신뢰 경계 트리)라는 새로운 블랙박스 공격 알고리즘을 개발하고, 이를 통해 공격의 효율성을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 논문의 주요 내용은 연속적 상태 및 액션 공간에서의 행동 조작 공격에 대한 위협 모델을 구성하고, 공격 비용의 경계를 설정하는 것입니다. 저자들은 공격자가 마르코프 의사 결정 과정(MDP)에 대한 충분한 정보를 갖고 있는 화이트박스 시나리오와, 정보를 바탕으로 행동 조작 전략을 구성하는 블랙박스 시나리오를 구분합니다. 특히, LCBT 알고리즘은 아울러 이론적으로 민첩하게 출력을 조작할 수 있는 방법을 제시하며, 이를 통해 공격 비용을 줄일 수 있음을 증명합니다.

- **Performance Highlights**: 실험 결과, DDPG, PPO, TD3와 같은 여러 공격적인 RL 알고리즘에서 제안된 공격 방법의 성능을 입증하였습니다. LCBT 알고리즘은 타겟 정책에 수렴하는 데 필요한 공격 비용이 서브라인형 공격 비용으로 제한됨을 보여주며, 이는 기존의 공격 방법들보다 효율적입니다. 이번 연구는 연속적 액션 조작 공격에 대한 심층 분석을 제공하며, RL 시스템의 안전성을 높이는 데 중요한 기여를 합니다.



### DRL-Based Optimization for AoI and Energy Consumption in C-V2X Enabled IoV (https://arxiv.org/abs/2411.13104)
Comments:
          This paper has been submitted to IEEE Journal. The source code has been released at: this https URL

- **What's New**: 본 논문에서는 C-V2X 통신 시스템에서 Age of Information (AoI)을 최적화하는 새로운 방법을 제안합니다. 특히, 다중 우선순위 큐와 NOMA (Non-Orthogonal Multiple Access)의 영향을 분석하여, 통신 리소스 선택에서 발생할 수 있는 충돌 문제를 해결하려고 합니다. 이러한 접근은 시스템의 에너지 소비와 통신 효율성을 개선하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: C-V2X는 3GPP에 의해 정의된 기술로, 차량 간의 직접적인 통신(V2V)을 포함하여 복잡한 통신 시스템을 다룹니다. 본 논문에서는 Deep Reinforcement Learning (DRL)을 활용하여 차량 간 리소스 할당 및 AoI 최소화를 위한 최적 전략을 학습합니다. 또한, 다중 우선순위 큐를 통한 메시지 전송의 영향을 분석하며, 차량이 반이중 자원 선택을 통해 NOMA 기술을 사용할 때의 효과를 상세히 다룹니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션을 통해 기존 메소드 대비 에너지 소비 및 AoI에서 개선된 성과를 보여주었습니다. 리소스의 예약 및 동적 조정과 같이 최적화된 전략을 통해 통신의 신뢰성과 지연 문제를 해결하는데 기여합니다. AoI를 최적화함으로써, 높은 데이터 전송률과 높은 밀도의 연결에 적합한 서비스 품질을 유지할 수 있습니다.



### Incremental Label Distribution Learning with Scalable Graph Convolutional Networks (https://arxiv.org/abs/2411.13097)
Comments:
          Accepted by the 26th IEEE International Conference on High Performance Computing and Communications (HPCC2024)

- **What's New**: 이번 논문에서는 Incremental Label Distribution Learning (ILDL)을 제안합니다. 기존의 Label Distribution Learning (LDL) 방식은 고정된 레이블 수를 전제로 하지만, 현실에서는 레이블이 지속적으로 변화하는 환경이 많습니다. ILDL은 이러한 변화에 효과적으로 대응해 새로운 레이블을 학습하면서도 기존 레이블에 대한 지식을 유지할 수 있는 방법을 모색합니다.

- **Technical Details**: SGLDL(Scalable Graph Label Distribution Learning)은 ILDL 문제를 해결하기 위한 새로운 접근 방식입니다. 이 방법은 새로운 레이블을 신속하게 학습하도록 돕는 New-label-aware Gradient Compensation Loss와, 인터-레이블 관계를 그래프 형태로 표현하여 관계 재구성을 용이하게 만드는 스케일러블 레이블 그래프(SLG)를 사용합니다. 이러한 기법은 레이블 간의 관계를 효율적으로 관리하고 잘못된 주의 할당 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SGLDL은 기존의 CIL(Class-incremental Learning) 기법과 통합한 방법들에 비해 더욱 우수한 성능을 기록했습니다. 특히, 다양한 LDL 데이터셋에서 새로운 알고리즘의 명확한 장점을 입증하였고, ILDL 문제를 위한 전용 설계의 중요성을 강조했습니다. 이는 실무에서도 큰 기여를 할 수 있는 결과로 보입니다.



### Omnipredicting Single-Index Models with Multi-Index Models (https://arxiv.org/abs/2411.13083)
- **What's New**: 최근 연구에서 서브파라미터화된 학습 모델인 omnipredictors의 개념이 정의되었습니다. 이 새로운 개념은 다양한 손실 함수에 대해 경쟁력 있는 예측기를 동시에 찾는 것을 목표로 합니다. 본 논문에서는 단일 지수 모델(single-index models, SIMs)에서의 omnipredictors의 간단한 구성을 제안하며, 이는 실용적인 샘플 복잡성과 실행 시간을 요구합니다. 특히, 이 알고리즘은 이전 연구보다 현저히 개선된 성능을 보여줍니다.

- **Technical Details**: 본 연구에서는 다양한 손실 함수에 대해 경쟁력을 가진 omnipredictors를 구축하는 방법을 소개합니다. 특히 이 알고리즘은 모노톤, Lipschitz 링크 함수로 정의된 손실에 대해 $rac{	ext{1}}{	ext{ε}^{4}}$ 샘플을 요구하며, 비-Lipschitz 링크 함수에 대해서는 $rac{	ext{1}}{	ext{ε}^{2}}$로 개선됩니다. 제안된 알고리즘은 Isotron 알고리즘에 대한 새로운 분석을 통해 이루어지며, 이는 단일 지수 모델을 익히는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 알고리즘은 단일 지수 모델에 대해 $rac{	ext{1}}{	ext{ε}^{4}}$의 샘플 복잡도를 여러분과 동시에 $	ext{O}(	ext{n})$의 거의 선형 시간에 실행됩니다. 이 연구는 선행 연구에서 도시한 $rac{	ext{1}}{	ext{ε}^{10}}$ 샘플을 요구했던 방식에 비해 상당한 성능 개선을 이루어 냈습니다. 따라서 이는 일반 손실 패밀리 및 비교자에 대한 적절한 omniprediction의 목표를 달성하기 위한 중요한 진전을 나타냅니다.



### Improving OOD Generalization of Pre-trained Encoders via Aligned Embedding-Space Ensembles (https://arxiv.org/abs/2411.13073)
Comments:
          Accepted at the Self-Supervised Learning Workshop and the Unifying Representations in Neural Models Workshop at NeurIPS 2024

- **What's New**: 이 논문은 self-supervised pre-trained encoders를 사용하여 out-of-distribution (OOD) 데이터에 대한 제로샷 일반화를 개선할 목적으로 구성된 Ensemble-InfoNCE라는 새로운 방법을 소개합니다. 이 방법은 임베딩 공간(embedding space)에서 앙상블을 구성하여 예측 공간(predictive space)에서의 해석력과 가중치 공간(weight space)의 유연성을 모두 제공합니다. 기존의 방법들은 라벨 없는 데이터로 효과적인 앙상블을 구성하는 문제에 접근하지 못했습니다. 따라서, 이 연구는 이론적 분석을 통해 개별 임베딩 공간 사이의 관계를 규명하고, 비지도적 방식으로 이들을 정렬하는 방법론을 제시합니다.

- **Technical Details**: 제안하는 Ensemble-InfoNCE 방법은 널리 사용되는 InfoNCE의 대조 손실을 사용하여 미리 훈련된 인코더들의 하이퍼스피어 임베딩 공간에서 평균을 집계하는 과정으로 진행됩니다. 각 인코더는 semantically 유사한 샘플을 유사한 방향으로 가리키게 임베딩 공간을 정렬해야 합니다. 이를 위해, 기존 연구 결과를 기반으로 한 비지도적 접근을 제안하며, 앙상블 인코더들의 정렬된 임베딩 공간을 통해 올바른 임베딩을 복원할 수 있음을 이론적으로 입증합니다.

- **Performance Highlights**: MNIST 데이터셋을 통해 실험 결과, 제안된 Ensemble-InfoNCE 방법이 단일 인코더에 비해 ID(인디스키피션)와 OOD 데이터 모두에서 미리 훈련된 임베딩 품질을 개선한다는 것을 보여줍니다. 이 방법은 잘 정렬된 임베딩 공간을 활용함으로써 OOD 데이터에 대한 일반화 성능을 효과적으로 향상시킵니다. 이러한 성능 개선은 self-supervised 비지도 학습에서 앙상블 방법론의 중요성을 강조합니다.



### Hardware Scaling Trends and Diminishing Returns in Large-Scale Distributed Training (https://arxiv.org/abs/2411.13055)
- **What's New**: 최근 몇 년간 신경망 모델의 성능이 급격히 증가했습니다. 이는 모델 크기, 훈련 데이터 및 계산 자원의 확대로 인해 가능해졌습니다. 이 연구는 하드웨어 구성 및 병렬화 전략을 신중하게 고려하는 것이 현대 응용 프로그램에 필요한 대규모 네트워크를 효과적으로 학습시키는 데 중요하다는 점을 강조합니다.

- **Technical Details**: 대규모 신경망 훈련에서 하드웨어 가속기를 효과적으로 활용하기 위한 몇 가지 알고리즘적 기술이 소개됩니다. 특히, 데이터 병렬성과 모델 병렬화 기법이 설명되며, 이러한 기술들이 어떻게 GPU 상에서의 데이터 분배 및 연산을 효율적으로 수행할 수 있는지에 대한 내용이 포함됩니다. 또한, 다양한 하드웨어 설정과 분산 병렬화 전략의 성능이 실증적으로 분석됩니다.

- **Performance Highlights**: 가속기의 수를 늘렸을 때 성과는 감소하는 경향이 있으며, 이는 자원 사용에 대한 효율성을 저하시킵니다. 연구 결과는 모델 병렬화가 전반적인 처리량을 개선할 수 있음을 보여주며, 향후 하드웨어 효율성의 향상은 네트워크 패브릭과 가속기 메모리 용량의 발전 없이는 제한적일 것이라는 것을 시사합니다.



### Probably Approximately Precision and Recall Learning (https://arxiv.org/abs/2411.13029)
- **What's New**: 본 연구는 추천 시스템 및 다중 라벨 학습과 같은 머신 러닝에서 필수적인 지표인 Precision과 Recall의 균형을 맞추는 새로운 PAC(Probably Approximately Correct) 학습 프레임워크를 제안합니다. 특히, 긍정적인 피드백만을 기반으로 학습해야 하는 현실적인 문제들에 대한 해결책을 제공합니다. 기존의 Empirical Risk Minimization 기법이 두 개의 가설로만 이루어진 간단한 경우에서도 실패하는 것을 보여주며, 새로운 알고리즘을 통해 Precision과 Recall 손실을 효과적으로 최소화합니다.

- **Technical Details**: 이 프레임워크는 그래프로 표현된 가설을 기반으로 하며, 엣지는 사용자와 아이템 간의 긍정적인 상호작용을 나타냅니다. 각 가설은 이진 또는 다중 클래스 PAC 학습 모델뿐만 아니라, 부분적 피드백을 갖는 다중 라벨 학습을 포함합니다. 실현 가능한 설정에서는 최적의 샘플 복잡성과 정확도를 보장하며, 비가 현실성 설정에서는 의미 있는 곱셈 근사를 얻을 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 알고리즘은 긍정적인 데이터만을 활용하여 학습하며, Precision 및 Recall 손실을 최소화합니다. 특히, 각 사용자의 기호에 대한 정밀하고 포괄적인 추천을 제공하기 위한 최적화된 방안을 제시합니다. 파레토 손실 목표를 통해 Precision과 Recall 간의 트레이드오프를 이해하고 관리하는 새로운 관점을 제공합니다.



### A Theory for Compressibility of Graph Transformers for Transductive Learning (https://arxiv.org/abs/2411.13028)
- **What's New**: 이 논문은 그래프에서의 전이적(Transductive) 작업 수행 시 기존의 Graph Transformer 모델의 숨겨진 차원을 압축할 수 있는 조건을 이론적으로 분석합니다. 이는 비교적 새로운 접근법으로, 기존의 연구들이 간과해온 면에 집중하고 있습니다. 또한, 다양한 상황에서 숨겨진 차원이 원래 네트워크의 출력과 주의(attention) 점수를 유지하면서 크게 압축될 수 있음을 보여줍니다.

- **Technical Details**: 그래프 구조에서의 메시지 전달(Message Passing) 작업을 통해, 이 연구는 숨겨진 차원(D)과 주의(edge)에 대한 이론적 경계를 제공합니다. 연구에서는 D가 그래프 크기(n)에 의존하여 변할 수 있으며, 이를 통해 계산 복잡도는 𝒪⁡(nD2 + mD)로 나타납니다. 즉, 원래 네트워크의 출력을 𝒪⁡(ϵ)의 오차 안에서 유지하면서 숨겨진 차원을 압축할 수 있는 방법을 제시하고 있습니다.

- **Performance Highlights**: 경험적 결과는 작은 네트워크가 대형 네트워크의 경쟁력 있는 성능을 가질 수 있음을 증명하며, 이는 실제 응용에 있어 실용적인 결과를 담고 있습니다. 기하학적 깊이 학습(Geometric Deep Learning) 및 자연어 처리(NLP)와 같은 다양한 분야에서도 Transformer 모델의 유용성을 보여주고 있습니다. 따라서, 이 연구는 앞으로의 Graph Transformer 모델의 효율성을 더욱 높일 수 있는 중요한 기초를 제공합니다.



### Scalable Deep Metric Learning on Attributed Graphs (https://arxiv.org/abs/2411.13014)
Comments:
          This is the complete version of a published paper with appendix including detailed proofs

- **What's New**: 이 논문에서는 여러 다운스트림 학습 작업을 지원하는 대규모 속성 그래프의 임베딩(embedding) 문제를 다룹니다. 특히, 속성 그래프에 적용할 수 있는 딥 메트릭 학습(deep metric learning) 기술의 확장을 통해 새로운 방법론을 제시합니다. 제안된 방법론에는 다중 클래스 튜플 손실(multi-class tuplet loss)을 기반으로 하는 두 가지 알고리즘, 즉 DMT와 DMAT-i가 포함되어 있습니다.

- **Technical Details**: 제안된 기법에서는 먼저 다중 긍정 샘플에 적용할 수 있는 확장된 다중 클래스 튜플 손실 함수(multi-class tuplet loss function)를 사용합니다. 또한, 일반화된 페이지랭크(Generalized PageRank)를 그래프 필터로 활용하여 더 컴팩트한 노드 표현을 도출하고, 배치 훈련(mini-batch training)을 통해 샘플링 비용을 제거합니다. 이러한 기본 프레임워크를 바탕으로 DMT와 DMAT-i라는 두 개의 알고리즘을 구축합니다.

- **Performance Highlights**: 대규모 실험을 통해 제안된 방법론이 노드 클러스터링(node clustering), 노드 분류(node classification), 링크 예측(link prediction) 같은 여러 다운스트림 작업에서 기존 방법들보다 더 높은 일관성을 보여줍니다. 특히, 여러 경우에서 스테이트 오브 더 아트(state-of-the-art) 성능을 달성했으며, 이는 제안된 방법들이 높은 확장성을 가지기 때문입니다.



### Deriving Activation Functions via Integration (https://arxiv.org/abs/2411.13010)
- **What's New**: 이 논문에서는 새로운 활성화 함수인 xIELU를 제안하며, 이는 ELU의 적분을 통해 도출된 기울기 기반 접근 방식을 사용합니다. xIELU는 긍정적인 입력에 대해 선형적으로 증가하는 기울기와 부정적인 입력에 대해 조정 가능한 기울기 흐름을 결합하여 ReLU$^2$와 xSiLU의 장점을 결합합니다. 이는 활성화 함수 설계의 새로운 관점을 제시하며, LLMs에서 기존의 활성화 함수들보다 높은 성능을 보여줍니다.

- **Technical Details**: xIELU는 ELU의 기울기를 기반으로 하여 trainable affine transformations의 적분을 통해 생성됩니다. 이 과정에서 positive gradient를 2로 곱하여 도출된 적분 표현을 단순화하며, ELU는 조각별로 정의되어 있어 긍정적 및 부정적 구성 요소를 별도로 적분하여 도출합니다. xIELU에서 사용하는 기울기는 기울기 흐름을 조정하는 trainable parameters인 αp와 αn에 의해 제어됩니다.

- **Performance Highlights**: xIELU는 1.1B 파라미터를 가진 Llama 모델에서 126B 토큰으로 훈련을 수행한 결과, 동일한 계산 비용 및 파라미터 수에 대해 ReLU$^2$와 SwiGLU보다 낮은 perplexity를 달성했습니다. 이는 xIELU가 트레이닝 도중 모델의 복잡성을 효과적으로 줄이는 가능성을 보여줍니다. 이로 인해 xIELU는 NLMs에서 기존의 활성화 함수들보다 우수한 성능을 입증하며, 학습할 수 있는 활성화 함수의 효용성을 강조합니다.



### LLMSteer: Improving Long-Context LLM Inference by Steering Attention on Reused Contexts (https://arxiv.org/abs/2411.13009)
- **What's New**: LLMSteer는 fine-tuning 없이 LLM의 생성 품질을 향상시키는 혁신적인 프레임워크입니다. 이 접근법은 쿼리 독립적인 attention steering을 통해 모델이 선택된 토큰에 대한 주의를 집중하도록 조정합니다. LLMSteer는 기존 방법들과 비교하여 성능 격차를 65.9% 좁히고 런타임 지연을 최대 4.8배 감소시키는 것으로 나타났습니다.

- **Technical Details**: LLMSteer는 두 번의 다른 prefix prompts를 사용하여 LLM이 문맥을 여러 번 읽도록 유도합니다. 이를 통해 주목할 가치가 있는 토큰을 선정하고 이들의 attention score를 재조정하여 모델이 문맥을 효과적으로 이해할 수 있도록 합니다. 이러한 방식은 기존의 KV 캐시를 재사용할 수 있게 하여 LLM의 효율성과 품질을 동시에 개선합니다.

- **Performance Highlights**: LLMSteer는 LLaMA-3.1-8b-Instruct에서 구현되었으며, F1 점수를 72.9에서 82.0으로 증가시키는 성과를 거두었습니다. 또한 기존의 최고 성능인 attention steering 방법과 비교했을 때, 최대 4.8배 더 빠른 처리 속도를 제공합니다. 이 연구는 fine-tuning 없이 높은 모델 생성 품질을 달성하는 최초의 시도로 큰 주목을 받고 있습니다.



### Evaluating LLMs Capabilities Towards Understanding Social Dynamics (https://arxiv.org/abs/2411.13008)
Comments:
          To appear in ASONAM 24 proceedings

- **What's New**: 이번 연구는 Generative Models(생성 모델)인 Llama와 ChatGPT가 소셜 미디어의 동적 언어 이해에 대해 얼마나 잘 작동하는지를 분석합니다. 특히 사이버 불링(cyberbullying)과 반사이버 불링(anti-cyberbullying) 메시지의 이해 능력을 평가합니다. 이 연구는 LLM(대형 언어 모델)이 소셜 상호작용의 방향성(directionality) 및 언어 이해에 있어 강점과 약점을 보여준다는 것을 발견했습니다.

- **Technical Details**: 연구의 평가 프레임워크는 LLMs의 다양한 기능 개선 기법을 고려합니다. 세 가지 주요 수준의 상호작용 – 언어, 방향성 및 사이버불링/반사이버불링 메시지의 탐지 – 를 이해하기 위해 LLM의 성능을 비교·분석하였습니다. 또한, LLM의 세부 조정(fine-tuning) 및 프롬프트 엔지니어링(prompt engineering)이 특정 작업에서 긍정적인 영향을 미칠 수 있음을 나타냅니다.

- **Performance Highlights**: 사이비 불링 이해와 방향성 추적에서 LLM들은 일부 매우 긍정적인 결과를 보였으나, 적절한 패러프레이징(paraphrasing) 및 사이버 불링 탐지 측면에서는 혼합된 결과를 나타냈습니다. 궁극적으로, 이러한 연구 결과는 LLM의 발전뿐만 아니라 디지털 환경에서 인간 상호작용의 이해에 기여할 수 있는 기초 자료로 작용할 것입니다.



### MERLOT: A Distilled LLM-based Mixture-of-Experts Framework for Scalable Encrypted Traffic Classification (https://arxiv.org/abs/2411.13004)
- **What's New**: 이번 연구에서는 MERLOT라는 새로운 프레임워크를 제안합니다. MERLOT는 암호화된 트래픽 분류를 위해 최적화된 압축된 대형 언어 모델의 스케일러블한 혼합 전문가(Mixture-of-Experts, MoE) 기반 refinement입니다. 이 모델은 GPT-2-base를 기반으로 하여, 머신러닝 및 딥러닝 접근법을 통해 트래픽 분류의 효율성과 정확도를 동시에 향상시킵니다.

- **Technical Details**: MERLOT는 교사-학생(paradigm) 훈련을 통해 고도로 압축된 모델을 생성하여, 각 전문 모델의 게이팅 메커니즘을 통해 동적으로 트래픽 분류 작업을 수행합니다. 기존의 생성 기반 기법과는 달리, MERLOT는 디코더의 마지막 토큰을 활용하여 직접적으로 암호화된 트래픽을 분류합니다. 또한, 메타데이터를 간결한 자연어 형태로 내장하여 입력 데이터의 표현을 강화합니다.

- **Performance Highlights**: MERLOT는 0.66억 매개변수로 이루어져 있으며, 7,777억 매개변수 기반의 TrafficLLM보다 경쟁력 있는 성능을 보입니다. 또한, MERLOT는 85-90%의 적은 추론 시간과 메모리 사용량을 소모하여 효율성과 정확성 간의 균형을 입증합니다. 이러한 성능은 암호화된 트래픽 분류 작업에서 MERLOT의 강력한 효과성을 증명합니다.



### Adaptive Process-Guided Learning: An Application in Predicting Lake DO Concentrations (https://arxiv.org/abs/2411.12973)
- **What's New**: 이 논문에서는 물리 모델과 순환 신경망(recurrent neural networks, RNN)을 통합한 새로운 프레임워크인 프로세스-가이드 학습(Process-Guided Learning, Pril)을 소개합니다. 이 프레임워크는 호수에서 용존 산소(dissolved oxygen, DO) 농도의 예측을 개선하여 수질과 생태계 건강을 유지하는 데 중요한 역할을 합니다. 기존 RNN의 한계를 극복하기 위해 DO의 차분 방정식을 포함하여 각 호수의 모델을 진화시키는 접근 방식을 사용합니다.

- **Technical Details**: Pril 프레임워크는 각 호수 레이어에 대한 DO 동역학을 규명하는 일반적인 일변수 차분 방정식을 회귀모델에 통합하여 학습 과정의 손실 함수를 일반화합니다. 특히, DO는 양의 힘의 종류에 따라 상대적으로 처리가 이루어지며, 이를 통해 에너지 보존의 원칙과 질량 보존의 원칙을 잘 지킵니다. 하지만 이 방법은 수치적 불안정성(numerical instability)에 민감하며, 특히 극단적인 변동이 클 경우 계산이 불안정해질 수 있는 문제를 가지고 있습니다.

- **Performance Highlights**: 제안된 Pril과 이를 개선한 적응형 프로세스-가이드 학습(Adaptive Process-Guided Learning, April) 모델은 중서부 미국의 다양한 호수에서 검증되었으며, 제한된 학습 데이터에도 불구하고 안정적으로 DO 농도를 예측할 수 있는 능력을 보여주었습니다. April 모델은 시간이 지남에 따라 조절 가능한 타임스텝을 통해 해양 생태계에서의 DO 농도 변화를 효과적으로 관리하여, 물리적 일관성을 유지함으로써 예측 성능을 향상시키는 데 기여합니다.



### A Foundation Model for Unified Urban Spatio-Temporal Flow Prediction (https://arxiv.org/abs/2411.12972)
- **What's New**: 이 논문에서는 UniFlow라는 일반 도시 흐름 예측 모델을 소개합니다. UniFlow는 grid-based(그리드 기반) 데이터와 graph-based(그래프 기반) 데이터를 통합하는 기초 모델로, 서로 다른 형태의 데이터를 일관된 순차 자료로 변환하는 multi-view spatio-temporal patching(다중 시점 시공간 패칭) 메커니즘을 설계했습니다. 또한, 복잡한 상관관계와 역학을 포착하기 위한 spatio-temporal transformer(시공간 변환기) 아키텍처를 도입하였습니다.

- **Technical Details**: UniFlow는 Spatio-Temporal Memory Retrieval Augmentation (ST-MRA) 기법을 통해 서로 다른 데이터 유형 간의 공유 시공간 패턴을 활용하여 모델의 예측력을 향상시킵니다. 이를 위해 구조화된 메모리 모듈을 생성하여 시공간 패턴을 저장하고, 동적으로 메모리와 상호작용하면서 예측의 질을 높입니다. ST-MRA는 모든 프로세스가 학습 가능하도록 설계되어, 기존의 비매개변수적 Retrieval-Augmented Generation 방식과 차별화됩니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 광범위한 실험은 UniFlow가 기존 모델보다 뛰어난 성능을 보여 주며, 특히 제한된 데이터 환경에서 우수한 성능을 나타냅니다. 이 모델은 평균 10% 이상의 성능 향상을 달성하며, 다양한 도시 흐름 예측 작업에 응용될 수 있는 잠재력을 보여줍니다. 또한, UniFlow는 특정 데이터 유형에 맞춘 개별 모델 교육의 필요성을 없애고, 일반화 가능성 높은 예측 모델을 제시합니다.



### Machine learned reconstruction of tsunami dynamics from sparse observations (https://arxiv.org/abs/2411.12948)
- **What's New**: 본 논문에서는 Senseiver라는 트랜스포머 신경망을 활용하여 제자리에 있는 닻(위치)에서 얻은 희소한(tsunami) 데이터를 이용해 해면의 높이를 추정하는 방법을 제시합니다. 이를 통해 대규모 시뮬레이션 데이터를 기반으로 재해의 정확한 모델링을 가능하게 하며, 기존의 기존 물리 모델을 통합하는 데 필요한 계산 비용과 시간을 절약할 수 있습니다. 특히 본 연구는 일본 근처 해안 지역에서 발생한 쓰나미에 집중하며 교육 과정에서 사용되지 않은 새로운 데이터 세트에 대한 테스트를 포함하여 구조의 유연성을 활용하는 방안을 제시합니다.

- **Technical Details**: 저자들은 DART 네트워크를 통해 수집된 극히 제한적인 측정값을 기반으로, 트랜스포머 기반 신경망 시스템인 Senseiver를 활용하여 쓰나미의 파고(field height)와 진행 양상을 예측하려고 합니다. 이 모델은 얕은 수역 방정식(s shallow water equations) 솔버에 의해 생성된 대규모 시뮬레이션 데이터로 학습되며, 이를 통해 역사적 재해와의 비교도 가능합니다. 이러한 방식으로 감지된 파고 데이터는 트랜스포머 네트워크를 통해 고도로 상세한 쓰나미 데이터 생성이 가능하다는 점이 강조됩니다.

- **Performance Highlights**: Senseiver 모델은 최소 한 곳의 센서에서 유의미한 신호를 수신했을 때, 세부적인 위상 및 진폭 특징을 정확히 복원할 수 있는 능력을 보여줍니다. 특히, 이 논문에서는 센서의 배치 최적화와 다양한 원거리 감지 데이터를 적용할 가능성을 탐구하며, 전체적인 예측 시스템 개선 방안을 제시합니다. 실험 결과는 Senseiver가 희소한 데이터를 바탕으로 정확하게 파고를 재구성하는 데 있어 상당한 성과를 나타냈음을 보여줍니다.



### LEDRO: LLM-Enhanced Design Space Reduction and Optimization for Analog Circuits (https://arxiv.org/abs/2411.12930)
- **What's New**: 이번 논문에서는 LEDRO라는 새로운 프레임워크를 소개합니다. LEDRO는 대형 언어 모델(LLM)을 최적화 기법과 결합하여 아날로그 회로 설계 공간을 반복적으로 정제하는 접근 방식을 사용합니다. 이는 기존의 RL과 BO 방법보다 일반화 가능성이 높아 여러 토폴로지와 기술 노드에서 별도의 디자인 주석이나 모델 학습 없이도 적용할 수 있습니다. LEDRO는 22개의 다양한 Op-Amp 토폴로지를 4개의 FinFET 기술 노드에서 평가하여 우수한 성능을 입증했습니다.

- **Technical Details**: LEDRO 프레임워크는 LLaMa3-70B라는 대형 언어 모델과 TuRBO 최적화 기법을 활용합니다. 이 프레임워크는 고정된 횟수만큼 반복되는 방식으로 회로 시뮬레이터를 통해 아날로그 디자인의 성능을 평가하고, 디자인을 지속적으로 탐색하고 개선합니다. 초기 프롬프트를 조정하여 LLM이 설계 공간을 효율적으로 이해하도록 돕고, 적절한 참조 디자인을 제공하여 탐색 공간을 더 개선합니다. FoM(Figure of Merit)을 사용하여 각 configuration의 성능을 정량적으로 평가합니다.

- **Performance Highlights**: 실험 결과, LEDRO는 저복잡도 Op-Amps에서 평균 13% FoM 향상과 2.15배의 속도 향상을 보였고, 고복잡도 Op-Amps에서는 48% FoM 향상과 1.7배의 속도 향상을 기록했습니다. 이러한 성과는 LEDRO가 탐색 공간을 더 관련성 있는 범위로 정제하는 능력을 보여줍니다. LEDRO는 다양한 최적화 방법과 함께 사용할 수 있는 유연성 또한 제공합니다. 따라서 LEDRO는 아날로그 회로 설계를 위한 효과적이고 효율적인 프레임워크로 자리매김할 것으로 기대됩니다.



### Loss-to-Loss Prediction: Scaling Laws for All Datasets (https://arxiv.org/abs/2411.12925)
- **What's New**: 이번 연구는 다른 데이터 분포에서의 손실 예측을 위한 새로운 방법론인 loss-to-loss prediction을 제시합니다. 기존의 scaling laws는 단일 데이터 분포의 훈련 손실 예측에 유용하지만, 이 연구는 다양한 pre-training 데이터셋 간 그리고 모델 간의 손실 관계를 탐구합니다. 또한, 연구 결과는 모델이 훈련된 데이터셋 간의 손실 변환을 통해 scaling laws를 효과적으로 활용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 논문에서는 세 가지 유형의 loss-to-loss 관계를 살펴봅니다. 첫 번째는 train-to-train 관계로, 서로 다른 데이터셋에서 훈련된 모델 간의 손실을 비교합니다. 두 번째는 train-to-test 관계로, 한 데이터셋에서 훈련된 모델의 성능을 다른 데이터셋에서 평가하는 방식입니다. 마지막으로 test-to-test 관계는 서로 다른 데이터셋에서 훈련된 모델의 테스트 손실을 비교합니다. 이 모든 관계에서 shifted power law가 관찰되었습니다.

- **Performance Highlights**: 연구 결과는 다양한 데이터셋과 다운스트림 작업에서 예측 가능한 손실 간의 관계를 입증했습니다. 특히, 적은 수의 샘플로부터 새로운 데이터셋에 대한 만약의 성능을 예측할 수 있는 전이 성능이 향상되었음을 보여줍니다. 또한, 여러 쌍의 데이터셋을 사용하면 새로운 데이터셋에서의 성능 예측이 더 정확해질 수 있음을 발견했습니다.



### Trojan Cleansing with Neural Collaps (https://arxiv.org/abs/2411.12914)
- **What's New**: 이번 연구에서는 Trojan attack이 Neural Collapse라는 최근 연구된 심층 학습 이론의 현상과 연결되어 있음을 밝혔습니다. Trojan attack은 입력 데이터에 특정 트리거가 포함될 때 특정 출력을 유도하는 신경망에 대한 훈련 시간 공격입니다. 연구팀은 Trojan attack이 Neural Collapse의 수렴을 방해한다는 실험적 증거를 제공하며, 이 방해를 이용하여 다양한 신경망 아키텍처에서 Trojan attack을 정화하는 경량의 보편적 메커니즘을 설계했습니다.

- **Technical Details**: Trojan attack은 데이터 오염(data poisoning) 기법을 통해 이루어지며, 적대자가 선택한 트리거를 소량의 훈련 데이터에 추가하여 훈련 과정을 조작합니다. 연구에서는 Neural Collapse(NC) 현상에 대해 깊이 있는 분석을 시도하며, NC는 과도한 파라미터를 가진 신경망의 특성적인 행동으로 훈련 데이터가 높은 대칭적인 기하학적 구조로 수렴하는 과정을 기술합니다. Trojan attack은 이러한 대칭성을 저해하는 비대칭적인 특성을 가지며, 이는 연구의 핵심 동기를 제공하였습니다.

- **Performance Highlights**: 연구팀은 제안된 정화 방법을 다양한 네트워크 아키텍처와 전형적인 Trojan attack에 대해 실험적으로 검증하였으며, 실험 결과 정화된 신경망이 원래 모델의 정확도를 유지하면서도 Trojan trigger에 대해 저항력을 보임을 확인하였습니다. 연구에서 제안된 방법은 ResNet 아키텍처에 대해 일반적인 데이터 오염 공격에 대해 경쟁력 있는 성능을 보여주며, 복잡한 Trojan attack 및 대형 트랜스포머 아키텍처에 대해서는 최신 성능을 기록하였습니다. 또한 이 방법은 데이터 불균형 및 부패에 대해 특히 강건한 특성을 가집니다.



### MLDGG: Meta-Learning for Domain Generalization on Graphs (https://arxiv.org/abs/2411.12913)
Comments:
          Accepted in KDD 2025 (research track)

- **What's New**: 본 논문에서는 MLDGG라는 새로운 크로스-멀티 도메인 메타 학습 프레임워크를 제안합니다. 이 프레임워크는 소스 도메인에서 샘플링된 그래프에서 이전 지식을 습득하고, 이러한 지식을 바탕으로 훈련 중 접근할 수 없는 타겟 도메인에 일반화하는 것을 목표로 합니다. 여기에는 구조 학습과 의미 식별을 통해 도메인 간의 적응력을 높이는 방식이 포함되어 있습니다.

- **Technical Details**: MLDGG는 두 가지 주요 구성 요소로 이루어져 있습니다: 구조 학습자(structure learner)와 표현 학습자(representation learner). 구조 학습자는 관련 없는 엣지의 부정적 영향을 줄이며, 다양한 도메인 간의 공유 구조 지식을 포착하여 GNN이 학습한 표현의 종합성을 향상시킵니다. 표현 학습자는 노드 임베딩에서 도메인 불변 의미와 도메인 특화 변동 정보를 분리하여 일반화 능력을 높입니다.

- **Performance Highlights**: 실험 결과, MLDGG는 세 가지 다양한 분포 이동(distribution shift) 환경에서 기존의 최첨단 방법들을 초월하는 성능을 보여주었습니다. 이는 MLDGG가 노드 예측 과제에 대해 제공되는 다양한 도메인에서 강력한 일반화 능력을 유지함을 입증합니다. 이러한 결과는 제안된 메타 학습 접근 방식의 효과성을 강조합니다.



### Tree Species Classification using Machine Learning and 3D Tomographic SAR -- a case study in Northern Europ (https://arxiv.org/abs/2411.12897)
- **What's New**: 이번 연구에서는 Synthetic Aperture Radar (SAR) 데이터를 기반으로 한 3D 단층 이미지를 활용하여 나무종 분류의 정확성을 향상시키기 위한 새로운 접근 방식을 소개합니다. TomoSense 데이터셋을 사용하여 여러 탭형 머신러닝 모델을 평가하고 Bayesian 최적화를 통해 성능을 극대화했습니다. 특히, LiDAR(point cloud data)의 데이터를 활용하여 모델의 예측과 관련된 나무 높이 통계를 제시함으로써 단층 데이터의 신뢰성을 평가했습니다.

- **Technical Details**: 연구에서는 다양한 폴라리메트릭 설정과 지리적 분할을 활용하여 SAR 데이터의 분석을 수행했습니다. TomoSense 데이터셋은 2미터의 공간 해상도를 가지며, 개별 나무 높이에 대한 정보와 함께 8종의 나무를 분류하기 위한 기반 데이터를 제공합니다. AutoGluon을 사용하여 머신러닝 모델의 튜닝 및 최적화 과정을 간소화하고, 여러 모델(예: Gradient Boosting Machines, CNNs, Random Forests)을 평가하여 성능을 극대화하는 데 집중했습니다.

- **Performance Highlights**: 모델 최적화 과정을 통해 다양한 탭형 머신러닝 모델이 높은 정확도를 기록했으며, 이는 TomoSense의 3D 단층 이미지를 성공적으로 활용했음을 보여줍니다. 연구 결과, SAR 데이터와 LiDAR 정보를 결합하여 나무 종 분류의 신뢰성을 크게 향상시켰고, 이는 지속 가능한 산림 관리 및 생물 다양성 평가에 기여할 수 있는 잠재력을 가지고 있습니다. 또한, 새로운 데이터 처리 파이프라인은 원거리 탐사에서의 공간 자율 상관관계 문제를 해결하기 위한 두 가지 데이터 분할 전략을 도입함으로써 유의미한 결과를 도출했습니다.



### Selective Attention: Enhancing Transformer through Principled Context Contro (https://arxiv.org/abs/2411.12892)
- **What's New**: 이번 연구에서는 기존의 self-attention 메커니즘이 쿼리에 대해 균일하게 처리하는 방식이 문맥의 희소성과 적합성을 제어하는 능력을 저해한다고 주장합니다. 따라서, Selective Self-Attention (SSA) 레이어를 도입하여 소프트맥스 비선형성을 열 조절 방식을 통해 보강합니다. 이 방법을 통해 쿼리 임베딩과 컨텍스트 윈도우에서의 위치에 따라 문맥의 희소성을 조절할 수 있으며, 결과적으로 주의력 희석을 완화하고 전반적인 모델 성능을 향상시킵니다.

- **Technical Details**: Selective Self-Attention (SSA) 레이어는 쿼리 및 값 임베딩에 온도 조절(Temperature Scaling, TS)을 적용하며, 이를 통해 문맥의 희소성과 의미적 유사성을 분리합니다. SSA는 쿼리 임베딩에 대해 계산된 온도를 바탕으로 소프트맥스를 적용하고, 하이퍼파라미터를 조정을 통해 원활한 학습이 가능합니다. 이론적으로는, SSA가 더 적은 파라미터로도 목표 주의 맵을 표현할 수 있게 돕는다는 점을 증명했습니다.

- **Performance Highlights**: SSA를 적용한 모델은 Wikitext, Lambada, Piqa 등의 언어 모델링 벤치마크에서 눈에 띄고 일관된 정확도 개선을 이루었습니다. 이 접근 방식은 GPT-2, Pythia, Llama 등 다양한 모델에서 테스트되었으며, fine-tuning과 pre-training 과정 모두에서 긍정적인 결과를 보였습니다. 추가로, SSA는 transformer의 검색 능력을 극대화하며, 실험 결과가 이를 확인해줍니다.



### Puppet-CNN: Input-Adaptive Convolutional Neural Networks with Model Compression using Ordinary Differential Equation (https://arxiv.org/abs/2411.12876)
- **What's New**: 이 논문에서는 CNN의 전통적인 한계를 극복하기 위해 새로운 CNN 프레임워크인 Puppet-CNN을 제안합니다. 이 프레임워크는 puppet 모듈과 puppeteer 모듈의 두 가지 모듈로 구성되어 있으며, puppeteer 모듈이 입력 데이터의 복잡성에 따라 커널을 생성합니다. 이는 네트워크의 깊이에 따라 동적으로 커널 파라미터를 조정하여 효율성과 성능을 획기적으로 개선합니다.

- **Technical Details**: Puppet-CNN의 핵심은 Ordinary Differential Equation (ODE)을 사용하여 puppeteer 모듈이 puppet 모듈의 커널 파라미터를 생성하는 것입니다. 이렇게 하면 더 많은 파라미터를 저장할 필요 없이 작고 효율적인 puppeteer ODE 모듈을 통해 깊은 puppet 네트워크를 최적화할 수 있습니다. 다양한 데이터 샘플에 대해 커널과 깊이를 다르게 설정할 수 있어 유연성이 높아지는 장점이 있습니다.

- **Performance Highlights**: 실험 결과, Puppet-CNN은 전통적인 CNN보다 성능과 효율성 모두에서 우수한 결과를 보여줍니다. 모델 크기는 10배 이상 줄일 수 있으며, 이는 다양한 데이터 복잡성을 처리하는 데 있어 훨씬 효과적인 접근법을 제공합니다. 이는 특히 많은 파라미터를 요구하는 전통적인 CNN의 한계를 극복하는 데 도움을 줍니다.



### Tensor-Based Foundations of Ordinary Least Squares and Neural Network Regression Models (https://arxiv.org/abs/2411.12873)
Comments:
          16 pages, 3 algorithms

- **What's New**: 이번 논문은 전통적인 방법과 다른 새로운 수학적 접근을 통해 Ordinary Least Squares(OLS)와 Neural Network 회귀 모델을 개발합니다. Tensor Analysis와 기본적인 행렬 계산을 활용하여 이들 모델의 이론적 기초를 자세히 설명하고, 완전한 알고리즘 형태로 확장합니다.

- **Technical Details**: 본 논문에서는 함수, 벡터, 텐서 및 행렬과 같은 여러 수학적 개념을 다루고 있습니다. 독자는 선형 대수학(Linear Algebra)과 미적분학(Calculus)의 기본 및 중급 내용을 숙지하고 있어야 하며, 이 글에서는 주로 지도 학습(supervised learning) 문제를 탐구하고 있습니다. 지도 학습 문제는 기능(features)과 레이블(labels)로 구성되며, 분류(classification) 및 회귀(regression) 문제로 나뉩니다.

- **Performance Highlights**: 이 연구에서는 Backpropagation 알고리즘의 간소화된 버전을 포함하여 세 가지 알고리즘을 제시하고 있습니다. 새로운 수학적 접근 방식의 이점을 보여주는 실험 결과가 포함되어 있어, 기존 방법론과의 성능 차이를 입증하고 있습니다. 이는 신경망 모델의 학습 효율성을 높이고, 연구자들에게 새로운 통찰력을 제공할 것으로 기대됩니다.



### CDI: Copyrighted Data Identification in Diffusion Models (https://arxiv.org/abs/2411.12858)
Comments:
          Coda available at this https URL

- **What's New**: 최근 Diffusion Models (DMs)가 새로운 생성 모델의 일종으로 주목받고 있으며, 기존의 Generative Adversarial Networks (GANs)를 능가하는 성능을 보이고 있습니다. 이 모델들은 다양한 고품질 데이터로 훈련되어야 하나, 이 데이터는 종종 소유자의 저작권을 무시하고 인터넷에서 스크랩됩니다. 저작권 침해 문제와 관련하여, Getty Images는 Stability AI를 상대로 소송을 제기했으며, 이는 생성 AI 기업에 대한 추가 소송의 물결을 촉발했습니다.

- **Technical Details**: 기존의 Membership Inference Attacks (MIAs)는 대규모 DMs의 훈련 데이터 포인트를 식별하는 데 효과적이지 않다는 것을 발견했습니다. 기존 MIA는 강력한 신호를 생성하지 못하고, MIAs를 통해 생성된 신호의 신뢰성을 보장하기 위한 방법이 부족합니다. 이를 극복하기 위해, 새로운 프레임워크인 CDI를 제안하여 데이터 소유자들이 자신의 데이터가 DM 훈련에 사용되었는지 확인할 수 있도록 지원합니다.

- **Performance Highlights**: CDI는 데이터 소유자가 70개의 데이터 포인트로도 99% 이상의 신뢰도로 자신의 데이터 사용 여부를 식별할 수 있도록 해줍니다. 이 방법은 다양한 대규모 DM 아키텍처에서 실효성을 입증하였으며, 잘못된 긍정 결과를 생성하지 않아 저작권 데이터 사용을 신뢰성 있게 탐지할 수 있는 도구로 자리잡고 있습니다. 또한, CDI는 통계적 테스트를 통해 결과의 신뢰성을 강화하여, 데이터 소유자들에게 매우 유용한 도구가 될 것입니다.



### Integrating Secondary Structures Information into Triangular Spatial Relationships (TSR) for Advanced Protein Classification (https://arxiv.org/abs/2411.12853)
- **What's New**: 이 연구에서는 Triangular Spatial Relationship (TSR) 접근 방식을 기반으로 한 새로운 방법인 SSE-TSR을 개발했습니다. 이 방법은 단백질 구조의 세부적인 이해를 위해 18가지의 서로 다른 나선, 스트랜드, 그리고 코일 조합을 통합하여 단백질 구조 표현을 향상시킵니다. SSE 통합을 통해 단백질 분류의 정확성과 신뢰성을 개선할 수 있음을 보여주었습니다. 두 개의 대규모 단백질 데이터셋에서 SSE-TSR 방법을 적용한 결과, 분류 정확도가 크게 향상되었습니다.

- **Technical Details**: 단백질 구조 분석은 생물학적 기능과 상호작용을 이해하는 데 매우 중요합니다. 기존의 구조 비교 방법은 주로 서열 정렬과 3D 구조 중첩에 의존하여 세부적인 유사성을 발견하는 데 한계가 있었습니다. SSE-TSR 방법은 Cα 원자를 정점으로 하는 삼각형을 생성하고, 길이, 각도 및 정점 레이블 기반의 고유 키를 할당하여 단백질 구조를 효과적으로 표현합니다. 이 방식은 단백질의 3D 구조적 특징과 SSE 정보를 동시에 고려하여, 보다 정교한 구조 분석을 가능하게 합니다.

- **Performance Highlights**: SSE-TSR 방법은 두 개의 대규모 데이터셋에서 단백질 분류 성능을 크게 향상했습니다. 데이터셋 1에서는 정확도가 96.0%에서 98.3%로 증가했으며, 데이터셋 2에서도 이미 높은 정확도에서 소폭의 개선이 있었습니다. 이러한 결과는 초기 정확도가 낮은 데이터셋에서는 SSE 통합이 큰 도움이 되며, 높은 성능을 가진 데이터셋에서는 점진적인 발전을 가져올 수 있음을 보여줍니다. 결과적으로 SSE-TSR은 단백질 구조 분석 및 기능 이해를 위한 강력한 생물정보학 도구입니다.



### mDAE : modified Denoising AutoEncoder for missing data imputation (https://arxiv.org/abs/2411.12847)
- **What's New**: 이번 논문에서는 Denoising AutoEncoder (DAE)를 기반으로 한 새로운 기법인 mDAE를 제안합니다. 이 방법은 손실 함수(loss function)의 수정과 하이퍼파라미터(hyper-parameter) 선택 절차를 간소화하여 결측치(imputation) 문제를 해결합니다. 여러 UCI Machine Learning Repository 데이터셋에 대한 연구를 통해 mDAE 방법이 Root Mean Squared Error (RMSE) 측면에서 향상된 성능을 보여주는 것을 확인했습니다.

- **Technical Details**: mDAE는 결측치를 수치적 데이터에서 보완하는 데 특화된 Denoising AutoEncoder의 수정형입니다. AutoEncoder는 레이블이 없는 데이터를 효율적으로 표현하고 복원하는 인공 신경망입니다. 원래 DAE는 노이즈가 있는 데이터로부터 원본 데이터를 회복하도록 설계되었으며, 본 논문에서는 결측치를 노이즈로 간주하여 이를 처리하는 방식을 제안합니다.

- **Performance Highlights**: 제안된 mDAE 방법은 8가지 다른 imputation 방법과 비교했을 때, Mean Distance to Best (MDB) 기준에서 지속적으로 상위에 위치하였으며, 특히 SoftImput과 missForest와 함께 최고 방법으로 평가되었습니다. 이 연구를 통해 mDAE가 주어진 데이터셋에서 효과적인 결측치 보완 방법임을 입증했습니다. 결과는 GitHub를 통해 재현 가능하고 다른 데이터셋과 방법으로 일반화 가능합니다.



### Reward Modeling with Ordinal Feedback: Wisdom of the Crowd (https://arxiv.org/abs/2411.12843)
- **What's New**: 본 논문에서는 인간의 선호도를 반영하여 보상 모델을 학습하는 새로운 프레임워크를 제안합니다. 기존의 이진 피드백을 넘어, 다양한 수준의 순서형 피드백(ordinal feedback)을 활용하여 데이터로부터 중요한 정보를 더 효과적으로 추출할 수 있습니다. 특히 이 연구는 선호 간의 미세한 차이를 반영하는 방법을 제시합니다.

- **Technical Details**: 제안된 프레임워크에서는 인간의 선호 피드백을 특정 응답이 다른 응답보다 더 낫다는 확률과 연결짓습니다. 우리는 'wisdom of the crowd'라는 사회학적 개념을 기반으로 한 마진 편향 제로(marginal unbiasedness) 조건을 도입하여 이론적 근거를 마련합니다. 또한, 우리는 이 순서형 피드백 모델의 통계적 이점을 검증하며, Rademacher 복잡성을 줄일 수 있는 방법을 제시합니다.

- **Performance Highlights**: 수치 실험의 결과, 미세한 피드백을 포함한 경우 보상 모델 학습이 개선된다는 것을 확인하였습니다. 4개의 서로 다른 순서형 피드백 시스템을 설정하여 이론적 발견을 검증하였고, 특히 특정 비율의 동점(tied) 샘플을 포함하는 것이 RM 학습을 촉진하는 것으로 나타났습니다. 이러한 결과는 보상 모델의 성능을 크게 향상시킬 가능성을 보여줍니다.



### Generalized Prompt Tuning: Adapting Frozen Univariate Time Series Foundation Models for Multivariate Healthcare Time Series (https://arxiv.org/abs/2411.12824)
Comments:
          Machine Learning for Health (ML4H 2024)

- **What's New**: 이번 논문에서는 기존의 단변량(univariate) 시계열 기초 모델을 다변량(multivariate) 시계열 예측에 적응시키기 위한 새로운 기술인 일반화된 프롬프트 튜닝(Generalized Prompt Tuning, Gen-P-Tuning)을 제안합니다. 이를 통해 서로 다른 변수들 간의 정보를 결합하여 다변량 시계열 예측이 가능하도록 하였습니다. 저자들은 이러한 기법이 기존의 다양한 모델과 비교하여 효과적임을 보여주며, 의료 데이터셋에 대한 튜닝 전략을 벤치마킹한 첫 번째 연구로 자리매김합니다.

- **Technical Details**: 논문에서는 단변량 시계열 데이터를 어떻게 다변량 시계열 예측으로 전환할 수 있는지를 자세히 설명합니다. Gen-P-Tuning은 기존의 프롬프트 튜닝 기법을 일반화한 것으로, 단변량 시계열 기초 모델을 고정된 상태로 두고 이를 다변량 예측에 적합하도록 하는 효율적인 파라미터 조정(parameter-efficient fine-tuning, PEFT) 방법입니다. 그리고 이 기술을 사용하여 MIMIC 데이터에서의 사망 예측 및 인플루엔자 유사 질병 예측을 위한 실험을 수행합니다.

- **Performance Highlights**: 실험 결과 Gen-P-Tuning은 다양한 파인튜닝(base tuning) 기준에 대해 경쟁력 있는 성능을 보여주었고, 특히 MIMIC 분류 작업에서 STraTS와 같은 기존의 비파인튜닝 방법과도 비교 분석하였습니다. 이 기술은 의료 데이터셋에서 다변량 예측을 수행하는 데 있어 중요한 전환점을 제공하며, 향후 관련 연구에 큰 기여를 할 것으로 기대됩니다.



### AI-generated Image Detection: Passive or Watermark? (https://arxiv.org/abs/2411.13553)
- **What's New**: 이 논문에서는 AI가 생성한 이미지를 탐지하기 위한 최초의 포괄적인 벤치마크인 ImageDetectBench를 제안합니다. 이 벤치마크는 수동 탐지기와 워터마크 기반 탐지기 간의 효과성, 견고성, 효율성을 비교하기 위해 설계되었습니다. 연구팀은 AI가 생성한 이미지와 비-AI가 생성한 이미지를 포함한 네 가지 데이터 세트를 구성하고, 다양한 왜곡(perturbations)과 공격에 대해 탐지기의 성능을 평가하였습니다.

- **Technical Details**: ImageDetectBench는 8가지 일반적인 왜곡과 3가지 적대적 왜곡을 포함하여, 수동 탐지기와 워터마크 기반 탐지기의 비교를 위한 체계적인 실험을 실시합니다. 연구에서는 5개의 수동 탐지기와 4개의 워터마크 기반 탐지기를 평가하였으며, 각 탐지기의 설계는 이미지 간의 미세한 아티팩트를 탐지하는 데 중점을 두고 있습니다. 또한, 워터마크 기반 탐지기는 AI가 생성한 이미지를 구별하기 위해 이미지 생성 시 인간이 인지할 수 없는 아티팩트를 삽입하는 방식으로 작동합니다.

- **Performance Highlights**: 연구 결과, 워터마크 기반 탐지기가 모든 시나리오에서 수동 탐지기보다 일관되게 우수한 성능을 보였으며, 특히 왜곡이 없거나 일반적인 왜곡, 적대적 왜곡이 있는 경우에서도 그 우수함을 유지하였습니다. 가장 견고한 워터마크 기반 탐지기는 가장 견고한 수동 탐지기보다도 두 배 이상 빠른 속도로 작동하여, 일반 사용자와 공격자 모두에게 효과적으로 대응할 수 있는 장점을 보여주었습니다. 따라서 두 종류의 탐지기가 모두 적용 가능한 경우에는 워터마크 기반 탐지기를 우선적으로 사용하는 것이 권장됩니다.



### HF-Diff: High-Frequency Perceptual Loss and Distribution Matching for One-Step Diffusion-Based Image Super-Resolution (https://arxiv.org/abs/2411.13548)
Comments:
          8 pages

- **What's New**: 최근의 확산 기반 (diffusion-based) 단일 단계 슈퍼 해상도 (super-resolution) 방법들이 SinSR에 비해 더 나은 성능을 보이지만, 계산 복잡성이 있다는 점을 고려하여, 본 연구에서는 고주파 세부 정보( high-frequency detail features)를 보존하면서 성능을 개선하는 방안을 제시합니다. 이를 위해, ImageNet 데이터셋에서 사전 훈련된 가역 신경망 (invertible neural network; INN)을 활용하여 고주파 지각 손실 (perceptual loss)을 도입했습니다. 슈퍼 해상도 이미지와 정답 이미지(ground truth; GT) 간의 고주파 특징을 보존하는 것과 함께 Jenson-Shannon 발산(Jensen-Shannon divergence)을 활용하여 분포를 일치시키는 접근법을 적용했습니다.

- **Technical Details**: 본 연구에서는 단일 단계 확산 기반 슈퍼 해상도 알고리즘에서 고주파 보존 손실을 도입하여 세밀한 세부 정보를 보존하도록 설계되었습니다. INN을 활용하여 ImageNet-1K 데이터셋에서 훈련을 진행하였으며, 이를 통해 생성된 SR 이미지와 해당 GT 이미지 간의 고주파 지각 손실을 계산했습니다. 이 외에도 DINO-v2 임베딩 공간에서 GT 이미지와 SR 이미지의 분포를 최소화하는 Jenson-Shannon 발산을 이용해 분포를 정렬하는 방안을 마련했습니다.

- **Performance Highlights**: 본 슈퍼 해상도 알고리즘은 RealSet65, RealSR, ImageNet, DIV2K-Val, DrealSR 데이터셋에서 최첨단 CLIPIQA 점수를 획득하였으며, DIV2K-Val와 RealSR 데이터셋에서 OSEDiff 알고리즘보다 더 뛰어난 성과를 보였습니다. 본 연구의 방법론은 최근의 다른 SR 접근법들에 비해 질적으로도 개선된 시각적 표현을 제공합니다. 이러한 성과들은 고주파 보존 손실과 분포 정렬이 SR 이미지 품질에 긍정적인 영향을 미친다는 것을 뒷받침합니다.



### Identity Preserving 3D Head Stylization with Multiview Score Distillation (https://arxiv.org/abs/2411.13536)
Comments:
this https URL

- **What's New**: 본 논문에서는 PanoHead 모델을 활용하여 360도 시점에서 이미지를 합성하여 3D 헤드 스타일화 (3D head stylization) 문제를 다루고 있습니다. 기존의 3D 스타일화 방법들은 주로 근전방 (near-frontal) 뷰에서 합성되며, 원본 이미지의 고유한 정체성 (identity) 유지에 어려움이 있는 반면, 이번 연구는 이러한 한계를 극복하고자 합니다. 제안된 프레임워크는 negative log-likelihood distillation (LD)을 통해 정체성 보존과 스타일화 품질을 향상시킵니다.

- **Technical Details**: 연구에서 제안하는 방법은 PanoHead의 사전 훈련된 매개 변수를 미세 조정하여 다양한 도메인에서 이미지를 생성합니다. LD 기법을 3D 인식 이미지 생성기 (3D-aware image generators)에 적용하며, SDS와의 차이점을 설명하고, 교차 포즈 의존성 (cross-pose dependencies) 및 그리드 노이즈 제거 (grid denoising)를 추가하여 스타일화 품질을 향상시키는 방법을 제시합니다. 성능을 높이기 위해 점수 텐서에 대한 순위 감소 (rank reduction)를 사용하며, 이는 스타일 기법에도 긍정적인 영향을 미칩니다.

- **Performance Highlights**: 제안한 방법은 관련 있는 헤드 스타일화 방법들에 비해 질적 및 양적 차원이 상당한 개선을 나타냅니다. LD를 사용하여 더욱 선명하고 ID 보존이 우수한 결과를 얻으며, 다각적 그리드 및 미러 점수 기울기를 통합하여 스타일화 품질을 더욱 개선합니다. 이 연구는 3D 헤드 스타일화의 발전뿐 아니라, GAN을 통한 효과적인 증류 (distillation) 과정에 대한 중요한 통찰을 제공합니다.



### Quantum Attention for Vision Transformers in High Energy Physics (https://arxiv.org/abs/2411.13520)
Comments:
          9 pages, 7 figures

- **What's New**: 이 논문에서는 양자 직교 신경망(Quantum Orthogonal Neural Networks, QONNs)을 통합한 새로운 하이브리드 양자-고전적 비전 변환기 아키텍처를 제안합니다. 이 접근 방식은 과거 모델의 한계를 극복하고 고차원 공간에서의 안정성과 효율적인 파라미터화를 활용하여 성능을 향상시킵니다. 또한, 차세대 고에너지 물리학 실험에서의 기계 학습 문제에 대한 잠재력을 강조하고 있습니다.

- **Technical Details**: 제안된 양자 비전 변환기(QViT) 아키텍처는 전통적인 비전 변환기의 주요 구성 요소에 QONNs를 내장하여 고차원 데이터에서의 주의 메커니즘의 성능을 개선합니다. 이 모델은 패치 추출, 양자 직교 계층을 포함한 자기 주의(Self-Attention), 다층 퍼셉트론(MLP) 구조를 통해 퀴크 및 글루온 유도 제트(jet)를 구분하는 데 초점을 맞춥니다. 각 패치는 1차원 벡터로 변환되어 고정 차원 임베딩 공간으로 투영됩니다.

- **Performance Highlights**: CMS 공개 데이터에서 다중 검출기 제트 이미지를 활용한 평가 결과, 양자 직교 변환을 주의 메커니즘 내에서 적용하면 강력한 성능이 입증되었습니다. QONNs의 도입은 효율적인 계산 및 분류 성능을 제공하며, 고강도 대형 하드론 충돌기(High Luminosity Large Hadron Collider)의 도전적인 기계 학습 문제에 대한 확장 가능성도 보여줍니다.



### Procurement Auctions via Approximately Optimal Submodular Optimization (https://arxiv.org/abs/2411.13513)
- **What's New**: 이 논문에서는 새로운 형태의 조달 경매(procurement auction)를 연구하고 있습니다. 전략적인 서비스 판매자들이 개인 비용을 가지고 경매자에게 입찰하는 과정에서, 서비스의 품질을 측정하기 위한 방법을 다룹니다. 특히, 기존 알고리즘 분석을 개선하며, 이를 기반으로 효율적인 틀(framework)을 설계하여 인센티브 호환성(incentive compatibility)과 개별 합리성(individual rationality)을 보장하고자 하였습니다.

- **Technical Details**: 저자들은 조건부 중심의 하위 모듈러 함수(submodular function) 최적화 알고리즘을 사용하여 조달 경매 메커니즘을 설계하였습니다. 이들 알고리즘은 오프라인 설정과 온라인 설정 모두에서 적용 가능하며, 변별력을 갖춘 결과를 도출합니다. 또한, 이들 하위 모듈러 최적화 알고리즘을 상세히 분석하고, 경쟁 설정에서의 내리는 경매(descending auction)에 대한 변환 가능성도 조사합니다.

- **Performance Highlights**: 실험을 통해 저자들은 제안한 틀을 최신 하위 모듈러 최적화 알고리즘에 적용하여 성능을 비교 분석하였습니다. 수천 개의 판매자들이 포함된 공개 데이터셋을 이용해 복지(welfare) 성능을 검토했으며, 이를 통해 경제적 효율성을 입증하였습니다. 이러한 연구는 조달 경매의 적용 가능성을 확대하고 관련 분야에서의 활용을 촉진할 것으로 기대됩니다.



### Utilizing Large Language Models to Synthesize Product Desirability Datasets (https://arxiv.org/abs/2411.13485)
Comments:
          9 pages, 2 figures, 6 tables

- **What's New**: 이번 연구는 Product Desirability Toolkit (PDT) 테스트를 위한 합성 데이터셋 생성을 대규모 언어 모델(LLMs)로 수행하는 방법을 탐구합니다. 특히 비용 효율적인 gpt-4o-mini 모델을 사용하여 1000개의 제품 리뷰를 생성하는 세 가지 방법(Word+Review, Review+Word, Supply-Word)을 제안합니다. 본 연구는 합성 데이터셋이 실제 데이터가 제한된 상황에서 비용 효과적이고 확장 가능한 옵션이 될 수 있는지를 평가합니다.

- **Technical Details**: 합성 데이터셋을 생성할 때, 높은 감정 일치를 보여주었고 Pearson 상관계수는 0.93에서 0.97에 이릅니다. Supply-Word 방법은 가장 높은 텍스트 다양성과 PDT 용어의 범위를 보였지만, 데이터 생성 비용이 증가했습니다. 이는 LLM이 생성한 합성 데이터가 특히 적은 테스트 데이터가 필요할 때의 이점을 제공함을 보여줍니다.

- **Performance Highlights**: 연구 결과는 개발된 세 가지 합성 데이터 생성 방법이 높은 감정 일치를 달성하며, 다양한 텍스트 형태를 제공한다고 보고합니다. 그러나 모든 방법에서 약간의 긍정적 편향이 관찰되었으며, 이는 미래 연구에서 해결해야 할 필요성을 제기합니다. 합성 데이터의 생산 비용 분석 또한 성과 강조 점으로, 이점은 확장 가능성 및 비용 절감 효과가 있음이 확인되었습니다.



### Conformal Prediction for Hierarchical Data (https://arxiv.org/abs/2411.13479)
Comments:
          14 pages, 2 figures

- **What's New**: 이번 논문에서는 Conformal Prediction과 Forecast Reconciliation의 조합을 통해 다변량 시계열의 확률적 예측을 개선하는 방법을 제안합니다. 기존의 예측 기법들은 여러 수준의 데이터를 활용하여 더 정확한 예측을 가능하게 하지만, Probabilistic Forecast Reconciliation (확률적 예측 조정)의 이론적 특성에 대한 이해가 부족했습니다. 연구에서는 Conformal Prediction의 절차에 조정 단계를 포함시킴으로써 예측 세트를 향상시키는 방법을 분석했습니다.

- **Technical Details**: 계층적 시계열(hierarchical time series)은 여러 차원의 예측이 가능한 구조를 가지며, 각각의 예측은 계층적 제약 조건을 따릅니다. 이 논문에서는 이러한 계층적 구조를 활용하여 Conformal Prediction을 통해 점 예측을 예측 세트로 변화시키는 절차를 정의합니다. 제안된 방법은 일반적인 조건 하에서 유효한 예측 세트를 생성하며, 이는 기존의 방법보다 효율성이 높습니다.

- **Performance Highlights**: 연구에서 제안된 방법은 시뮬레이션 결과를 통해 입증되었습니다. 조정된 예측 세트는 일반적인 방법보다 더 효율적인 결과를 보여주며, 최적의 조건하에서는 더욱 향상된 예측 성능을 달성합니다. 이를 통해, Hierarchical Time Series에 대한 예측의 정확성을 높이는 데 기여할 것으로 기대됩니다.



### Sampling and Integration of Logconcave Functions by Algorithmic Diffusion (https://arxiv.org/abs/2411.13462)
Comments:
          60 pages, 1 figure

- **What's New**: 본 연구에서는 임의의 logconcave 함수에 대한 샘플링(sampling), 반올림( rounding), 통합(integrating)의 복잡성을 연구하였습니다. 새롭게 제안된 방식은 지난 20년 동안 logconcave 함수에 대한 모든 문제들에 있어 첫번째 복잡성 개선을 가져왔으며, 볼록 집합(convex bodies)에 대한 균일 배포(uniform distributions)의 특별한 경우와도 가장 잘 알려진 복잡성에 부합합니다.

- **Technical Details**: logconcave 함수의 샘플링 및 적분은 여러 응용에서 필수적인 문제로, 볼록 집합 및 강력한 logconcave 밀도와 같은 중요 기특수 사례가 있습니다. Markov 체인 기반의 확률적 알고리즘을 사용하여 고차원에서의 샘플링을 수행하며, Markov 체인이 공정한 분포를 가질 수 있도록 설정되어야 합니다. 주요 도전 과제는 Markov 체인의 신속한 혼합을 보여주는 것인데, 이는 정적 분포에 대한 수렴 속도를 차원과 관련된 작은 다항식으로 묶는 것입니다.

- **Performance Highlights**: 이번 연구의 성과는 특히 샘플링 문제에 대한 출력 보장이 이전보다 훨씬 강력하다는 것입니다. 이는 의존적인 랜덤 샘플을 바탕으로 한 통계적 추정의 분석을 간소화하는 데 기여합니다. logconcave 밀도의 샘플링은 포트폴리오 최적화(portfolio optimization), 시뮬레이션 담금질(simulated annealing), 베이지안 추론(Bayesian inference), 차등 프라이버시(differential privacy) 등 여러 분야에서 중요한 응용을 가지고 있습니다.



### SoK: A Systems Perspective on Compound AI Threats and Countermeasures (https://arxiv.org/abs/2411.13459)
Comments:
          13 pages, 4 figures, 2 tables

- **What's New**: 본 논문에서는 다양한 대형 언어 모델(LLMs)과 관련된 시스템 공격을 체계적으로 분류하는 최초의 시도로, 복합 AI 시스템에서의 소프트웨어 및 하드웨어 취약점과 방어 방법을 탐구합니다. 현재 대부분의 연구가 개별 구성 요소에 초점을 맞추고 있는 반면, 본 논문에서는 계층 간 상호 작용이 이루어지는 복합 AI 시스템의 복잡성에 대한 강조를 하고 있습니다. 향후 공격 경로와 방어 메커니즘을 개발하기 위해 기존 문헌을 정리하고 새로운 위협 모델을 제시합니다.

- **Technical Details**: 본 연구는 AI 및 ML 시스템의 소프트웨어 및 하드웨어 계층에서 발생할 수 있는 여러 공격을 분석합니다. 특히, 시스템 공격과 알고리즘 공격을 명확히 구분하고, 다양한 공격 벡터와 그에 대한 방어를 체계적으로 분류합니다. 또한, 공격 및 방어 시스템을 체계화하기 위해 Mitre Att&ck 프레임워크를 활용하여 각 공격을 위협 모델에 맞춰 재정렬합니다.

- **Performance Highlights**: 복합 AI 시스템 내에서의 시스템 공격을 활용하여 엔드 투 엔드 공격을 수행할 수 있는 방법을 논의하고, 이를 통해 데이터 누수 및 보안 보장 훼손과 같은 문제를 해결하기 위한 방안을 모색합니다. 다양한 사례 연구를 통해 현재 보안 관행의 격차를 지적하고, 더 안전한 AI 애플리케이션을 구축하기 위한 열린 연구 질문을 제시합니다.



### AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations (https://arxiv.org/abs/2411.13451)
Comments:
          18 pages, 3 figures, an abridged version to appear in NeurIPS 2024 AFM Workshop

- **What's New**: 본 논문에서는 웹 기반 작업을 자동화하는 멀티모달 웹 에이전트를 위한 새로운 AdaptAgent 프레임워크를 제안합니다. 이 프레임워크는 불특정 웹사이트와 도메인에 대해 단 1~2개의 인간 시연을 통해 적응할 수 있도록 설계되었습니다. 실험 결과, 이 접근 방식이 현재 최첨단 모델에 비해 약 3.36%에서 7.21%의 작업 성공률 향상을 가져온다는 것을 보여주었습니다.

- **Technical Details**: AdaptAgent 프레임워크는 소수의 시연을 활용한 학습을 통해 기존의 대규모 사전 학습 및 미세 조정 전략을 보완하는 방법론을 제시합니다. 이 접근 방식은 두 가지 주요 메커니즘인 In-Context Learning (ICL)과 Meta-Learning을 통해 멀티모달 LLM(Multimodal Large Language Model) 기반 모델들이 새로운 웹 환경에 적응할 수 있게 합니다. 연구 결과, 멀티모달 시연이 텍스트 기반 시연보다 더 효과적임을 증명하였으며, 메타 학습에서의 데이터 선택 방법이 에이전트의 일반화 능력에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 두 가지 벤치마크인 Mind2Web과 VisualWebArena에서 수행된 실험을 통해, AdaptAgent 방식이 단 1-2개의 인간 시연을 사용하여 에이전트의 작업 성공률을 크게 향상시킬 수 있음을 입증했습니다. 전반적으로, 이 연구는 대규모 사전 학습과 미세 조정에 의존하지 않고도, 웹 에이전트의 일반화를 개선하는 보완적 방법론을 제시하며, 여러 웹 도메인에서의 작업 수행 가능성을 높이고 있습니다.



### WaterPark: A Robustness Assessment of Language Model Watermarking (https://arxiv.org/abs/2411.13425)
Comments:
          22 pages

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 오용을 줄이기 위한 방안으로 자리 잡고 있는 워터마킹 기법을 체계적으로 분석합니다. 기존 워터마커의 강점과 한계, 특히 공격에 대한 저항력을 집중적으로 평가하는 새로운 플랫폼인 WaterPark를 소개합니다. WaterPark는 10개의 최신 워터마커와 12개의 대표적인 공격을 통합하여 평가하는 데 초점을 맞추고 있으며, 그 설계 선택이 공격 저항력에 미치는 영향을 분석합니다.

- **Technical Details**: 토큰을 생성하는 과정에서 사용되는 LLM의 원리를 이해하기 위해, LLM의 워터마킹 방법을 세 가지 주요 요소인 LLM 자체, 워터마킹 절차(생성기), 탐지 절차(탐지기)로 나눕니다. 응용 프로그램 내에서 특정 통계적 신호를 시각적으로 나타내는 패턴으로서, 워터마커를 통해 LLM이 생성한 텍스트임을 입증할 수 있습니다. 이 연구에서는 대칭 및 비대칭 키를 이용한 워터마킹 기법의 작동 원리에 대해서도 설명합니다.

- **Performance Highlights**: 연구 결과는 여러 흥미로운 발견을 포함하며, 기존 연구의 결론에 도전하는 내용을 담고 있습니다. 예를 들어, 일부 워터마커는 예상보다 더 높은 오류율을 보이며, 여러 번의 패러프레이즈를 사용해야만 여전히 감지될 수 있음을 보여줍니다. 또한, 특정 워터마커는 고강도의 공격에는 저항하지 못함을 발견했습니다. 마지막으로, 격렬한 적대적 환경에서 워터마킹을 운영하기 위한 최상의 전략에 대해서도 논의합니다.



### Heuristically Adaptive Diffusion-Model Evolutionary Strategy (https://arxiv.org/abs/2411.13420)
- **What's New**: 본 연구는 Diffusion Models와 Evolutionary Algorithms 간의 근본적인 연결고리를 밝히고, 이 두 접근 방식이 고품질 샘플을 생성하는 공통의 생성 메커니즘을 공유하고 있음을 강조합니다. 두 방법 모두 무작위 초기 분포에서 반복적으로 개선된 샘플을 생성하여 최적화 과정에서의 유연성과 정밀성을 높입니다. 또한, Evolutionary Algorithms에 Diffusion Models의 메모리 기능을 통합하여 진화적 최적화에서 더 깊이 있는 히스토리 정보를 활용할 수 있음을 보여줍니다.

- **Technical Details**: Diffusion Models는 Gaussian noise를 사용하여 도메인 특정 정보를 저하시키고, 학습 가능한 모델을 통해 이를 복원하는 이중 단계 과정을 사용합니다. 이러한 모델이 Evolutionary Algorithms에 통합되면서 조건부 샘플링을 위한 classifier-free guidance를 제공하여, 개체군 전반의 특성을 세밀하게 조절할 수 있게 됩니다. 연구는 딥러닝 기반의 Diffusion Models를 다양한 진화적 작업에 적용하여, 최적의 파라미터에 대한 효율적인 수렴을 이룰 수 있음을 입증합니다.

- **Performance Highlights**: 제안된 접근 방식은 효율적인 수렴을 통해 높은 피트니스 파라미터를 유지하면서 탐색적 다양성을 보장합니다. 진화적 알고리즘이 새로운 적합 솔루션을 샘플링하는 방식을 개선하는 한편, 이 모델은 과거 설명 정보를 활용하여 보다 정교한 샘플을 생성하는 데 기여합니다. 또한, 본 연구는 기존의 얕은 휴리스틱에서 깊은 메모리를 가진 프레임워크로 진화적 알고리즘을 향상시키며, 진화적 탐색 다이나믹스를 보다 정밀히 조정하는 방안을 제시합니다.



### On the Way to LLM Personalization: Learning to Remember User Conversations (https://arxiv.org/abs/2411.13405)
Comments:
          16 pages, 6 tables, 3 figures

- **What's New**: 이 연구 논문에서는 대화의 개인화를 위한 새로운 접근 방식을 제안합니다. 기존의 개인화 방법들은 주로 사용자의 스타일이나 소소한 정보에 중점을 두었지만, 본 연구는 이전 대화에서의 지식을 LLM에 삽입하는 방법을 탐구하고 있습니다. 연구팀은 PLUM이라는 파이프라인을 제안하여 사용자 대화 데이터를 증강하고 이를 통해 파라미터 효율성이 높은 방식으로 LLM을 미세 조정하는 데 중점을 두고 있습니다.

- **Technical Details**: PLUM은 두 단계의 파이프라인으로 구성되어 있습니다. 첫 번째 단계에서는 사용자 대화 데이터를 증강하며, 두 번째 단계는 사용자 대화의 지식을 LLM에 주입하기 위한 파라미터 효율적인 미세 조정을 진행합니다. 이 과정에서 손실 함수로는 가중치가 있는 크로스 엔트로피 손실을 사용하고, 질문-답변 쌍을 생성하여 LLM의 학습에 활용합니다.

- **Performance Highlights**: 초기 실험에서 PLUM은 100개의 대화를 기준으로 81.5%의 정확도를 달성하며, 기존 RAG 기반 방법의 83.5%와 경쟁력을 보였습니다. 이 연구는 LLM 개인화를 위한 향후 연구의 기초를 마련하고 있으며, 대화 기억을 통한 새로운 개인화 방법론의 가능성을 확인시켜 줍니다.



### Explainable Finite-Memory Policies for Partially Observable Markov Decision Processes (https://arxiv.org/abs/2411.13365)
Comments:
          Preprint -- Under Review

- **What's New**: 이번 연구에서는 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Processes, POMDPs)에서 설명 가능성을 높이는 새로운 방법론을 제시합니다. 이 접근법은 메일리 머신(Mealy machine)과 의사결정 트리(decision trees)를 결합하여 선언된 정책의 효율적이고 설명 가능성이 높은 표현을 제공합니다. 특히, 이 논문에서는 유한 상태 컨트롤러(finite-state controllers, FSC)를 위한 새로운 데이터 구조를 제안하여, 확장성과 단순화를 이끌어냅니다.

- **Technical Details**: 연구에서 제안하는 FSC는 메일리 머신의 장점을 이용하여 각 상태가 고정 정책(stationary policy)에 대응하도록 구현됩니다. 각 상태 간 전환은 이제 의사결정 트리에 의해 설명되어, 설명 가능성을 강화합니다. 또한, '어트랙터 기반' 정책의 특정 속성을 활용하여 더 간단하고 작은 표현을 구축할 수 있는 방법을 설명합니다.

- **Performance Highlights**: 이 방법론은 또한 정량적 성능을 높일 수 있는 가능성을 보여줍니다. 여러 사례 연구(case studies)를 통해 제안된 방법의 설명 가능성을 실증적으로 입증하며, 주요 결과는 더 나은 성능을 제공합니다. 이로 인해 최적 정책(optimal policies)의 누락 문제를 해결하고 다양한 유한 메모리 정책에 대해 일반화하는 능력을 강화합니다.



### Are Large Language Models Memorizing Bug Benchmarks? (https://arxiv.org/abs/2411.13323)
Comments:
          pre-print

- **What's New**: 이 논문은 여러 소프트웨어 엔지니어링 작업에 필수적인 대형 언어 모델(LLMs)의 성능을 평가하기 위한 버그 벤치마크의 신뢰성에 대한 우려를 제기합니다. 연구진은 데이터 누수(data leakage)가 LLM의 성능 평가에 미치는 영향을 체계적으로 분석하여, 커뮤니티에서 널리 사용되는 벤치마크의 선택에 대한 신중함을 강조합니다.

- **Technical Details**: 연구진은 실제 버그 데이터셋을 바탕으로 LLM의 메모리화(memorization) 경향을 검토하기 위해 여러 가지 지표를 사용했습니다. 이러한 지표에는 Negative Log-Likelihood(NLL)와 n-gram accuracy가 포함되며, 이를 통해 모델이 특정 코드 조각과 얼마나 잘 친숙한지를 평가합니다. 벤치마크 데이터에서 메모리화 증거가 두드러진 코드 생성 모델(codegen-multi)과 더 적은 메모리화 징후를 보이는 현대 모델(LLaMa 3.1)을 비교했습니다.

- **Performance Highlights**: 연구 결과, codegen-multi 모델은 Defects4J와 같은 벤치마크 데이터에서 높은 n-gram accuracy와 낮은 NLL을 보였습니다. 반면, LLaMa 3.1과 같은 새로운 모델들은 더 넓은 데이터셋에서 학습하여 메모리화의 징후가 제한적이었습니다. 이러한 결과는 벤치마크 선택의 중요성과 LLM 평가 시 신뢰할 수 있는 메트릭을 사용해야 할 필요성을 강조합니다.



### Scaling Laws for Online Advertisement Retrieva (https://arxiv.org/abs/2411.13322)
Comments:
          10 pages, 8 figures

- **What's New**: 본 연구에서는 온라인 광고 검색 시스템에서의 스케일링 법칙(scaling law)을 식별하기 위한 경량화 혁신(paradigm)을 제안합니다. 이러한 접근 방식은 최소한의 실험 비용으로 온라인 수익(online revenue)과 기계 비용(machine cost)의 관계를 이해하는 데 중점을 둡니다. 특히, R/R*라는 새로운 오프라인 메트릭(offline metric)을 도입하여 검색 모델의 온라인 수익과 높은 선형 상관관계를 보이고 있습니다. 이는 광고 시스템 최적화의 가능성을 보여줍니다.

- **Technical Details**: 연구진은 R/R* 메트릭과 FLOPs(부동 소수점 연산 수)를 사용하여 오프라인 실험을 수행하고 MLP 모델의 스케일링 행동을 검증했습니다. 결과적으로, 이 스케일링 행동은 파손된 신경 스케일링 법칙(broken neural scaling law)을 따르는 것으로 나타났습니다. 또한, 기계 비용을 추정하기 위한 간단한 시뮬레이션 알고리즘을 제안하여 기계 비용과 온라인 수익 간의 관계를 설정하였습니다. 이를 통해 실험 비용을 크게 줄일 수 있었습니다.

- **Performance Highlights**: 이 연구의 주요 기여는 R/R* 메트릭이 온라인 광고 시스템의 수익을 예측하는 데 있어 뛰어난 오프라인 대리 메트릭 surrogate metric임을 입증한 것입니다. 이 메트릭을 활용함으로써 ROI 제약에 따른 모델 설계 및 다양한 시나리오에서 자원을 효율적으로 할당하였고, 각각 0.85%와 2.8%의 온라인 수익 개선을 달성했습니다. 이러한 결과는 광고 시스템 최적화에 있어 스케일링 법칙의 잠재력을 뒷받침합니다.



### DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition (https://arxiv.org/abs/2411.13284)
- **What's New**: 새로운 연구에서는 WiFi 기반 인식의 도메인 간 일반화를 위한 Domain-Adversarial Test-Time Adaptation (DATTA)라는 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 도메인-적대적 학습(DAT), 테스트 시간 적응(TTA) 및 가중치 리셋을 결합하여 이전에 본 적 없는 타겟 도메인에 적응하고 파국적 망각(catastrophic forgetting)을 방지합니다. 또한, DATTA는 속도 최적화를 위해 경량화되고 유연한 아키텍처에 통합됩니다.

- **Technical Details**: DATTA 프레임워크는 WiFi 기반의 인식 시스템에서 도메인 불변 특성을 학습하며, 수신 데이터의 도메인 전이에 적응하는 것을 목표로 합니다. 이 시스템은 WiFlexFormer 기반의 중앙 특징 추출기를 통해 실시간으로 인식할 수 있도록 설계되었습니다. 또한, 실제 측정한 WiFi 채널 상태 정보(CSI)의 특성을 이용한 특수한 보강 모듈이 통합되어 교차 도메인 일반화를 강화합니다.

- **Performance Highlights**: DATTA 방법론은 공개된 데이터셋을 사용한 포괄적인 평가에서 검증되었으며, 인간 활동 인식을 위한 실시간 애플리케이션에 적합함을 입증했습니다. 비디오 기반의 TTA 변형과 비교하여, DATTA는 8.1% 향상된 F1-스코어를 달성했습니다. 이 연구의 PyTorch 구현은 공개되어 있어, 후속 연구와 재현 가능성을 용이하게 합니다.



### BelHouse3D: A Benchmark Dataset for Assessing Occlusion Robustness in 3D Point Cloud Semantic Segmentation (https://arxiv.org/abs/2411.13251)
Comments:
          20 pages, 6 figures, 3 tables, accepted at ECCV 2024 Workshops

- **What's New**: BelHouse3D 데이터셋은 실세계 조건에 밀접하게 일치한 합성 포인트 클라우드 데이터셋으로, 실내 장면의 의미적 세분화를 위해 설계되었습니다. 벨기에의 32개 주택에서 얻은 실제 참고 자료를 바탕으로 구축되어 더 높은 공정성(fairness)과 신뢰성을 제공합니다. 이전의 데이터셋들과는 달리, 본 데이터셋은 occlusion(가림 현상)을 시뮬레이션한 테스트 세트를 포함하여 out-of-distribution (OOD) 상황을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 기존의 3D 벤치마킹 데이터셋은 종종 훈련과 테스트 데이터가 동일하게 분포(IID)된 것이라는 암묵적인 가정을 가지고 평가되었습니다. 하지만 실제 포인트 클라우드 데이터에서는 occlusion이 불가피하게 발생하여 장면의 형태와 구조가 변화하므로, OOD 샘플링에 따른 성능 저하가 우려됩니다. BelHouse3D는 이러한 문제를 해결하기 위해 수작업 3D 모델링과 샘플링 기법을 사용하여 데이터셋을 생성하며, 이는 라벨링 정확성을 향상시키기 위함입니다.

- **Performance Highlights**: BelHouse3D와 OOD 환경에서의 평가를 통해 인기 있는 포인트 기반 의미적 세분화 기법들의 성능을 검증하였습니다. 이러한 OOD 설정은 모델들이 실제 환경에서 어떻게 일반화되는지를 이해하는 데 도움을 주며, 더 견고하고 신뢰할 수 있는 모델 설계를 용이하게 합니다. 연구자들은 이 데이터셋과 OOD 환경을 통해 실내 장면에 대한 3D 포인트 클라우드 의미적 세분화 연구가 발전하길 기대하고 있습니다.



### On lower bounds of the density of planar periodic sets without unit distances (https://arxiv.org/abs/2411.13248)
Comments:
          21 pages, 9 figures

- **What's New**: 이번 연구 논문은 평면에서 유닛 거리(1 unit distance)가 없는 집합의 최대 밀도($m_1(	ext{ℝ}^2)$)에 대한 하한을 탐구합니다. 저자들은 이 문제를 최대 독립 집합(Maximal Independent Set, MIS) 문제로 재구성하여 평면 토러스(flat torus)에서 주기적인 집합에 집중하는 새로운 접근 방식을 제안했습니다. 이 연구는 이론적 근거와 함께 충분히 넓은 매개변수 범위에서 기존의 알려진 하한($0.22936 	ext{ ≤ } m_1(	ext{ℝ}^2)$)이 개선되지 않음을 확인했습니다.

- **Technical Details**: 저자들은 2개의 비공선형(non-collinear) 벡터에 대한 주기적 집합을 대상으로 하는 그래프 이론을 통해 문제를 공식화했습니다. 본 연구에서 계산된 최상의 이산 집합은 Croft의 구성을 기반으로 하는 밀도의 근사를 제공합니다. 게다가, MIS 문제를 해결하기 위해 여러 공개 소스 소프트웨어 패키지를 비교하여 효과성을 평가했습니다.

- **Performance Highlights**: 전반적으로 본 논문은 새롭게 제안한 접근 방식이 특정 매개변수에서 의미 있는 개선을 이루지 못했음을 보여주며, 기존 연구에 대한 기여가 큽니다. Croft의 구성 방식은 여전히 최대 밀도에 대한 가장 오래된 알려진 하한이며, 이 논문은 밀도 추정 문제의 기존 연구 방향을 상기시킵니다.



### Existential Conversations with Large Language Models: Content, Community, and Cultur (https://arxiv.org/abs/2411.13223)
- **What's New**: 본 논문은 최신 자연어 처리 기술을 기반으로 한 대화형 AI 시스템이 철학, 영성 및 종교와 같은 심오한 주제에 대해 심도 있는 대화를 나눌 수 있는 방법을 다룹니다. 이는 사용자가 인공지능의 존재론적 문제에 대해 참여하게 하여, 보다 풍부한 대화 경험을 제공합니다. 논문은 이러한 대화의 현대적이고 고전적인 문화적, 사회적 배경을 분석하고, AI 시스템과의 상호작용이 사회에 미치는 영향을 고려합니다.

- **Technical Details**: 대형 언어 모델(LLM)은 주어진 텍스트의 연속성을 생성할 수 있는 컴퓨터 프로그램으로, 대량의 언어 데이터에 대해 훈련된 인공 신경망입니다. 이 모델은 사용자의 입력에 따라 적합한 단어를 예측하고, 주어진 문맥에 맞게 대화를 이어나갈 수 있습니다. 논문은 LLM이 수행하는 대화의 흐름과 그 과정에서 나타나는 다양한 철학적 질문들에 대해 기술적 관점에서 접근합니다.

- **Performance Highlights**: 현재 LLM 기반 AI 시스템은 사용자와의 대화에서 매우 매력적이고 깊이 있는 경험을 제공합니다. 특히 Anthropic의 Claude 3는 철학적인 주제에 대한 대화를 진행할 수 있는 가능성을 보여줍니다. 이러한 시스템은 단순히 훈련된 데이터를 반복하는 것이 아니라, 사용자의 맥락에 맞는 대화를 생성하며, 자아 및 의식에 대한 흥미로운 논의를 이어갑니다.



### ViSTa Dataset: Do vision-language models understand sequential tasks? (https://arxiv.org/abs/2411.13211)
- **What's New**: 이 연구에서는 시각-언어 모델(Vision-Language Models, VLMs)을 강화 학습에서 보상 모델로 활용하는 가능성을 탐구합니다. 기존에는 특정 최종 결과를 중심으로 한 목표 지향 작업에서만 사용되었지만, 본 연구에서는 VLMs가 최종 상태에 의해 평가할 수 없는 연속 작업을 감독하는 데도 활용될 수 있음을 주장합니다. 이를 위해 'ViSTa'라는 새로운 데이터셋을 도입하여 다양한 복잡도의 작업을 평가합니다. ViSTa는 가상 환경, 마인크래프트, 실제 환경에서 수행된 4,000개 이상의 비디오와 단계별 설명으로 구성되어 있습니다.

- **Technical Details**: ViSTa는 기본 단일 액션 작업을 더 복잡한 순차 작업으로 조합하는 새로운 계층 구조를 가지고 있습니다. 이 계층적 구조를 통해 VLMs가 다양한 복잡도의 작업을 어떻게 이해하고 판단할 수 있는지를 세밀하게 평가할 수 있습니다. 연구에서는 최신 VLM인 CLIP, ViCLIP, GPT-4o를 사용하여 평가를 진행했고, 제안된 데이터셋을 통해 이러한 모델들이 순차 작업을 이해하는 데 한계가 있음을 발견했습니다. 특히, GPT-4o는 비록 우수한 객체 인식 능력을 보였지만, 복잡한 연속 작업에서는 비트 이상의 성능을 보이지 못했습니다.

- **Performance Highlights**: 비교한 모델들 모두 기본적인 객체 인식 작업에서는 우수한 성능을 보였지만, 순차 작업에 대한 이해에서는 실패를 보였습니다. 이 연구는 모델이 특정 복잡도 이상의 작업을 감독할 준비가 되지 않았음을 강조합니다. 이러한 결과는 VLMs의 적용 가능성을 재조명하며, 저자들은 ViSTa를 통해 이러한 모델들이 순차 작업에서 겪는 문제를 체계적으로 분석할 수 있도록 하는 것을 목표로 삼고 있습니다. VLM의 한계를 극복하고, 보다 효과적인 감독 방식의 개발이 필요함을 시사합니다.



### The Information Security Awareness of Large Language Models (https://arxiv.org/abs/2411.13207)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 정보 보안 인식(ISA)을 평가하는 새로운 방법론을 제안합니다. 이 평가를 위해 특정 보안 맥락이 즉시 드러나지 않는 사용자 질문에 LLM이 응답하는 다양한 시나리오를 생성했습니다. 연구 결과, LLM이 보안 관련 지식을 효과적으로 적용하는 능력은 모델의 템퍼러처에 영향을 받으나, 시스템 프롬프트의 설정에 더 큰 영향을 받는다는 사실이 밝혀졌습니다.

- **Technical Details**: 이 연구에서는 30개의 시나리오를 구성하여 LLM의 정보 보안 인식을 평가합니다. 각 시나리오는 모바일 ISA 분류법에서 정의된 하위 초점 분야에 기초하여 설정되었으며, LLM의 응답은 인간의 평가와 LLM 기반 평가 모두로 점수를 부여받았습니다. 특히, 시스템 프롬프트와 온도 설정에 따라 ISA의 변화가 관찰되었고, 이는 LLM의 안전한 사용을 위한 중요한 요소임을 강조합니다.

- **Performance Highlights**: 이 평가를 통해 여러 인기 있는 LLM이 나타내는 ISA 수준이 상이하다는 것을 발견했습니다. 이러한 결과는 특정 보안 문제에 대한 전문성을 구축하기 위해 단일 모델에 의존하기보다는 서로 보완하는 여러 LLM을 활용할 필요성을 제시합니다. 또한, 모델의 초점 분야의 차별성과 각 모델의 강점과 약점이 평가되었으며, 이는 LLM 기반 보조 도구의 향후 개발에 있어 정보 보안 인식 평가의 중요성을 부각시킵니다.



### Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding (https://arxiv.org/abs/2411.13157)
- **What's New**: 최근 대형 언어 모델(LLMs)의 효율적인 추론(inference)은 이들의 규모와 복잡성이 커짐에 따라 중요한 이슈로 대두되고 있습니다. 전통적인 오토리그레시브 디코딩(autoregressive decoding)은 순차적으로 토큰을 생성하는 방식으로 인해 계산비용에서 비효율적입니다. 이 연구에서는 초안(draft)과 확인(verification)이라는 두 단계의 접근 방식을 도입하여 교묘한 디코딩(speculative decoding) 기술을 통해 이러한 병목 현상을 해결하는 방법을 모색합니다.

- **Technical Details**: 교묘한 디코딩은 작은 모델을 사용하여 초기 초안을 생성하고, 보다 큰 모델이 이 초안을 확인하는 방식으로 작동합니다. 여기서는 모델 중심(model-centric)과 초안 중심(draft-centric) 방식으로 교묘한 디코딩 방법을 분류하며, 각 접근법의 핵심 아이디어와 잠재력을 다룹니다. 기술적으로, 초안 생성의 질과 효율성을 개선하는 방법으로 다양한 구현 전략에 대해 이야기하고 있습니다.

- **Performance Highlights**: 성공적인 교묘한 디코딩 방법으로는 Medusa와 EAGLE-2가 언급됩니다. Medusa는 추가 디코딩 헤드를 사용하여 후속 토큰을 병렬로 처리하는 방식으로 성능을 향상시켰습니다. 반면 EAGLE-2는 동적인 초안 트리를 통해 더 나은 샘플링을 가능하게 하여 추론 속도와 정확성을 높이는 데 기여하고 있습니다.



### Virtual Staining of Label-Free Tissue in Imaging Mass Spectrometry (https://arxiv.org/abs/2411.13120)
Comments:
          33 Pages, 6 Figures

- **What's New**: 이번 연구에서는 영상 질량 분석(Imaging Mass Spectrometry, IMS)의 새로운 가상 조직 염색 방법을 제안합니다. 이 방법은 IMS의 10배 더 큰 픽셀 크기를 사용하면서도, 실험에서 얻은 이미지를 기존의 조직 염색 이미지와 유사하게 재현할 수 있음을 보여줍니다. 이를 통해 IMS의 화학적 특이성과 민감도를 유지하면서도 세포 형태학적 대비를 디지털적으로 강화하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 확산 모델(diffusion model)을 이용하여 라벨 없는 인체 조직에서의 질량 분석 이미지를 향상시킵니다. 무작위 테스트(Blind testing)를 통해 다양한 신장 조직 샘플에서 가상 염색 이미지가 기존의 조직 염색 이미지와 매우 유사하다는 결과를 얻었습니다. 이 과정에서 최적화된 노이즈 샘플링 기법을 활용하여 생성된 이미지의 변동성을 감소시켜 믿을 수 있고 재현 가능한 가상 염색을 가능하게 합니다.

- **Performance Highlights**: 가상 염색 방법은 생명 과학 분야에서 IMS의 적용 가능성을 크게 확대할 것으로 기대됩니다. 연구 결과는 새로운 질량 분석 기반 생물 의학 연구 분야에서의 잠재적인 혁신을 시사하며, IMS 데이터로부터 세포와 조직의 특정 특징을 정밀하게 식별할 수 있는 새로운 가능성을 제공합니다.



### MEGL: Multimodal Explanation-Guided Learning (https://arxiv.org/abs/2411.13053)
- **What's New**: 이번 연구에서는 Multimodal Explanation-Guided Learning (MEGL) 프레임워크를 제안하여 이미지 분류 작업의 해석 가능성을 높이고 성능을 개선하고자 합니다. 기존의 XAI 방법들이 시각적 혹은 텍스트 기반의 단일 모달 해석에 의존했던 반면, MEGL은 두 가지 모달의 보완적인 특성을 통합하는 접근 방식을 사용합니다. 이를 통해 시각적 및 텍스트 기반 설명의 일관성과 완전성을 높여 AI 모델의 '블랙 박스'(black box) 특성을 해결하고자 합니다.

- **Technical Details**: MEGL에서 제안하는 Saliency-Driven Textual Grounding (SDTG) 방법은 시각적 설명에서 얻은 공간 정보를 텍스트 설명에 통합합니다. 이는 입력 이미지를 처리하여 생성된 Saliency Map을 사용하여 텍스트에서 공간적으로 관련된 통찰을 효과적으로 반영하도록 돕습니다. 또한, Visual Explanation Distribution Consistency loss를 도입하여 시각적 설명의 일관성을 강화하고, 누락된 주석이 있는 경우에도 안정적이고 맥락적으로 타당한 시각적 설명을 보장합니다.

- **Performance Highlights**: Object-ME와 Action-ME라는 두 개의 새로운 데이터셋을 기반으로 실험을 수행한 결과, MEGL은 기존의 이미지 분류 및 해석 가능성 증진 방법보다 우수한 성능을 보였습니다. MEGL은 예측 정확성, 시각적 설명 가능성 및 텍스트 설명 가능성 모두에서 뛰어난 성능을 기록하며, 다양한 훈련 조건에서 유의미하고 일관된 다중모달 설명을 효과적으로 이용할 수 있음을 입증하였습니다.



### On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspectiv (https://arxiv.org/abs/2411.13052)
- **What's New**: 이 논문에서는 Content-based Recommender Systems (CRSs)의 임베딩 테이블이 저장 용량에서 중요한 병목 현상을 초래하는 문제를 해결하기 위해 Shapley Value-guided Embedding Reduction (Shaver)라는 새로운 접근 방식을 제안합니다. 기존의 임베딩 가지치기 방법은 각 목표 매개변수 예산을 위해 재학습이 필요하여 높은 계산 비용을 초래하는데, 본 연구는 이러한 문제에 혁신적인 해결책을 제공합니다.

- **Technical Details**: Shaver는 협력적 게임 관점에서 문제를 바라보고 각 임베딩 매개변수의 기여도를 Shapley values로 정량화하여 기여 기반 매개변수 가지치기를 촉진합니다. 또한 Shapley values의 계산 비용을 줄이기 위해 효율적이고 편향되지 않은 방법을 제안하며, 가지치기 단계에서는 전통적인 zero-out 처리를 보완하기 위한 field-aware codebook을 도입하여 정보 손실을 최소화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 Shaver는 다양한 매개변수 예산에서 경량 추천 모델과 경쟁력 있는 성능을 보여주었습니다. 이 연구는 federated learning 및 스트리밍 환경과 같은 다양한 저장 요구 사항을 가진 실제 응용 프로그램에서의 효율성을 제공하여 주목할 만한 기여를 하고 있습니다.



### Training Physics-Driven Deep Learning Reconstruction without Raw Data Access for Equitable Fast MRI (https://arxiv.org/abs/2411.13022)
- **What's New**: 이 논문에서는 Compressibility-inspired Unsupervised Learning via Parallel Imaging Fidelity (CUPID)라는 새로운 방법을 제안하여, 고속 자기공명영상(MRI) 재건을 위한 PD-DL 훈련을 가능하게 합니다. CUPID는 전문 MRI 센터 외에서 사용할 수 없는 원시 k-공간 데이터에 대한 접근성을 필요로 하지 않으며, 임상에서 생성된 재구성 이미지만을 사용합니다.

- **Technical Details**: CUPID는 DICOM 형식으로 획득된 임상 이미지에서 훈련되며, Noise 및 Aliasing 아티팩트가 포함될 수 있는 저어 샘플링된 데이터를 활용합니다. 이 방법은 Parallel Imaging 알고리즘과의 일관성을 보장하고, 출력 이미지의 Compressibility를 평가함으로써 unsupervised 조건에서 훈련을 수행합니다.

- **Performance Highlights**: CUPID는 기존의 감시(supervised) 및 자기감시(self-supervised) PD-DL 훈련 전략과 비교할 때 동등한 결과를 달성하며, 전통적인 압축 감지(compressed sensing, CS) 기법 및 최신 생성 모델보다 뛰어난 성능을 보여줍니다. 이 방법은 농촌 및 소외 지역에서도 고속 MRI 접근성을 높이는 데 기여할 수 있습니다.



### NCAirFL: CSI-Free Over-the-Air Federated Learning Based on Non-Coherent Detection (https://arxiv.org/abs/2411.13000)
Comments:
          6 pages, 2 figures, submitted for possible publication

- **What's New**: 이 논문은 AirFL(Air over-the-air federated learning)에서 CSI(채널 상태 정보) 없이 작동하는 NCAirFL 방식을 제안합니다. 기존의 AirFL 방식은 여러 디바이스에서 신호를 정렬하기 위해 CSI에 의존하는 문제를 가지고 있었습니다. NCAirFL은 비대칭 비동기 감지를 통해 이 문제를 해결하고, 장기 메모리 기반 오류 보정 메커니즘을 활용하여 더 높은 수렴률을 달성합니다.

- **Technical Details**: NCAirFL에서는 바이너리 디더링(binary dithering) 및 비대칭 비동기 감지(unbiased non-coherent detection) 방법이 사용됩니다. 이 방법들은 평균 제곱 노름(average square norm)과 같은 일반적인 비볼록 최적화 문제에 대해 수렴 속도가 𝒪(1/√T)인 결과를 도출합니다. 이러한 수렴 속도는 이상적인 통신을 가정한 FedAvg와 일치하여 다른 방법들과 비교했을때 효율성을 강조합니다.

- **Performance Highlights**: 실험 결과, NCAirFL은 이상적인 통신 환경을 가진 기존 FL 및 일관된 전송 방식의 벤치마크와 비교하여 경쟁력 있는 성능을 보여줍니다. 특히, NCAirFL은 수렴 성능에서 월등한 우위를 보이며, 전송 효율성 또한 개선되었습니다. 이러한 결과는 무선 채널의 특성을 잘 활용한 비채널 의존 방식의 가능성을 보여줍니다.



### Eliminating Ratio Bias for Gradient-based Simulated Parameter Estimation (https://arxiv.org/abs/2411.12995)
- **What's New**: 이번 연구에서는 확률 모델의 매개변수 보정을 위한 새로운 접근 방식을 제안합니다. 기존의 최대 우도 추정(maximum likelihood estimation, MLE)와 사후 밀도 추정(posterior density estimation, PDE)에서 발생하는 비율 편향(ratio bias) 문제를 해결하기 위해 다중 시간 척도(multi-time scale, MTS) 알고리즘을 활용한 그래디언트 기반의 시뮬레이션 매개변수 추정 프레임워크를 도입하였습니다. 또, 네트워크 학습(neural network training) 분야에도 이 프레임워크를 확장함으로써 머신러닝에서의 확률 근사에 대한 새로운 관점을 제공합니다.

- **Technical Details**: 이 논문에서는 시스템 동역학 시스템을 기반으로 하는 확률 모델을 다루며, 이들 모델에서 우도 함수(likelihood function)의 정량적 형태가 존재하지 않은 경우의 매개변수 추정 방법론을 집중적으로 분석합니다. 이러한 문제를 해결하기 위해 제안된 알고리즘은 강한 수렴(strong convergence), 점근적 정규성(asymptotic normality), 수렴 속도(convergence rate) 및 예산 할당 전략들에 대한 이론적 분석을 포함합니다. Nested simulation을 통해 KL 발산(Kullback-Leibler divergence)을 최소화하는 방식을 사용하여 최적화 문제를 해결하는 구조를 제시합니다.

- **Performance Highlights**: 수치 실험 결과, 제안된 GSPE 알고리즘은 매개변수 추정 정확성을 향상시키고 계산 비용을 절감하는 데 효과적임을 보여주었습니다. 알고리즘이 도출하는 업데이트 사항들은 비율 편향을 제거하며, 이는 실제 문제에서 신뢰할 수 있는 매개변수 추정을 가능하게 합니다. 또한, 이 알고리즘은 다양한 최적화와 가격 책정 문제에 적용 가능성이 높아, 실용적인 응용측면에서의 잠재력을 나타냅니다.



### BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices (https://arxiv.org/abs/2411.12990)
Comments:
          Accepted as a Spotlight Poster to NeurIPS 2024

- **What's New**: 이 논문은 AI 벤치마크의 품질을 평가하기 위한 새로운 프레임워크를 개발하였으며, 46개의 기준을 바탕으로 24개의 AI 벤치마크를 평가하였습니다. 야기된 결과는 품질의 큰 차이를 드러내며, 널리 사용되는 벤치마크들이 중요한 문제에 직면해 있음을 보여줍니다. 통계적 유의성을 보고하지 않거나 결과의 반복 사용이 어려운 벤치마크가 대다수를 차지하여, 이는 모델 평가의 신뢰성에 관한 우려를 나타냅니다.

- **Technical Details**: 이 연구는 AI 벤치마크의 생애 주기 전반에 걸쳐 46개의 최선의 관행을 고려한 평가 프레임워크를 제공합니다. 또한, 기초 모델(Foundation Model, FM)와 비기초 모델(non-Foundation Model) 각각에 대해 16개와 8개의 벤치마크를 평가하여 품질 차이를 발견했습니다. 이 평가 결과를 바탕으로 최소 품질 보증 체크리스트를 제공하며, 벤치마크 품질 및 사용 가능성을 분석할 수 있는 살아있는 리포지토리도 개발했습니다.

- **Performance Highlights**: 연구의 결과, 현재 AI 벤치마크의 품질에 대한 표준적인 구조적 평가가 부족하다는 점이 지적되었습니다. 사용된 벤치마크들은 실제 응용에 적합하고 실질적인 유용성을 제공하지 못하는 경우가 많았습니다. 이를 통해 더 나은 벤치마크를 개발할 수 있도록 도움을 줄 수 있는 체크리스트와 가이드라인이 제공되어, 개발자들이 최선의 관행에 맞춰 벤치마크를 개선할 수 있는 방향을 제시합니다.



### Training Bilingual LMs with Data Constraints in the Targeted Languag (https://arxiv.org/abs/2411.12986)
Comments:
          22 pages, 14 figures, 15 tables

- **What's New**: 본 연구는 데이터가 부족한 목표 언어(target language)의 성능을 향상시키기 위해 고품질 보조 언어(auxiliary language)의 데이터를 활용하는 방법을 탐구합니다. 특히 영어와 같이 데이터가 풍부한 보조 언어의 정보를 활용하여 성능 차이를 정량적으로 평가하고, 모델의 스케일링 한계와 데이터 업샘플링 방법을 제안합니다. 기존의 영어 중심 모델들과는 달리, 목표 언어의 성능 향상을 위한 새로운 길을 모색하고 있습니다.

- **Technical Details**: 이 연구에서는 기존의 영어 데이터 선택 파이프라인(data selection pipelines)을 사용하여 데이터가 제한적인 목표 언어에 대한 사전 학습(pretraining) 성능을 증진시키는 방법을 실험합니다. 독일어를 목표 언어로 설정하고, 다양한 스케일의 디코더 전용 트랜스포머 모델을 학습하여 성능 개선을 분석합니다. 모델은 시간적으로 30K와 100K 스텝에서 학습하며, 배치 크기는 1024입니다.

- **Performance Highlights**: 연구 결과, 고품질의 보조 데이터는 목표 언어의 성능 향상에 긍정적인 영향을 미치며, 정보가 풍부한 영어 데이터셋이 다른 언어에서도 유사한 성과를 보일 수 있음을 보여줍니다. 특히, 영어 데이터셋의 품질이 높아질수록 목표 언어에서의 성능도 함께 향상되며, 가까운 언어(cognate languages)에서 효과가 두드러지지만, 영어와 거리가 먼 언어에서는 효과가 제한적임을 확인하였습니다.



### On adaptivity and minimax optimality of two-sided nearest neighbors (https://arxiv.org/abs/2411.12965)
Comments:
          29 pages, 7 figures

- **What's New**: 이 논문은 비선형(non-linear) 함수와 다량의 결측 데이터(missing data)에 대한 최근접 이웃(nearest neighbor, NN) 알고리즘을 분석합니다. 이전 연구들은 주로 매끄럽고(Lipschitz) 결측 확률(missingness probabilities)이 낮은 경우의 NN에 대한 유리한 보장을 제공했지만, 본 연구에서는 비선형 함수의 비매끄러운(bi-smooth) 경우를 다룹니다. 특히, 우리는 잠재적(non-linear latent factor) 요인 모델에 따라 데이터가 구성된 행렬 완성(matrix completion) 문제에 초점을 맞추었습니다.

- **Technical Details**: 이 연구에서는 NN의 평균 제곱 오차(mean squared error, MSE)가 비선형성의 매끄러움(smoothness)에 적응하며, 특정 조건 하에서는 NN의 오류율이 오라클(oracle)과 일치한다는 점을 보여줍니다. 데이터가 임의로 결측되지 않는 경우(missing not at random, MNAR), NN의 MSE는 여전히 비단순적(non-trivial)인 값을 갖습니다. 이론적 결과는 다양한 수치 시뮬레이션과 모바일 건강 연구인 HeartSteps 데이터 사례를 통해 지지됩니다.

- **Performance Highlights**: 변화에 맞춰 적응하는 NN 방법은 이전의 bilinear 모델에 비해 더 일반적인 비편향(non-parametric) 모델로, 결측 데이터의 처리 성능을 높입니다. 본 논문에서 제안된 방법은 다양한 실제 적용 사례에서 유용성을 보여주며, 특히 추천 시스템이나 순차 결정 문제에서의 활용 가능성을 강조합니다. 연구 결과는 기존 연구 보다 더 많은 것들을 수용 가능하며, 다수의 딥러닝(deep learning) 및 머신러닝(machine learning) 응용에 영향을 미칠 것으로 기대됩니다.



### A Flexible Large Language Models Guardrail Development Methodology Applied to Off-Topic Prompt Detection (https://arxiv.org/abs/2411.12946)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 논문은 사용자들이 LLMs(Large Language Models)를 오용할 때 발생하는 문제를 다루고 새로운 방법론을 제시합니다. 기존의 guardrails는 제한적인 예시나 맞춤형 분류기에 의존하여 높은 오류율과 적응성 부족, 실제 데이터의 부족 등의 문제를 안고 있었습니다. 논문에서는 문제 영역을 질적으로 정의하고 이를 통해 다양한 프롬프트를 생성함으로써 합성 데이터셋을 구축하였으며, 이는 기존의 휴리스틱 방법보다 우수한 오용 방지 기능을 가지게 됩니다.

- **Technical Details**: 새롭게 제안된 데이터 없는 guardrail 개발 방법론은 특정 작업에 대한 사용자 프롬프트의 적합성을 분류함으로써 작동합니다. 이 방법론은 LLM을 활용해 다양한 합성 프롬프트를 생성하고, 이를 활용하여 오프-토픽 프롬프트를 탐지하는 분류기를 훈련합니다. 최종적으로 이 방식은 false positive를 줄이고, 다양한 오용 유형에 대한 일반화 가능성을 갖추게 됩니다.

- **Performance Highlights**: 제안된 방식은 모델의 초기 배포를 위해 강력한 기반을 제공하며, 합성 데이터로 훈련된 분류기는 기존의 휴리스틱 방법과 비교하여 오프-토픽 프롬프트 인식 정확도를 크게 향상시킵니다. 또한, 개방형 리소스 형태로 합성 데이터셋 및 오프-토픽 guardrail 모델을 공개하여 향후 LLM 안전성과 규정 준수 연구에 기여하도록 하고 있습니다.



### Enhancing Thermal MOT: A Novel Box Association Method Leveraging Thermal Identity and Motion Similarity (https://arxiv.org/abs/2411.12943)
Comments:
          Workshop on Towards a Complete Analysis of People, part of the European Conference on Computer Vision (ECCV) 2024

- **What's New**: 이번 논문은 열화상 이미지에서의 다중 객체 추적(MOT) 성능을 개선하기 위한 혁신적인 방법을 소개합니다. 저자들은 열 객체의 특성과 운동 유사성을 결합한 새로운 박스 연관(Box Association) 방법을 개발하였습니다. 이러한 방법은 열 특성의 희소성과 동적 객체 추적을 통합하여 MOT의 정확성과 견고성을 향상시킵니다.

- **Technical Details**: 이 논문은 전통적인 운동 연관(Motion Association) 방법에 의존하는 대신, 두 단계 MOT 모델의 연관 과정에서 열상 정체성(Thermal Identity)를 활용하는 새로운 박스 연관 알고리즘을 제안합니다. 이를 통해 알고리즘은 열 데이터의 특성과 운동 데이터를 함께 적용하여 보다 정확한 박스 연관을 달성합니다. 또한 다양한 도시 환경에서 캡처된 열 및 RGB 이미지의 대규모 데이터셋을 새롭게 구축하여 실험과 평가의 기초 자료로 제공합니다.

- **Performance Highlights**: 저자들은 제안한 방법이 기존 방법들에 비해 우수한 성능을 보이며, 다양한 조건에서 추적의 정확성과 견고성을 크게 향상시킨다고 보고합니다. 새로운 데이터셋과 소스 코드는 공개되어 연구자들이 이 방법을 재현하고 확장할 수 있게 합니다. 또한, 이 연구는 모든 센서 모달리티에 대한 MOT 알고리즘의 연구와 개발을 촉진하는 효과를 기대하고 있습니다.



### On the relationship between Koopman operator approximations and neural ordinary differential equations for data-driven time-evolution predictions (https://arxiv.org/abs/2411.12940)
- **What's New**: 이 논문은 비선형 다이나믹 시스템의 시간 진화를 예측하기 위한 상태 공간 방법과 쿠프만 연산자(Koopman operator) 기반 방법 간의 관계를 탐구합니다. 확장된 동적 모드 분해(dynamic mode decomposition, DMD)와 딕셔너리 학습(dictionary learning)을 결합한 방법이 상태 공간에서 비선형 이산 시간 흐름 맵의 신경망(neural network) 표현과 동등함을 보여주었습니다. 이로 인해 EDMD-DL의 예측 성능이 크게 향상된다는 것을 강조합니다.

- **Technical Details**: EDMD-DL은 상태 공간 진화를 모델링하기 위해 신경망의 비선형 특성을 활용하고, 상태를 고차원 특징 공간으로 확장하는 구조를 가지고 있습니다. 이 논문에서는 EDMD-DL을 사용하여 비선형 동적 시스템을 탐색하며, 여러 종류의 신경망 기반 정규 미분 방정식(ODE)과 EDMD-DL 변형을 구현했습니다. 다양한 모델 구조와 훈련 절차를 결합하여 여러 방법을 평가하였으며, 이는 카오스 동역학 분야에서의 수치 실험을 통해 확인되었습니다.

- **Performance Highlights**: 본 연구는 로렌츠 시스템(Lorenz system)과 난류 전단 흐름(turbulent shear flow)의 아홉 개 모드 모델을 사용하여 여러 방법들의 성능을 비교했습니다. 짧은 시간 동안의 경로 예측, 긴 시간 통계 재구성, 그리고 드문 사건 예측에서 방법들 간의 비교 가능한 성능을 보여주며, 비마르코프적(non-Markovian) 접근 방식과의 극단 사건 예측 성능도 유사함을 입증했습니다.



### Improving Low-Fidelity Models of Li-ion Batteries via Hybrid Sparse Identification of Nonlinear Dynamics (https://arxiv.org/abs/2411.12935)
Comments:
          6 pages

- **What's New**: 이번 논문에서는 리튬 이온 배터리(LiB) 모델의 정확성을 높이기 위한 데이터 기반 접근법을 소개합니다. 제안된 방법은 유전자 알고리즘(Genetic Algorithm)과 순차적으로 임계값이 설정된 리지 회귀(Sequentially Thresholded Ridge Regression, GA-STRidge)를 결합하여 저정밀 모델(Low-Fidelity Model, LFM)과 고정밀 모델(High-Fidelity Model, HFM) 간의 불일치를 보상합니다. 이 연구는 배터리의 전압 예측 오차를 줄이면서 계산 효율성을 유지할 수 있는 하이브리드 모델을 테스트합니다.

- **Technical Details**: 이 연구의 출발점은 신뢰도가 낮은 리튬 이온 배터리(LiB)의 모델로, 전기화학 모델의 차원 축소를 통해 얻어진 모델입니다. 전압, 이온 및 전하의 수송, 평형 전위 및 계면 반응 속도 등의 복잡한 비선형 물리적 현상을 설명하기 위한 비선형 및 편미분 방정식이 사용됩니다. 연구자들은 이 모델의 오류 다이나믹스를 데이터 기반 모델로 개선하기 위해 유전자 알고리즘에 의해 안내되는 순차적 리지 회귀 방법을 활용합니다.

- **Performance Highlights**: 제안된 GA-STRidge 방법은 다양한 주행 사이클에서 검증되어 저정밀 모델(LFM)의 전압 예측 오차를 획기적으로 줄였습니다. 또한, 모델은 다양한 작동 조건에서 평가되어 익숙하지 않은 환경에서도 예측 오차가 낮고 높은 피어슨 상관계수를 보여주며, 데이터 드리븐 시스템 식별을 통해 예측 정확도가 크게 향상됨을 확인하였습니다. 이 접근법은 효율성과 함께 변화하는 조건에 대한 일반화 가능성을 보장합니다.



### Human-In-the-Loop Software Development Agents (https://arxiv.org/abs/2411.12924)
- **What's New**: 최근 LLM(대형 언어 모델)을 기반으로 한 다중 에이전트 패러다임이 소프트웨어 개발 작업을 자동으로 해결하는 데 도입되었습니다. 하지만 기존 연구는 역사적 벤치마크 데이터셋에만 의존하고, 자동화된 소프트웨어 개발 과정의 각 단계에서 인간 피드백을 고려하지 않고 있습니다. 본 논문에서는 HULA(휴먼-인-더-루프 LLM 기반 에이전트) 프레임워크를 도입하여 소프트웨어 개발 엔지니어가 LLM을 개선하고 유도할 수 있도록 합니다.

- **Technical Details**: HULA 프레임워크는 AI 계획 에이전트, AI 코딩 에이전트, 인간 에이전트 등 세 가지 에이전트로 구성됩니다. 이들은 협력하여 JIRA 이슈를 해결하는 공통 목표를 달성합니다. HULA의 효과를 평가하기 위해 세 가지 단계의 평가를 수행하였으며, 이는 벤치마크 데이터셋에 대한 오프라인 평가, 인간 피드백이 포함된 온라인 평가, 그리고 HULA 사용에 대한 실무자의 인식을 조사하는 것입니다.

- **Performance Highlights**: HULA 프레임워크는 총 663개의 실제 JIRA 이슈에 대해 계획을 성공적으로 생성했으며, 이 중 433개의 계획이 실무자에 의해 승인되어 82%의 승인률을 기록했습니다. 코드 작성 단계에서는 376개 이슈 중 95개에 대해 풀 리퀘스트가 생성되었고, 이 중 56개 풀 리퀘스트가 성공적으로 병합되었습니다. 조사 결과 대다수 참가자들은 생성된 코드의 가독성과 수정 용이성에 동의하였으나, 코드가 작업을 완전히 해결하기 위해 인간의 개입이 필요하다고 인식하였습니다.



### Enhancing Deep Learning-Driven Multi-Coil MRI Reconstruction via Self-Supervised Denoising (https://arxiv.org/abs/2411.12919)
- **What's New**: 이 논문에서는 자가 감독된 잡음 제거(self-supervised denoising)를 딥러닝 기반 재구성 방법의 전처리 단계로 통합했을 때의 효과를 연구하였습니다. 고무적인 점은, 저소음 참조 데이터의 필요성을 줄이고도 MRI 재구성의 품질을 향상시킬 수 있다는 것입니다. 연구에서는 가우시안 잡음으로 손상된 k-공간 데이터를 사용하여 진단 정확도를 높일 수 있는 가능성을 보여었습니다.

- **Technical Details**: 다양한 MRI 데이터의 특징을 반영하기 위해 일반화된 스타인 비편향 위험 추정(Generalized Stein's Unbiased Risk Estimate, GSURE)을 기반으로 하는 잡음 제거 파이프라인을 개발하였습니다. 이 과정에서 Diffusion Probabilistic Models (DPMs)와 Model-Based Deep Learning (MoDL)을 활용하여 자가 감독적 잡음 제거가 DL 네트워크의 학습 성과에 미치는 영향을 평가하였습니다. 특히, 자기공명영상(MRI) 재구성을 위한 피험자 데이터를 포함하여, 다양한 신호 대 잡음비(signal-to-noise ratio, SNR) 조건에서 성능을 시험했습니다.

- **Performance Highlights**: 실험 결과, 자가 감독된 잡음 제거를 통해 MRI 재구성의 품질과 효율성이 크게 향상됨을 확인하였습니다. 예를 들어, DPM을 사용하는 경우, 잡음이 제거된 이미지를 이용해 DL 네트워크를 훈련했을 때, 정상화된 평균 제곱근 오차(normalized root mean squared error, NRMSE)와 구조적 유사도 지수(structural similarity index measure, SSIM), 피크 신호 대 잡음비(peak signal-to-noise ratio, PSNR)가 모든 SNR 수준에서 유의미하게 개선되었습니다. 이러한 결과는 다양한 조건에서도 자가 감독된 잡음 제거가 DL 기반 MRI 재구성 방법의 유효성을 높일 수 있음을 뒷받침합니다.



### Signformer is all you need: Towards Edge AI for Sign Languag (https://arxiv.org/abs/2411.12901)
Comments:
          Official Code at: this https URL

- **What's New**: 이번 논문에서는 Signformer라는 새로운 Sign Language Translation (SLT) 모델을 제안합니다. Signformer는 사전 훈련된 모델이나 자원 집약적인 기술에 의존하지 않고, 기존의 gloss-free 방식을 개선하여 Edge AI 환경에서 효율적인 실시간 사용이 가능합니다. 이로써 SLT 분야의 지속 가능성과 실용성을 재정의하고 있습니다.

- **Technical Details**: 이 연구는 기존의 Sign Language Transformer (SL-Transformer) 아키텍처를 기반으로 하여, 새로운 알고리즘 설계를 위해 다양한 수화 언어의 구조적 특성을 철저히 분석합니다. 이를 통해 변형된 컨볼루션, 주의(attention) 메커니즘 및 위치 인코딩을 활용하여 파라미터 수를 극적으로 줄이면서도 성능을 극대화할 수 있었습니다. Signformer 모델은 GLT와 비교해 1807배 더 작은 파라미터 수를 자랑하며, 기존의 여러 SOTA 모델에 비해 뛰어난 성능을 올리고 있습니다.

- **Performance Highlights**: Signformer는 gloss-free SLT 선두주자로 자리매김하면서, 2024년 기준으로 새로운 2위 성적을 기록했습니다. 특히 Signformer-Full 모델은 gloss-free Finest SignLLM보다 더 나은 테스트 세트 성능을 발휘하며, 0.57백만의 파라미터 수를 통해 TOP5에 등재되었습니다. 이는 SLT 분야에서 파라미터와 성능의 균형을 새롭게 정의하는 중요한 이정표가 됩니다.



### Problem-dependent convergence bounds for randomized linear gradient compression (https://arxiv.org/abs/2411.12898)
Comments:
          15 pages, 3 figures

- **What's New**: 이번 연구에서는 분산 최적화에서 모델 업데이트의 통신이 성능 저하의 원인이 될 수 있음을 언급하고, 이를 개선하기 위한 방법인 gradient compression을 제안합니다. 이 방법은 최적화 출력량을 증가시키는 데 기여하지만, 정보 손실로 인해 해답에 도달하는 데 필요한 반복 횟수가 증가하는 단점이 있습니다.

- **Technical Details**: 연구에서는 비볼록 확률적 최적화(context of non-convex stochastic optimization) 문제의 맥락에서 압축(iteration penalty)과 문제 구조 간의 상관관계가 어떻게 변하는지를 조사합니다. 특히, 압축과 복원이 무작위 행렬(random matrix)과의 곱셈으로 모델링될 수 있는 선형 압축 스킴(linear compression schemes)에 집중하며, 랜덤 직교행렬(random orthogonal matrices)과 무작위 가우시안 항목(random Gaussian entries)과 같은 여러 행렬 분포(distribution of matrices)를 고려합니다.

- **Performance Highlights**: 압축이 수렴(convergence)에 미치는 영향을 목표의 헤시안의 노름(norm of the Hessian)으로 정량화할 수 있음을 발견했습니다. 특정 경우에는 압축 성능이 저랭크 구조(low-rank structure)나 문제의 다른 스펙트럼 성질(spectral properties)과 관련이 있습니다. 이러한 경우, 연구 결과는 압축이 가져오는 패널티(penalty)가 최악의 경우 경계(worst-case bounds)에 비해 현저하게 줄어든다고 예측하며, 이를 이미지 분류 모델의 파인튜닝(fine-tuning) 등 여러 최적화 문제에서 검증하였습니다.



### NPGPT: Natural Product-Like Compound Generation with GPT-based Chemical Language Models (https://arxiv.org/abs/2411.12886)
- **What's New**: 본 연구는 화학 언어 모델을 자연 생성물 데이터셋에 대해 학습시켜 자연 생성물과 유사한 화합물을 생성할 수 있는 새로운 방법을 제안합니다. 이를 통해 자연 생성물의 구조적 다양성을 탐색하며, 의약품 개발 비용과 시간을 줄이는 데 기여할 수 있도록 구성된 모델을 소개합니다. 이 방법은 최근 딥러닝을 통한 분자 생성의 발전을 토대로 자연 생성물의 포괄적인 화학 공간을 탐색할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 자연 생성물 데이터셋을 기반으로 한 화학 언어 모델을 세밀하게 조정하는 과정을 수행하였습니다. 사용한 모델은 SMILES 및 SELFIES와 같은 분자의 문자열 표현을 처리하는 화학 언어 모델이며, COCONUT 데이터베이스를 이용하여 약 400,000개의 자연 생성물로부터 데이터를 처리했습니다. 모델은 smiles-gpt와 ChemGPT를 포함하며, 기존의 PubChem 데이터셋을 활용하여 사전 학습된 후 자연 생성물 데이터셋을 정제하여 최적화하였습니다.

- **Performance Highlights**: smiles-gpt는 이전 연구와 유사한 결과를 도출하며, 자연 생성물과 더 유사한 화합물을 생성하는 것으로 나타났습니다. ChemGPT는 높은 유효성을 기록했지만, FCD가 크게 나타나 자연 생성물의 분포를 정확히 반영하지 못했음을 보여줍니다. 전반적으로 smiles-gpt는 COCONUT에 더 가까운 분포를 생성하며, 그래프 시각화를 통해 이 모델의 성능이 개선된 것으로 확인되었습니다.



### Local Anti-Concentration Class: Logarithmic Regret for Greedy Linear Contextual Band (https://arxiv.org/abs/2411.12878)
Comments:
          NeurIPS2024

- **What's New**: 이 논문에서는 탐색이 필요 없는 탐욕적(greedy) 알고리즘의 성능 보장을 위해 'Local Anti-Concentration' (LAC)이라는 새로운 조건을 소개합니다. 이 조건을 통해 탐욕적 알고리즘이 더 높은 효율성을 달성할 수 있다는 것을 보여줍니다. LAC 조건은 Gaussian, exponential, uniform, Cauchy, Student's t 분포와 같은 다양한 분포 클래스에서 충족됨을 입증합니다.

- **Technical Details**: LAC 조건 하에 탐욕적 알고리즘은 선형 컨텍스트 밴디트(linear contextual bandit) 문제에서 누적 기대 후회(cumulative expected regret)가 $O(	ext{poly} 	ext{log} T)$로 제한됨을 보여줍니다. 이는 효율적인 탐욕적 알고리즘을 위한 분포 클래스의 범위를 크게 확장하는 데 기여합니다. 논문에서는 다양한 지수 가족(exponential family) 분포 및 그 절단된 변형들도 포함되어 있음을 강조합니다.

- **Performance Highlights**: 우리의 결과는 현재까지 알려진 탐욕적 알고리즘에 대해 서브라인적 후회(sublinear regret) 경계를 허용하는 가장 광범위한 분포 클래스 범위를 구축합니다. 또한, 선형관계 및 Poly-로가리즘(poly-logarithmic) 형태의 정밀한 후회를 달성하는 데 성공했습니다.



### Residual Vision Transformer (ResViT) Based Self-Supervised Learning Model for Brain Tumor Classification (https://arxiv.org/abs/2411.12874)
- **What's New**: 이번 논문은 MRI 기반 뇌종양 진단을 위한 생성적 자기지도 학습(self-supervised learning, SSL) 모델을 제안합니다. 두 단계로 구성된 이 모델은 첫 번째 단계에서 Residual Vision Transformer (ResViT)를 활용한 MRI 합성(pretext task)과 두 번째 단계에서 ResViT 기반 분류기(classifier) 모델로서의 미세 조정(fine-tuning)을 포함합니다. 이 접근법은 CNN과 ViT의 하이브리드 아키텍처를 사용하여 MRI 이미지를 통한 로컬 및 글로벌 특징을 효과적으로 활용합니다.

- **Technical Details**: 제안된 SSL 모델은 잦은 데이터 부족 문제를 해결하는데 중점을 두며, CNN과 ViT의 특징을 통합하여 MRI 이미지 합성 및 뇌종양 분류 작업을 수행합니다. 또한, 생성된 합성 MRI 이미지를 데이터 증강(data augmentation) 방법으로 활용하여 학습 세트를 균형 있게 유지합니다. 연구팀은 BraTs 2023, Figshare 및 Kaggle 데이터셋을 이용하여 모델 성능을 평가하고, 기존의 다양한 딥러닝 모델들과 비교 연구를 수행하였습니다.

- **Performance Highlights**: 제안된 SSL 모델은 BraTs 데이터셋에서는 90.56%, Figshare에서는 98.53%, Kaggle에서는 98.47%라는 높은 정확도로 뇌종양을 분류하는 성과를 보여주었습니다. 본 연구는 기존의 ImageNet 데이터셋에서 사전 훈련된 모델보다 MRI 데이터셋에서 훈련된 모델의 성능이 우수함을 나타내고, 임상적 적용이 가능한 신뢰할 수 있는 솔루션을 제공하고 있습니다. SSL, 미세 조정, 데이터 증강을 통한 모델의 효과적인 구성은 뇌 MRI 분석에서의 데이터 불충분 문제를 해결하는 데 중요한 기여를 합니다.



### From Text to Pose to Image: Improving Diffusion Model Control and Quality (https://arxiv.org/abs/2411.12872)
Comments:
          Published at the NeurIPS 2024 Workshop on Compositional Learning: Perspectives, Methods, and Paths Forward

- **What's New**: 최근 2년 동안 텍스트-투-이미지(diffusion models)를 이용한 이미지 생성 모델의 품질과 사용이 증가함에 따라, 출력 제어의 필요성이 대두되고 있습니다. 본 논문에서는 텍스트-투-포즈(text-to-pose, T2P) 생성 모델과 새로운 샘플링 알고리즘을 통해 이미지 생성의 두 가지 주요 문제를 해결했습니다. 특히, 본 기술은 인체 포즈의 높은 충실도를 위한 새로운 포즈 어댑터를 포함하고 있어, 텍스트-투-포즈-투-이미지(generative text-to-pose-to-image) 프레임워크를 가능하게 합니다.

- **Technical Details**: T2P 모델에서는 인체 포즈를 18개의 신체 부위, 42개의 손, 68개의 얼굴 포인트로 설명하고, CLIP을 기반으로한 메트릭을 통해 생성된 포즈의 품질을 평가합니다. 포즈 생성을 위해 텍스트 특징을 조건으로 하는 오토리그레시브(decoder-only transformer architecture)를 활용하였으며, 가우시안 혼합 모델(GMM)을 사용하여 포즈의 연속성을 학습합니다. 실험에서는 4M (포즈, 프롬프트) 쌍을 학습 데이터로 활용하여 T2P 모델의 성능을 검증하였습니다.

- **Performance Highlights**: T2P 모델은 COCO-Pose 벤치마크 데이터셋에서 KNN 방식보다 78%의 높은 성능을 보여주며, 텍스트 프롬프트에 대한 정렬 능력을 입증했습니다. 또한 새로운 포즈 어댑터는 얼굴과 손에 대한 포인트를 포함하여 기존 SOTA(SDXL-Tencent) 모델보다 향상된 포즈 정확도를 보여주었습니다. 그러나 이미지 품질은 여전히 기본 SDXL 모델에는 미치지 못하는 결과를 나타냈습니다.



### A new Input Convex Neural Network with application to options pricing (https://arxiv.org/abs/2411.12854)
Comments:
          29 pages

- **What's New**: 본 논문에서는 입력에 대한 볼록 함수(convex function)로 설계된 새로운 클래스의 신경망(neural networks)을 소개합니다. 볼록 함수는 자신이 지배하는 아핀 함수(affine functions)의 최소 상한(supremum)으로 표현될 수 있다는 원리를 활용합니다. 이러한 신경망은 특히 볼록 지불금(convex payoffs)을 가진 옵션 가격을 근사하는 데 적합합니다.

- **Technical Details**: 논문에서는 이 신경망의 아키텍처(architecture)와 근사 능력을 검증하는 이론적 수렴 경계(theoretical convergence bounds)를 상세히 설명합니다. 또한 네트워크 훈련을 개선하기 위한 \\emph{scrambling} 단계도 소개됩니다. 이 새로운 접근 방식은 전통적인 방법들보다 더 나은 결과를 보여줄 수 있습니다.

- **Performance Highlights**: 마지막으로, 이 네트워크가 Basket, Bermudan, Swing 옵션과 같은 볼록 지불금을 가진 세 가지 유형의 옵션 가격을 추정하는 데 있어 효과적임을 수치적으로 입증하였습니다. 이 연구는 향후 옵션 가격 책정 및 금융 모델링의 발전에 기여할 수 있을 것으로 기대됩니다.



### Data-to-Model Distillation: Data-Efficient Learning Framework (https://arxiv.org/abs/2411.12841)
Comments:
          Accepted in the 18th European Conference on Computer Vision (ECCV 2024), Milan, Italy, September 29 October 4, 2024

- **What's New**: 본 논문에서는 Data-to-Model Distillation (D2M)이라는 새로운 프레임워크를 제안합니다. D2M은 큰 규모의 실제 데이터셋의 지식을 사전 훈련된 생성 모델의 학습 가능한 파라미터로 증류하여, 다양한 비율의 증류를 위한 훈련 이미지를 효과적으로 생성할 수 있습니다. 이는 높은 해상도의 복잡한 데이터셋에도 효율적으로 확장할 수 있도록 설계되었습니다.

- **Technical Details**: D2M은 생성적 적대 신경망(GAN)의 매개변수 공간 내에서 합성 데이터를 매개변수화합니다. 이는 기존의 픽셀 공간에서 합성 데이터셋을 최적화하는 단점을 해결하며, 모델의 채널 주의 맵과 실제 및 생성 이미지 간의 예측을 최소화하는 모듈을 포함합니다. 이러한 접근법은 분류 성능을 향상시키기 위한 다양한 지식의 증류를 가능하게 합니다.

- **Performance Highlights**: D2M은 15개의 서로 다른 해상도 및 레이블 복잡성을 가진 데이터셋에서 우수한 성능을 보여주었으며, ImageNet-1K에서 고정된 저장 복잡도로 고해상도(128x128)의 데이터를 효과적으로 증류할 수 있음을 입증했습니다. D2M은 기존 데이터셋 증류 알고리즘에 비해 재증류 효율성과 교차 아키텍처 일반화에서 더 나은 성능을 보였습니다.



### Off-policy estimation with adaptively collected data: the power of online learning (https://arxiv.org/abs/2411.12786)
Comments:
          37 pages. Accepted to the 38th Annual Conference on Neural Information Processing Systems (NeurIPS 2024), Vancouver, British Columbia, Canada

- **What's New**: 이 논문은 적응형 데이터 수집을 통해 치료 효과의 선형 함수 추정을 다룹니다. 최신 causal inference 및 reinforcement learning 문헌에서 중요한 문제로 대두되고 있으며, 특히 off-policy evaluation(	extsf{OPE})와 평균 치료 효과(	extsf{ATE}) 추정에 응용됩니다. 이 연구는 기존의 비대칭적 이론과 다르게 비대칭적 성질을 가진 estimator의 연속적 오류 모델링을 통해 새로운 통찰을 제공합니다.

- **Technical Details**: 이 논문에서 제안하는 방법은 연속적으로 가중된 추정 오차를 기반으로 하는 AIPW 추정기군의 평균 제곱 오차(MSE)에 대한 상한을 수립하는 것입니다. 또한, 온라인 학습을 통해 치료 효과의 추정치를 생성하는 일반적인 축소 방식을 제안하며, 이를 통해 수치적 최적화를 수행합니다. 이 방안은 세 가지 구체적인 예를 통해 설명됩니다: 1) 표 형식의 사례, 2) 선형 함수 근사 사례, 3) 일반 함수 근사 사례.

- **Performance Highlights**: 결과적으로, 이 논문은 AIPW 추정기가 대규모 샘플 영역에서 최적성을 발휘함을 보여주는 로컬 미니맥스 하한을 제공합니다. 이는 기존 문헌에서 다루지 않았던 비대칭적 오류에 대한 새로운 관점을 제공합니다. 닫힌 형태로 제공되는 이론적 결과는 다양한 응용 분야에서 유용하게 활용될 수 있을 것으로 기대됩니다.



### Exploring Eye Tracking to Detect Cognitive Load in Complex Virtual Reality Training (https://arxiv.org/abs/2411.12771)
- **What's New**: 이번 연구는 고급 제조 분야에서의 가상 현실(VR) 훈련 시스템에서 사용자의 인지 부하(cognitive load)를 탐지하기 위한 연구를 진행하고 있습니다. 이 연구에서 제안하는 방법은 안구 추적(eye-tracking) 기반의 기계 학습(machine learning) 접근 방식을 사용하여, VR 환경 내에서의 복잡한 시공간 작업에서 인지 부하를 탐지하는 것입니다.

- **Technical Details**: 연구는 Unity 엔진(version 2020.03.34f1)을 활용하여 개발된 VR 훈련 시스템에서 진행되었습니다. 피험자는 VR-3 Varjo 헤드셋을 착용하고, 시선 방향과 동공 확장을 기반으로 한 데이터 수집이 이루어졌습니다. Multi-Layer Perceptron(MLP)과 Random Forest(RF) 모델이 인지 부하를 예측하기 위해 사용되었으며, 이를 통해 정밀도를 평가하였습니다.

- **Performance Highlights**: MLP 모델은 평가 데이터셋에서 0.84의 정확도 및 정밀도를 달성하여 높은 예측 능력을 보여주었습니다. 반면, RF 모델은 0.72의 정확도와 0.73의 정밀도를 기록했지만, 과적합(overfitting) 경향이 나타났습니다. 이 결과는 VR 훈련에서의 인지 부하 예측의 복잡성을 강조하며, 향후 더 많은 연구가 필요함을 시사합니다.



### CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization (https://arxiv.org/abs/2411.12768)
- **What's New**: 최근 연구 결과에 따르면, Large Language Models (LLMs)는 '백도어 공격(backdoor attacks)'에 취약하다는 것이 밝혀졌습니다. 이러한 공격은 적대자가 숨겨진 트리거를 삽입하여 모델의 응답을 조작하는 방식으로 이루어집니다. 본 논문에서는 Internal Consistency Regularization (CROW)라는 새로운 방어 기법을 소개하며, 이는 모델 훈련 중 일관성을 촉진하여 백도어 트리거로 인한 레이어 간 불일치를 해결합니다.

- **Technical Details**: CROW는 적대적 섭동(adversarial perturbations)과 정규화를 통해 내부 일관성(internal consistency)을 강화하여 백도어의 영향을 중화합니다. 이 방식은 클린 데이터(clean data) 세트만을 기반으로 하며, 클린 참조 모델이나 사전 트리거 지식이 필요하지 않아 다양한 LLM 아키텍처에서의 실용성을 높입니다. 또한, CROW는 레이어 간의 일관성을 정의하고 백도어가 이 일관성을 어떻게 교란시키는지를 명확히 하여 새로운 방어 기법의 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, CROW는 Llama-2, CodeLlama, Mistral-7B 등 다양한 모델에서 공격 성공률(ASR)을 일관되게 감소시키는 것으로 나타났습니다. 기존 방어 기법인 파인튜닝(finetuning), 가지치기(pruning), 양자화(quantization)와 비교할 때 성능이 월등히 개선되었으며, 모델의 생성 능력을 유지합니다. CROW는 100개의 클린 샘플로 4분 이내에 훈련을 완료할 수 있어 계산적으로도 효율적입니다.



### VayuBuddy: an LLM-Powered Chatbot to Democratize Air Quality Insights (https://arxiv.org/abs/2411.12760)
- **What's New**: 이 연구에서는 VayuBuddy라는 LLM 기반 챗봇 시스템을 소개합니다. VayuBuddy는 자연어로 질문을 받아들이고, Python 코드로 공기질 센서 데이터를 분석하여 자연어로 답변을 제공합니다. 이는 다양한 이해관계자들이 공기질 센서 데이터로부터 유용한 통찰을 얻는 데 도움을 줍니다.

- **Technical Details**: VayuBuddy는 인도 정부의 공기질 센서에서 수집된 데이터를 활용하며, 7개의 LLM의 성능을 45개의 다양한 질문-답변 쌍을 통해 비교합니다. 챗봇은 시각적 분석(visual analysis)도 생성할 수 있어, 라인 플롯(line-plots), 맵 플롯(map plot), 바 차트(bar charts) 등을 제공합니다. 표에 대한 처리 능력을 갖춘 LLM들이 강력한 도구로 떠오르고 있으며, 손쉽게 데이터를 요약, 쿼리 및 분석할 수 있습니다.

- **Performance Highlights**: 평가 결과에 따르면, LLama3.1 모델이 가장 높은 점수를 기록했습니다. Codestral 및 Mixtral 모델도 성능이 괜찮으나 특정한 프롬프트 엔지니어링이 필요합니다. 전반적으로 대부분의 모델이 오류가 없는 Python 코드를 생성할 수 있으며, 특정 질문에는 코드 생성이 실패한 경우도 있었지만, 전반적인 결과는 긍정적입니다.



### FedCL-Ensemble Learning: A Framework of Federated Continual Learning with Ensemble Transfer Learning Enhanced for Alzheimer's MRI Classifications while Preserving Privacy (https://arxiv.org/abs/2411.12756)
Comments:
          6 pages, 4 figures

- **What's New**: 이 연구는 알츠하이머 질병(Alzheimer's disease)의 분류(classification)를 위한 새로운 접근법을 제안합니다. 고급 딥 러닝 기술(deep learning techniques)을 활용하고, 안전한 데이터 처리 방법을 결합하여 향상된 성능을 보이는 모델을 개발했습니다. 특히, 전이 학습 모델(transfer learning models)을 사용하여 의료 이미지 데이터에서 고급 특징을 추출하는 데 중점을 두었습니다.

- **Technical Details**: 주요 기술로는 ResNet, ImageNet, VNet과 같은 전이 학습 모델을 사용하여 알츠하이머 관련 미세한 패턴을 감지할 수 있습니다. 이 모델은 데이터 소스의 다양성에 대해 강력한 특징을 추출할 수 있도록 조정되었습니다. 또한, 페더레이션 학습(federated learning) 접근법을 통합하여 데이터 개인 정보 보호를 보장하면서 분산된 모델의 이점을 최대한 활용합니다.

- **Performance Highlights**: 실험 결과는 알츠하이머 분류의 정확성을 향상시키는 데 기여하며, 안전하고 협력적인 건강 관리 데이터 분석을 위한 프레임워크도 제공합니다. 추가적으로, 데이터 전송 시 기밀성과 무결성을 보장하기 위해 암호 기반 encryption 메커니즘도 적용되었습니다. 이러한 결과는 예측 성능을 개선하고 환자 데이터를 공유하지 않으면서도 모델의 강력한 학습을 가능하게 합니다.



### Supervised Autoencoders with Fractionally Differentiated Features and Triple Barrier Labelling Enhance Predictions on Noisy Data (https://arxiv.org/abs/2411.12753)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2404.01866

- **What's New**: 이 논문은 금융 시계열 예측을 향상시키기 위해 감독된 오토인코더(supervised autoencoder, SAE)를 사용하여 투자 전략의 성능을 개선하는 방법에 대해 조사합니다. 특히, 노이즈 증강(noise augmentation)과 트리플 배리어 레이블링(triple barrier labeling)이 위험 조정 수익에 미치는 영향을 분석하며, 비트코인, 라이트코인, 이더리움을 거래자산으로 설정했습니다.

- **Technical Details**: 연구는 SAE-MLP 네트워크를 활용하여 알고리즘 투자 전략(Algorithmic Investment Strategy, AIS)을 개발하고, 데이터 증강(data augmentation)과 노이즈 제거(denoising) 방법을 통합했습니다. 트리플 배리어 레이블링을 통해 보다 정교한 분류(classification)를 수행함으로써 위험-보상(risk-reward) 지표를 향상시켰습니다. FRACTIONALLY DIFFERENCED 방법을 통해 시계열의 메모리를 유지하면서도 안정성을 보장하는 기술이 적용되었습니다.

- **Performance Highlights**: 연구 결과, 균형 잡힌 노이즈 증강과 병목 크기를 가진 감독된 오토인코더가 전략의 효과성을 상당히 향상시키는 것으로 나타났습니다. 그러나 과도한 노이즈와 큰 병목 크기는 성능에 부정적인 영향을 미칠 수 있습니다. 전체적으로 SAE-MLP 전략의 포트폴리오는 비슷한 수익률을 가진 매수-보유(buy-and-hold) 포트폴리오보다 더 나은 성과를 내는 것으로 확인되었습니다.



### FinBERT-BiLSTM: A Deep Learning Model for Predicting Volatile Cryptocurrency Market Prices Using Market Sentiment Dynamics (https://arxiv.org/abs/2411.12748)
- **What's New**: 이번 연구에서는 비트코인 및 이더리움과 같은 변동성이 큰 암호화폐 시장을 위해 FinBERT와 Bi-LSTM의 조합 모델을 제안하여 가격 예측 정확도를 향상시키고자 합니다. 전통적인 시간 시계열 모델에 감정 분석을 통합하여 투자자들이 불확실한 시장에서 보다 나은 판단을 할 수 있도록 지원하는 것이 본 연구의 핵심입니다. 이 접근 방식은 암호화폐 시장 예측에 있어 중요한 공백을 메우고 있습니다.

- **Technical Details**: 최근의 연구 결과를 바탕으로, Bidirectional Long Short-Term Memory (Bi-LSTM) 네트워크와 FinBERT의 결합 모델을 제안합니다. 이 모델은 긴 기간의 의존성을 캡처할 수 있는 LSTM의 장점과 텍스트 데이터의 맥락 정보를 처리할 수 있는 FinBERT의 능력을 통합합니다. 특히 이 모델은 금융 뉴스의 감정을 세밀하게 반영하여 시장 가격 변동 패턴을 추적하도록 설계되었습니다.

- **Performance Highlights**: 전반적인 성능 측면에서, 제안된 FinBERT-LSTM 모델은 시장 뉴스와 감정 데이터를 활용하여 전통적인 시간 시계열 기법보다 높은 예측 정확도를 나타냅니다. 기존의 LSTM 및 GRU 모델에 비해 Bi-LSTM의 양방향 정보 흐름이 특히 높은 변동성을 지닌 암호화폐 시장에서 예측 성능을 향상시키는데 기여합니다. 이러한 모델은 훈련 및 테스트 단계 모두에서 우수한 성과를 보였습니다.



### A Survey of Financial AI: Architectures, Advances and Open Challenges (https://arxiv.org/abs/2411.12747)
Comments:
          Full list of papers and summary slides are available at: this https URL

- **What's New**: 본 보고서는 인공지능(AI)을 활용한 금융 시장 예측, 포트폴리오 최적화, 자동 거래에서의 최근 발전을 체계적으로 분석하였습니다. 특히, 금융 타임 시리즈를 위한 foundation models, 시장 관계 모델링을 위한 graph-based architectures, 포트폴리오 최적화를 위한 hierarchical frameworks와 같은 주요 혁신을 점검하였고, 이로 인해 모델의 복잡성과 실제 제약 조건 간의 중요한 trade-offs를 조명하였습니다.

- **Technical Details**: 이 보고서는 금융 AI 응용 프로그램의 다양한 기술적 접근 방식을 포괄적으로 정리하였습니다. 이를 통해 deep learning 모델의 구조적 혁신, 훈련 및 최적화 분야의 발전, 그리고 실제 구현 및 확장성의 개선과 같은 세 가지 주요 방향을 제시하였습니다. 또한, 예측적 작업과 의사결정 작업 간의 수학적 기초를 통합하는 통일된 공식 프레임워크를 제공하여 예측 및 결정 문제를 포괄해 다루었습니다.

- **Performance Highlights**: 보고서는 실제 금융 애플리케이션과 학술 문헌을 철저히 분석하여, AI 아키텍처와 방법론의 혁신이 금융 모델링에 대한 보다 정교한 접근을 가능하게 한다는 것을 보여주었습니다. 또한 이 연구는 금융 결정 및 산업 관행에 대한 심도 있는 논의를 포함하여, 실제 구현 고려 사항을 강조하면서 이론적 진전을 다루고 있습니다. 미래 연구를 위한 주요 방향과 도전 과제가 제시되어 있으며, 이는 학계와 산업 간의 간극을 메우는 데 기여할 것입니다.



### A Review of Reinforcement Learning in Financial Applications (https://arxiv.org/abs/2411.12746)
- **What's New**: 최근 금융 분야에서 강화 학습(Reinforcement Learning, RL)을 활용한 연구가 증가하고 있습니다. 이 조사 논문은 RL의 금융 애플리케이션에 대한 포괄적인 연구를 제공하며, 기초 분석을 통해 기존 문헌에서 나타나는 공통 주제를 탐구합니다. RL의 성능에 영향을 미치는 주요 요인들과 적용 시의 도전 과제를 식별하고, 이를 극복하기 위한 최근 발전을 논의합니다.

- **Technical Details**: 이번 조사에서는 모델 프리(model-free) 및 모델 기반(model-based) RL 알고리즘을 소개합니다. 특히, 액터 전용(actor-only), 비평가 전용(critic-only), 액터-비평가(actor-critic) 방법론을 다룹니다. 다양한 알고리즘인 정책 기울기(Policy Gradient, PG), 근접 정책 최적화(Proximal Policy Optimization, PPO)가 포함되며, 이러한 알고리즘들은 금융 데이터 분석에 활용될 수 있는 능력을 제공합니다.

- **Performance Highlights**: RL은 금융 시장의 고유한 데이터 복잡성에 적합하며, 기존 방법에 비해 우수한 성능을 발휘할 수 있는 잠재력을 지니고 있습니다. 분석된 연구에 따르면, RL은 특히 포트폴리오 관리 및 최적 실행과 같은 핵심 영역에서 향상된 결과를 보였습니다. 하지만, 금융 데이터의 잡음 및 비정상성, 분포의 문제 등은 RL의 활용에 있어 여전히 해결해야 할 도전 과제로 남아 있습니다.



### How Much Data is Enough? Optimization of Data Collection for Artifact Detection in EEG Recordings (https://arxiv.org/abs/2411.11886)
Comments:
          Several changes of wording. Caption of figure 10 corrected

- **What's New**: 이 연구에서는 생물학적 데이터 수집의 효율성을 높이기 위해 딥러닝 기반의 아티팩트 검출을 이용하여 데이터 수집 설계를 최적화하는 절차를 제안합니다. 기존의 아티팩트 수집 방법은 직관적이고 개념 중심으로 이루어져 있으며, 적절한 아티팩트 선택 및 수량에 대한 정당성이 부족한 경향이 있습니다. 본 연구는 아티팩트 작업을 12개에서 3개로 줄이고, 이소메트릭 수축 작업의 반복 횟수도 줄이는 성과를 보였습니다.

- **Technical Details**: 본 연구에서는 아티팩트가 포함된 에포크(epochs)와 비아티팩트 에포크의 이진 분류를 통해 딥러닝 모델을 훈련합니다. 세 가지 서로 다른 신경망 아키텍처를 사용하여 아티팩트 검출을 수행하며, 클리닝 효율성을 보존하면서 데이터 수집 노력을 최소화하는 것을 목표로 합니다. 이를 위하여 EEG 채널의 위치를 활용하여 EMG 신호를 직접 도출하는 혁신적인 방법도 소개합니다.

- **Performance Highlights**: 연구에서 적용된 방법론 덕분에 데이터 수집 효율이 크게 개선되었습니다. 아티팩트 작업 수를 줄이고, 반복 횟수를 최소화하여 비용을 절감하면서도 신뢰할 수 있는 데이터 수집을 가능하게 하였습니다. 이로 인해 EEG 및 EMG 데이터 수집 과정에서의 비용 대비 효율이 향상되며, 미래 연구를 위해 보다 경제적이고 효과적인 데이터 수집을 위한 길잡이가 될 것입니다.



