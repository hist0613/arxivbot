New uploads on arXiv(cs.AI)

### EvidenceMap: Learning Evidence Analysis to Unleash the Power of Small Language Models for Biomedical Question Answering (https://arxiv.org/abs/2501.12746)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 연구에서는 EvidenceMap이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 생물 의학 질문에 대해 다양한 증거를 수집, 평가, 요약하는 과정을 통해 보다 나은 답변 생성을 목표로 합니다. EvidenceMap은 상대적으로 작은 미리 훈련된 언어 모델을 기반으로 하여, 증거 분석을 학습하고 이를 통해 효과적으로 텍스트 응답을 생성하는 능력을 강화합니다.

- **Technical Details**: EvidenceMap은 세 가지 주요 정보 유형, 즉 증거가 질문을 지원하는 평가(ℛe⁢v⁢a⁢l), 증거 간의 상관관계(ℛc⁢o⁢r), 및 모든 증거의 요약(Es⁢u⁢m)을 활용합니다. 이 프레임워크는 작은 언어 모델을 통해 각 질문에 대한 증거 맵을 구성하고, 이로 인해 다양한 분석의 구성 요소를 학습합니다. 목표는 의료 질문에 대해 최적의 응답을 생성할 수 있는 방식으로 증거 분석을 학습하는 것입니다.

- **Performance Highlights**: 실험 결과, 66M 파라미터로 미세 조정한 모델이 8B LLM을 사용한 RAG 방법보다 기준 품질에서 19.9%, 정확도에서 5.7% 더 높은 성능을 보여줍니다. 이러한 결과는 작은 모델에서도 효과적인 증거 분석 학습이 가능함을 시사합니다. 결국, EvidenceMap은 생물 의학 질문-응답 성능을 크게 향상시키는 방법으로 자리잡게 될 것입니다.



### How Should I Build A Benchmark? Revisiting Code-Related Benchmarks For LLMs (https://arxiv.org/abs/2501.10711)
Comments:
          42 pages

- **What's New**: 이번 논문에서는 How2Bench라는 새로운 코드 관련 기준 체크리스트를 제안합니다. 이 체크리스트는 55개의 기준으로 구성되어 있으며, 코드 관련 벤치마크의 개발을 포괄적으로 관리합니다. How2Bench는 벤치마크의 생애 주기를 아우르며 품질 보증과 데이터 공개의 중요성을 강조합니다.

- **Technical Details**: How2Bench는 코드 관련 벤치마크 개발을 위한 가이드라인을 제공하며, 여러 가지 기준을 적용하여 현재의 벤치마크에서 발생하는 문제를 분석했습니다. 274개의 벤치마크를 프로파일링한 결과, 데이터 품질 보증이 부족하거나 정보 공개가 불충분한 경우가 많았습니다. 이는 벤치마크의 신뢰성에 대한 우려를 불러일으킵니다.

- **Performance Highlights**: How2Bench는 개발자와 연구자들이 벤치마크 품질을 향상시킬 수 있는 유용한 도구로 기능합니다. 실험 결과 49명의 참가자 모두가 품질 향상을 위한 체크리스트의 필요성에 동의했으며, 55개의 기준의 중요성을 인정했습니다. 이 연구는 벤치마크 개발에서 표준 관행의 문제점을 알리는 데 기여할 것입니다.



