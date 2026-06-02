New uploads on arXiv(cs.LG)

### A Multi-Evidence Framework Rescues Low-Power Prognostic Signals and Rejects Statistical Artifacts in Cancer Genomics (https://arxiv.org/abs/2510.18571)
Comments:
          17 pages (main text), 4 figures (main text), 7 supplementary figures, 4 supplementary tables. Focuses on a computational framework using causal inference and biological validation for underpowered cancer genomic studies

- **What's New**: 이번 연구에서는 전통적인 유전체 연관 연구가 통계적 유의성에 의존하며, 권장 연관 분석에서 발생할 수 있는 낮은 검사 능력의 문제를 해결하기 위한 새로운 방법론을 제안합니다. 특히 TCGA 유방암 데이터셋을 활용하여 기존 방법론의 한계를 극복하고, 유전자 분석의 정확성을 높이기 위해 다섯 가지 기준의 계산 프레임워크를 개발하였습니다.

- **Technical Details**: 이 프레임워크는 인과 추론(causal inference) 기법인 역확률 가중치(inverse probability weighting)와 이중 견고 추정(doubly robust estimation) 기술을 통합하고, 생물학적 검증(biological validation) 방법과 결합하여 결과의 신뢰성을 높입니다. TCGA-BRCA의 사망률 분석에 적용했을 때, 기존 방법에서는 유의미한 유전자를 찾지 못했으나, 새로운 프레임워크는 검증이 필요한 후보 유전자를 판별해냈습니다.

- **Performance Highlights**: 이 연구의 결과, KMT2C 유전자가 보이는 통계적 유의성은 경계선에 위치하였지만, 강력한 생물학적 증거를 바탕으로 더욱 깊은 검증이 필요하다는 것을 보여주었습니다. 또한 프레임워크가 유전자 신호와 아티팩트를 구분하는 데 성공했음을 확인하였고, 이는 생물학적 해석 가능성을 통계적 유의성보다 우선시하는 분석 접근 방식을 제시합니다.



