---
layout: single
title: "[논문 분석] Vision Transformer With Deformable Attention (CVPR 2022)"
categories: AIPaperCV
tag: [Computer Vision, ViT, Transformer, Deformable Attention]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/vit.png
sidebar:
    nav: "docs"
---

[논문링크](https://arxiv.org/abs/2201.00520)

<!-- <span style="color:blue"> ???? </span> -->

****
# 한줄요약 ✔
- Deformable Convolution의 Offset 개념을 ViT의 Self-Attention에 적용한 Deformable Attention Module을 제안.
- 기존 Swin Transformer(ICCV`21) 및 Deformable DETR(CoRL`21)과 달리 **Query Agnostic**한 형태로 Patch 개수보다 적은 Reference Points를 통해 Attention 연산.
    - Query Agnostic: 모든 Query가 하이퍼 파라미터로 고정된 개수의 Reference Points를 공유하여 Offset을 업데이트합니다. 

****
# Preliminaries 🍱
## Swin Transformer
[여기](https://hchoi256.github.io/aipapercv/swin-transformer/)

## Deformable Convolution
[여기](https://hchoi256.github.io/aipapercv/od-deformable-detr/#deformable-convolution)

## Deformable DETR
[여기](https://hchoi256.github.io/aipapercv/od-deformable-detr/)

## DeepViT
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/39c66335-4b4a-422c-97be-96b446f61445)

DeepViT(arXiv`21) 논문에서 기존 ViT 구조는 CNN과 달리 깊이가 깊어져도 성능 향상이 어렵다는 점을 지적했습니다.

상기 이미지에서<span style="color:red"> 빨간점</span>은 각기 다른 Query의 위치이며, 궁극적으로 깊이가 깊어지면 다른 Query에 대한 Self-Attention Map이 거진 동일하게 생성되는 모습입니다.

이것은 Layer가 깊어져도 거진 유사한 Self-Attention Map이 학습되는 문제점(**Attention Collapse**)을 유발합니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/de2ef356-afd9-4780-9f32-7e74be5d62de)

하여 본 논문은 **Re-Attention 구조**를 제안하여 Learnable Matrix로 Self-Attention Map을 섞어주어 깊이가 더 깊어져도 다른 Self-Attention Map이 생성합니다.

Learnable Matrix는 학습 가능한 가중치 파라미터로, 여러 Query들 간의 관계를 조절하는 역할을 수행합니다. 

이로 인해 깊이가 깊어져도 다양한 Self-Attention Map이 생성되어 다양한 특징을 학습하고 입력 데이터(Query)의 다양한 관점을 고려하여 특징을 추출할 수 있습니다.

이는 **query-dependent**한 특징 추출을 개선하여 모델의 성능을 향상시키는데 도움이 됩니다.

가령, 상기 이미지에서 Block 23 깊이에서 Self-Attention Map은 **Uniform**해지기 시작하지만, Re-Attention에서는 Block 30 깊이 정도에서 Uniform해지기 시작합니다.

하나 DeepViT 또한 어느 정도 깊이에서는 다시금 Attention Collapse 현상이 발생했습니다.

****
# Challenges and Main Idea💣
**C1)** <span style="color:orange"> CNN 기반의 객체탐지 모델은 고정된 크기의 Kernel를 사용한 합성곱 연산을 수행하여 큰 객체탐지 성능이 저조합니다. </span>

**Idea 1)** <span style="color:lightgreen"> Deformable Attention Module을 통해 객체에 특화된 수용 영역을 형성하여 학습합니다. </span>

**C2)** <span style="color:orange"> DeepViT에서 언급된 Global Attention 관점에서 결국 모두 동일한 Self-Attention Map으로 수렴하기 때문에, 기존의 Deformable 객체탐지 기법들이 각 Query마다 독립적인 Reference Points를 생성하는 과정은 연산 측면에서 비효율적입니다. </span>

**Idea 2)** <span style="color:lightgreen"> 모든 Query가 동일한 Reference Points를 통해 Attention 연산을 하도록 합니다. </span>

****
# Proposed Method 🧿

****
# Experiment 👀

****
# Open Reivew 💗

****
# Discussion 🍟

****
# Major Takeaways 😃

****
# Conclusion ✨
## Strength
## Weakness

****
# Reference