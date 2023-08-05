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
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/eb28ad9a-8050-4057-8c5c-4ff82da474fd)

- Backbone으로 Swin Transformer 구조를 따르며, 마지막 2개의 Stage에서 Deformable Attention Module을 사용합니다.
- Deformable Convolution의 Offset 개념을 ViT의 Self-Attention에 적용한 Deformable Attention Module을 제안.
- 기존 Swin Transformer(ICCV'21) 및 Deformable DETR(CoRL'21)과 달리 **Query Agnostic**한 형태로 Patch 개수보다 적은 Reference Points를 통해 Attention 연산.
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


**C3)** <span style="color:orange"> Deformable DETR의 경우 Deformable Attention을 통해 small sets of key sampling points만을 이용하여 Attention 연산을 하지만, Multi-Scale Feature Maps를 사용하는 경우 Exponential 연산량 증가가 문제가 됩니다. </span>

**Idea 2)** <span style="color:lightgreen"> 본 논문은 Swin Transformer을 Backbone으로 취하여 Multi-Scale Feature Map에 대한 연산량을 선형 복잡도로 줄입니다. </span>

****
# Proposed Method 🧿
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/e7cb5c40-5f93-4084-9143-5317aed44d65)

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/985e10e3-5a9c-4be2-a34d-dfeb19b96628)

DMHA 모듈은 Offset 계산을 위해 상기 연산 복잡도에서 Offset Network 부분만 추가된 모습이며, 해당 과정의 연산량은 $$6%$$ 밖에 증가하지 않습니다.

## Deformable Attention Module
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/11f29c0a-cfa4-46bb-9c8b-53b4d36f251c)

- $$x$$: Input Feature Map.
- $$q$$: Query.
- $$\tilde{k}, \tilde{v}$$: deformed key and value.
- $$\phi$$: sampling function.
- $$W_q,W_k,W_v$$: Query, Key, Value 가중치 행렬.
- Reference Points: 하이퍼 파라미터로 Patch 개수보다 적도록 Input Feature Map 상에 Uniform하게 생성됩니다.
- $$\theta_{offset}$$: offset network.

## 동작 과정
1) Input Feature Map $$x \in (B,C, H, W)$$을 마치 MSHA처럼 Channel-Wise하게 Group $$G$$ 단위로 $$x \in (B,G \times C, H, W) \rightarrow (B \times G, C, H,W)$$ 분할하여 **Deformed Points의 다양성을 향상시킵니다**.

- Head의 개수 $$M$$은 $$G$$의 배수로 설정되어 각 Group이 다수의 attention head를 통해 연산될 수 있도록 합니다. 
- 각 Group마다 shared subnetwork를 통해 offsets를 계산합니다.

2) $$x \in (B \times G, C, H,W)$$에 대해 **Reference Points $$p \in (B \times G, H_G,W_G,2)$$를 Uniform Grid $$((H,W) \rightarrow (H_G,W_G))$$로 생성하고 각 좌표를 $$[-1,1]$$ 범위로 Normalization 합니다**.

- $$H_G={H \over r}, W_G={W \over r}$$.
    - $$r$$: a factor.
- Reference Points: $$\{(0,0),...,(H_G-1,W_G-1)\}$$ 좌표들을 $$[-1,+1]$$ 범위로 Normalization을 수행합니다.
    - Normalization 이후, $$(-1,-1)$$ 좌표는 top-left corner이 되게 됩니다.
    - 각 Reference Point 마다 $$2$$개의 $$x$$축과 $$y$$축 Sampling Offset을 가집니다.

3) Input Feature Map을 $$W_q$$과 곱하여 **쿼리 $$q$$를 구합니다**.

<span style="color:yellow"> $$q=xW_q$$ </span>

4) 해당 쿼리 $$q$$를 Offset Network $$\theta_{offset}$$에 넣어서 Reference Points에 대한 **Offset Vector를 구합니다**.

<span style="color:yellow">$$\bigtriangleup \textbf{p}=\theta_{offset}(q)$$ </span>

5) 해당 Offset Vector과 Reference Points들을 더하여 **Deformed Points(Sampling Points)를 구합니다**.

<span style="color:yellow">$$\tilde{x}=\phi(x;p+\bigtriangleup p)$$ </span>

6) 해당 Deformed Points들은 실수값을 갖고 있기 때문에 **Bilinear Interpolation을 통해 알맞은 Sampled Features를 추출합니다**.

<span style="color:yellow">$$\phi(z;(p_x,p_y))=\Sigma_{r_x,r_y}g(p_x,r_x) \cdot g(p_y,r_y) \cdot z[r_y,r_x,:]$$ </span>

<span style="color:yellow"> $$g(a,b)=max(0,1-\left\vert a-b \right\vert)$$ </span>

- $$r_x,r_y$$: indexes on $$z \in \mathbb{R}^{H \times W \times C}$$.

7) 해당 Sampled Features들을 각각 $$W_k,W_v$$와 곱하여 Deformed Key $$\tilde{k}$$와 Deformed Value $$\tilde{v}$$를 구합니다.

<span style="color:yellow">$$\tilde{k}=\tilde{x}W_k,\tilde{v}=\tilde{x}W_v$$ </span>

8) Deformed Points로 부터 Swin Transformer에서 제시된 **Relative Position Bias Offsets을 동일한 방법으로 구하고, 최종 Attention 출력을 계산합니다**.

<span style="color:yellow">$$z^m=\sigma({q^m (\tilde{k}^{m})^T \over \sqrt{d}}+\phi(\hat{B};R)W)\tilde{v}^m$$ </span>

- $$z^m$$: $$m$$번째 head의 Attention 결과.
- $$\phi(\hat{B};R) \in \mathbb{R}^{HW \times H_G W_G}$$: Relative Position Bias Offset.
    - $$G$$: Group.
    - $$\hat{B}$$: Relative Position Bias.

## Offset Network
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/43ce0b3d-86d8-4638-8063-416863e17e6e)

Offset Network는 Query Feature를 입력으로 각 Reference Point에 해당하는 Offset Value를 예측합니다.

또한, Offset Network는 **Depthwwise Separable Convolution, GELU, 1x1 Convolution**으로 구성됩니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/7b46d1c5-7046-45cb-9ee3-78104de010f3)

상기 이미지는 Depthwise Separable Convolution을 시각화하며, 이는 Channel 단위로 Convolution을 우선 진행한 후, Pointwise Convolution을 진행합니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/89dca30d-bb7c-49ff-a800-a1f78164b92d)

상기 이미지에서 Standard Convolution에 비해 연산 복잡도가 $${1 \over N} + {1 \over D^2_k}$$ 만큼 줄어든 모습입니다.

<span style="color:yellow">$$\bigtriangleup p \leftarrow s \cdot tanh(\bigtriangleup p)$$ </span>

상기 수식을 통해 Reference Point는 $$x \times s$$ Region에서 정의되도록 하여 Offset이 범위를 초과하여 변형시키지 않도록 제한합니다.

****
# Experiment 👀
## SOTA Performance
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/8d37a810-f7e3-47d5-8d20-c6574eb92793)

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/79a85476-3654-4742-9633-97a10c66e061)

## Ablation
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/030fbf3d-5b98-4a75-ae00-efb665f79047)

## Visualization
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/f141c1c6-d5d3-432e-8865-bcfadb6fc057)

상기 이미지는 Swin Transformer 시각화 결과이며, 보시는 것처럼 한정된 영역에서 객체탐지를 하는 모습입니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/a605d796-6ad7-4880-a458-55041ac6b69e)

하나 본 논문의 DAT는 Reference Point를 Deformable하게 만듦으로써 Recognition 성능을 향상시키면서도, Swin Transformer의 선형 연산 복잡도를 유지하였습니다.

****
# Open Reivew 💗
NA

****
# Discussion 🍟
NA

****
# Major Takeaways 😃
- Query Agnostic

****
# Conclusion ✨
NA

****
# Reference
NA