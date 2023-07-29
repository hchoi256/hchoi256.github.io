---
layout: single
title: "[논문 분석] Swin Transformer (ICCV 2021)"
categories: AIPaperCV
tag: [Computer Vision, Transformer, Swin Transformer]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/swin.png
sidebar:
    nav: "docs"
---

[논문링크](https://arxiv.org/abs/2103.14030)

<!-- <span style="color:blue"> ???? </span> -->

****
# 한줄요약 ✔
- **다양한 Vision Task**에 적합한 구조.
- **Local Window**를 적용하여 inductive bias 완화.
- **Patch Merging**을 통해 레이어 간 계층적 구조를 형성하여 이미지 특성을 고려합니다.

> **이미지 특성**: 해상도 혹은 전체 이미지에 존재하는 객체의 크기 및 형태.

****
# Preliminaries 🍱
## ViT 모델
![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/8204eabe-c472-4f9b-b289-f2c22c8f41b3)

ViT 논문에 대한 설명은 [여기](https://hchoi256.github.io/aipapercv/vit/)를 참조해주세요.

## Local Window
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/45a5c7cf-98fe-4104-951c-8ab940a73f95)

- $$H$$: 이미지 세로 길이.
- $$W$$: 이미지 가로 길이.
- $$C$$: 이미지 채널수.
- $$P_h$$: 패치 세로 길이.
- $$P_w$$: 패치 가로 길이.
- $$N$$: 패치 개수.
- $$M$$: local window 크기.
- $$n$$: local window 개수.

Local window 방식은 입력 이미지를 작은 패치로 분할하고, 이웃한 패치들 간에만 self-attention을 적용하여 지역적인 상호작용을 강조하는 Transformer의 변형 방법입니다.

이를 통해 시퀀스 간의 관계를 모델링하면서 지역성을 보존하여 inductive bias를 보완하고, 큰 이미지에 대한 효율적인 처리를 가능케합니다.

****
# Problem Definition ✏
                Given a 2D image dataset

                Return a more efficient Transformer-based Vision model on the dataset

                Such that it outperforms the performance of the original model in terms of inference time while retaining accuracy.

****
# Challenges and Main Idea💣
**C1)** <span style="color:orange"> 기존 ViT 모델은 Classification Task를 풀기 위한 모델로 제안되었습니다.</span>

**C2)** <span style="color:orange"> 기존 ViT 모델은 텍스트와 달리 이미지를 위한 특성이 없습니다. </span>

**C3)** <span style="color:orange"> 기존 ViT 모델은 입력 Token의 개수가 증가함에 따라 Transformer 구조상 quadratic한 Time Complexity를 갖습니다. </span>

**Idea)** <span style="color:lightgreen"> 본 논문의 Swin Transformer는 **Local Window** 및 **Patch Merging**을 도입하여 상기 문제들 해결합니다. </span>

****
# Proposed Method 🧿
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/fd4b782a-d397-43d7-ac5f-88f3ae4ef604)

## Hierarchical Local Window
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/31cf153c-2841-4e3a-9f16-5bc1671472cb)

모든 레이어에서 동일한 패치 크기를 공유하는 기존 ViT 모델과 달리, Swin Transformer는 높은 레이어로 갈수록 큰 크기의 패치를 사용합니다.

Swin Transformer에서 높은 레이어로 갈수록 큰 크기의 패치를 사용하는 것은 몇 가지 효과를 가져옵니다:
- **Hierarchical Representation**: 높은 레이어에서 큰 크기의 패치를 사용함으로써 모델은 더 넓은 영역의 정보를 바라볼 수 있습니다. 이는 입력 이미지의 다양한 전역적 특징을 포착하는데 도움이 되며, 다양한 추상적인 개념을 학습할 수 있게 합니다.
- **하위 레이어에서 미세한 특징 감지**: Swin Transformer는 저수준 레이어에서는 작은 패치를 사용하여 미세한 특징을 감지하고, 이러한 특징을 높은 레이어로 전달하면서 점차 더 큰 패치를 사용하여 높은 수준의 추상적 특징을 학습할 수 있습니다.

하여 Local Window의 사용은 inductive bias를 강화하며, 이미지 데이터에서 유용한 특징을 추출하는 데 도움이 됩니다.

한 가지 특이점으로 Swin Transformer는 ViT와 달리 [CLS] 토큰을 사용하지 않고, **마지막 레이어에서 모든 패치의 정보의 평균값**을 사용합니다.

전체 이미지의 정보를 평균하여 사용하기 때문에, 전역적인 이미지 정보를 잘 반영하면서도 [CLS] 토큰을 사용하지 않을 수 있는 장점이 있습니다.

$$\Omega (MSA)=4hwC^2 + 2(hw)^2C$$

$$\Omega (W-MSA)=4hwC^2 + 2M^2 hwC$$

- $$\Omega(MSA)$$: 기존 MSA의 계산 복잡도를 나타냅니다.
- $$\Omega(W-MSA)$$: Swin Transformer에서 로컬 윈도우 기반의 계산 복잡도를 나타냅니다.
- $$h$$: self-attention 헤드(heads)의 개수,
- $$w$$: 패치의 가로 크기,
- $$C$$: 패치 임베딩의 차원,
- $$M$$: 로컬 윈도우 크기입니다.

Swin Transformer에서는 기존 MSA와 비교하여 로컬 윈도우에 대해 선형 복잡도를 가지기 때문에, 윈도우 크기에 비례하여 계산 복잡도가 증가하지 않습니다.

## Patch Merging
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/9c75e864-ca74-4460-8dc7-aaeb0466d3ca)

- $$C$$: 채널 사이즈.

**Patch Merging**은 가령 상기 이미지에서 $$(2,2)$$ 사이즈의 패치 정보를 가져와서 하나의 차원으로 축소해주는 과정입니다.

Linear Reduction을 통해 다시 $$2C$$ 차원으로 축소하여 계산 복잡도를 줄이며, 이렇게 얻어진 결과들을 합쳐줍니다.

## Swin Transformer Block
### (1) Window-Multi-head Self Attention (W-MSA)
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/786978c9-cc3a-4e3c-a141-ca9a7b88e01e)

하나의 Local Window 안에 존재하는 패치끼리 self attention을 수행합니다.

각각의 패치로 구성된 이미지를 윈도우로 나누어서 각 윈도우 별도로 셀프 어텐션을 수행합니다.

로컬 윈도우가 나눠진 개수만큼 셀프 어텐션의 횟수가 증가합니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/5cb4df0c-da22-4ff2-b959-70478812f8c2)

상기 이미지처럼 서로 다른 로컬 윈도우 간에 각각 셀프 어텐션을 수행합니다.

전체 이미지에서 셀프 어텐션을 했을 때보다 반복해서 여러 번 셀프 어텐션을 수행해야 하는 애로 사항을 타파하고자 본 논문은 **Efficient Batch Computation**을 제안합니다.

#### Efficient Batch Computation
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/25bafbf6-2ac0-412a-918d-b939f9f7756e)

각 윈도우 사이즈만큼 전체 이미지를 분할하고, 그들을 Batch와 같은 차원으로 합쳐주어 한 번에 병렬처리로 연산하여 반복 연산을 피하고 병렬 처리로 효율적으로 연산 속도를 높입니다.

#### Relative Position Bias
$$Attn(Q,K,V)=SoftMax({QK^T \over \sqrt{d}} + B)V$$

- $$B \in \mathbb{R}^{M^2 \times M^2}$$: Relative Position Bias.

Swin Transformer에서는 패치들 간의 상대적인 위치 정보를 수집하여 저장합니다. 이 정보를 활용하면 패치 간 거리에 따라 가중치를 부여하여 자연어 처리에서 사용되는 어텐션 메커니즘과 유사하게, 이미지 내에서 더 먼 패치들과의 상호작용을 조절할 수 있습니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/57df2b25-59d2-4b69-87ce-b5ecc9ccfa88)

<span style="color:orange"> 기존 **Bias Index Matrix** $$(\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)})$$는 패치 간의 상대적인 위치에 대한 편향 정보의 차원이 $$2M-1$$로 작아서 보다 정확한 편향 정보를 끌어내기 어렵습니다. </span>

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/3861b469-5cef-40b7-9ca8-2cfc59c8cde9)

<span style="color:lightgreen"> 하여 본 논문에서는 보다 적은 학습 파라미터로 넓은 범위의 Relative Position Bias$$(B \in \mathbb{R}^{M^2 \times M^2})$$를 나타낼 수 있습니다. </span>

이 때 **Relative Position Bias**는 패치 간 상대적인 위치에 대한 편향 정보를 나타내는 개념으로, self-attention 메커니즘에 적용되어 **패치 간의 상대적인 위치에 따른 중요도**를 반영합니다.

이를 통해 Swin Transformer는 긴 범위의 이미지 정보를 효과적으로 학습하면서도 연산 복잡도를 크게 늘리지 않고 성능을 개선할 수 있습니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/d946962c-72e5-46cd-8370-027d870f352d)

상기 이미지에서 $$x$$축과 $$y$$축을 기점으로 패치 간 상대적인 위치에 대한 편향 정보를 계산합니다.

이후, 하기 전개를 따라 최종 Relative Position Bias를 얻게 됩니다.

            # Step 1
            x_axis_matrix += window_size -1 # index가 0부터 시작되도록 변환
            y_axis_matrix += window_size -1 # index가 0부터 시작되도록 변환

            # Step 2
            x_axis_matrix *= 2 * window_size - 1
            relative_position_M = x_axis_matrix + y_axis_matrix

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/18fbb450-e3c8-48a8-95a0-d83282c2f4d2)

### (2) Shifted-Window-Multi-head Self Attention (SW-MSA)
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/9628b2f2-0685-4747-8c74-e602ee33e815)

다수의 Local Window 간의 self attention을 수행합니다.

이전 레이어의 결과와 달리, 연결되지 않았던 부분들에 대해서 윈도우를 형성합니다.



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
[DSBA Lab](https://www.youtube.com/watch?v=2lZvuU_IIMA)