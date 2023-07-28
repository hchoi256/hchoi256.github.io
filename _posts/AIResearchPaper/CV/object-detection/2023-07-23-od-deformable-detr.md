---
layout: single
title: "[논문 분석] Deformable DETR (ICLR 2021)"
categories: AIPaperCV
tag: [Computer Vision, Object Detection, DETR, Transformer]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/od.png
sidebar:
    nav: "docs"
---

<!-- <span style="color:blue"> ???? </span> -->

[**논문**](https://arxiv.org/abs/2010.04159)

****
# 한줄요약 ✔
![img](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/97d64f9d-c010-429b-94a5-64286fac5516)

- DETR 모델에서 Transformer 아키텍처의 한계 해결
    - slow convergence $$\rightarrow$$ **deformable attention**(key sampling points)로 해결.
    - 작은 물체 객체 탐지 성능 저조 $$\rightarrow$$ **multi-scale features**로 해결.
- $$\times 10$$ 배 빠른 학습 속도.
- Object query 개수 = 300.
- 기존 DETR과 동일한 헝가리안 손실 함수 사용.

****
# Preliminaries 🙌
## DETR 모델 한계
- **느린 수렴 속도**
    - DETR은 초기화 단계에서 CNN이 출력한 하나의 feature map의 모든 pixels에 대해 uniform attention weights를 부여합니다. 이는 모든 픽셀이 동일한 중요도를 가진다고 가정하는 것인데, 실제로는 객체가 존재하는 영역과 배경이나 노이즈가 있는 영역 등에 대해 다른 중요도를 가지는 것이 보통입니다. <span style="color:orange"> 하여 객체의 위치를 다소 부정확하게 판단하게 되는 **sparse meaningful locations** 현상이 발생하고, 수렴하기 까지 더 많은 시간이 소요됩니다 </span>.
    - DETR은 Transformer 인코더에서 각 픽셀들은 서로 다른 픽셀들과의 attention weight를 계산한다.
        - <span style="color:orange"> Quadratic time complexity ($$N^2$$) </span>.
            - $$N$$: the number of pixels.
- **작은 물체에 대한 객체 탐지 성능 저조**
    - 작은 물체 탐지를 위해 **high-resolution feature maps**가 중요합니다.
        - <span style="color:orange"> DETR은 Transformer 기반의 모델로 CNN 기반 모델보다 더 복잡한 구조를 가지고 있기 떄문에, 높은 해상도의 특징 맵을 처리하는데 더 많은 메모리와 계산량이 필요하게 됩니다. </span>
    - 작은 물체 탐지를 위해 **multi-scale features** 활용이 중요합니다.
        - 작은 객체는 이미지에서 크기가 작고 미세한 구조를 가지는 경우가 많습니다. Multiscale 피처맵은 다양한 해상도를 가지며, 작은 객체를 더 잘 포착하기 위해 다양한 크기의 특징을 제공합니다.
        - <span style="color:orange"> DETR은 multi-scale features가 아닌 동일한 크기의 patch로 features를 생성합니다. </span>

## Deformation
기존의 일반적인 컨볼루션은 사각형의 그리드 형태의 고정된 필터를 사용하여 이미지의 공간적인 특징을 인식하는데, 이는 일부 객체의 형태가 사각형이 아닌 **비정형적인 형태**일 때 문제가 될 수 있습니다.

하여 입력 피처맵에 대해 deformation 방식을 적용하여 적응형 수용 영역을 생성합니다.
- <span style="color:lightgreen"> **sparse spatial locations**을 통해 sparse meaningful locations 현상 해결 가능 </span>
    - sparse spatial locations: 특정 이미지나 피처맵에서 일부 픽셀 또는 위치만을 선택하는 것을 의미합니다.
    - sparse meaningful locations: 적은 양의 의미있는 픽셀 또는 위치를 선택하는 것을 의미합니다. 
- <span style="color:orange"> **element relation modeling**이 약함 </span>
    - element relation modeling: 입력 이미지의 pixels들 간의 상대적인 관계 모델링입니다.
        - 객체의 위치, 크기, 클래스 등을 파악하는 데에 용이합니다.
    - Transformer는 **self-attention을 통해 입력 시퀀스 간의 관계성 파악**으로 element relation modeling을 잘합니다.
    - Deformable 방식은 입력 시퀀스 간의 관계가 아닌 **주변 픽셀들의 locality**를 조사하기 때문에 element relation modeling이 약합니다.

## Deformable Convolution
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/e4d20bee-4efd-435e-bed6-6fea054617c3)

$$y(\textbf{p}_0)=\Sigma_{\textbf{p}_n \in \mathcal{R}} w(\textbf{p}_n) \cdot x(\textbf{p}_0+\textbf{p}_n)$$

- $$p_0$$: 입력 피처맵의 타겟 수용 영역의 center 위치입니다.
- $$\mathcal{R}$$: $$p_0$$의 수용 영역에 존재하는 각 픽셀 위치 방향 벡터입니다.
    - $$\mathcal{R}=\{(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)\}$$.
- $$w(\cdot)$$: 합성곱 연산을 적용하는 데 사용되는 입력 데이터와 커널 간의 가중치 값들의 집합입니다.
- $$x(\cdot)$$: 입력 피처맵에서 주어진 픽셀 위치에 저장된 값입니다.

기존의 일반적인 합성곱(Convolution)은 고정된 커널을 사용하여 입력 피처맵과 합성곱을 수행하며, 이로 인해 모든 픽셀에 동일한 수용 영역이 적용됩니다.

<span style="color:orange"> 하지만, 객체의 크기나 모양이 다양한 경우, 고정된 수용 영역으로는 객체를 정확하게 검출하는 데 어려움이 있을 수 있습니다 </span>.

> **수용 영역(receptive field)**: 출력 레이어의 뉴런 하나에 영향을 미치는 입력 뉴런들의 공간 크기입니다.

**Deformable convolution**은 객체의 형태를 고려하여 feature maps들을 **더 정확하게** 추출하기 위한 기법으로 아래와 같은 순서를 따릅니다.

1. 기존 입력 피처맵($$A$$)을 입력으로 **Convolution 레이어**를 통과한 출력 피처맵($$B$$)을 생성합니다.
2. $$B$$와 Ground-truth를 비교하여 이동 벡터를 찾기 위해 **bilinear interpolation**을 사용합니다.
3. 이동 벡터를 기존 입력 피처맵 $$A$$의 각 픽셀 위치에 더하여 **deformation**을 수행합니다.
4. 이렇게 변형된 픽셀들과 대응되는 커널 위치의 픽셀들과의 합성곱 연산을 수행하여 각 위치마다 합성곱 연산마다 매번 하나의 출력 픽셀을 뽑아냅니다.

> **Bilinear Interpolation**은 입력 좌표값이 정수가 아닌 실수일 때 입력 좌표값에 대한 출력 값을 부드럽게 보정하기 위해 사용되는 보간 기법입니다. 입력값이 실수인 경우 출력 값 또한 실수이며, 해당 실수 위치로 이동한 후에 가장 가까운 정수형 픽셀의 값을 사용합니다. 이를 통해 Deformable Convolution에서는 위치 이동이 가능한 컨볼루션 연산을 보다 정확하고 유연하게 수행할 수 있습니다. 

하기 이미지에서 입력 피처맵으로 부터 두 가지 branch로 나뉘게 됩니다.
- **Branch 1**: offset을 계산하는 conv layer.
- **Branch 2**: offset 정보를 받아서 conv 연산을 수행합니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/a06b96b8-1c3e-4633-b9f4-bfbc05df1b7f)

- **offset field** $$(2N,H,W)$$: 오프셋 필드는 입력 피처맵의 각 위치에 대응되는 값으로, 객체의 수용 영역을 조정하기 위해 사용됩니다.
    - $$(H,W)$$: 각각 입력 피처맵의 높이 및 너비입니다.
    - $$N$$: 커널 크기로, ($$3 \times 3$$) 커널에 대해 $$9$$라는 값을 가집니다.
        - $$2N$$: 각 픽셀에 대해 $$x축/y축$$ 이동 벡터값을 표현하기 위해 채널수는 2배가 됩니다.
- **offset** $$(H,W)$$: 입력 피처맵에서 수용 영역의 각 픽셀 영역에 대한 $$x축/y축$$ 방향의 이동 벡터입니다.

$$y(\textbf{p}_0)=\Sigma_{\textbf{p}_n \in \mathcal{R}} w(\textbf{p}_n) \cdot x(\textbf{p}_0+\textbf{p}_n+\bigtriangleup \textbf{p}_n)$$

- $$\bigtriangleup p_n$$: $$p_n$$ 위치의 픽셀에 대한 **학습 가능한** offset입니다.
    - $$x(\textbf{p})=\Sigma_q G(\textbf{q},\textbf{p}) \cdot x(\textbf{q})$$.
        - **Bilinear interpolation**: $$G(\textbf{q},\textbf{p})=g(q_x,p_x) \cdot g(q_y, p_y)$$.
        - $$g(a,b)=max(0,1-\left\lvert a-b \right\rvert)$$.

입력 피처맵(input feature map)을 굳이 Convolution 레이어를 통과시켜서 얻은 출력 피처맵을 사용하여 offset field를 학습하는 것은 **더 많은 고차원의 추상적인 정보를 활용하여 정확한 위치 조정을 가능케 하는데 있습니다**.

<span style="color:orange"> 만약, 입력 피처맵 자체를 offset field의 입력으로 사용한다면, 학습 과정에서 offset field가 입력 피처맵과 유사한 모습으로 학습되어버릴 수 있으며, 이는 offset field가 올바른 위치 정보를 포착하지 못하게 되는 문제점을 야기합니다</span>.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/4f03adfd-273c-4e5f-97e8-a32eee87fd6c)

- $$(a)$$: 기존 그리드 합성곱 연산.
- $$(b),(c),(d)$$: 기존 수용 영역$$(a)$$의 각 픽셀에 변형을 가한 deformation convolution.

기존의 그리드 합성곱 연산($$a$$)과 다르게 입력 피처맵의 각 픽셀의 위치에 변형(deformation)을 가하여 다양한 형태$$(b)$$의 convolution 연산을 수행하는 모습입니다.

상기 이미지에서 각 위치의 pixels들에 대해 filter를 적용할 때, 해당 위치를 중심으로 작은 offset $$\in \mathbb{R}$$을 부여하여 객체에 따라 적절한 수용 영역을 생성합니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/f1f86c9e-8f88-4a8f-a735-5c8a51f1575f)

하여 **deformable convolution**을 통해 상기 이미지에서 작은 객체에는 더 작은 수용 영역을 적용하고, 큰 객체에는 더 큰 수용 영역을 적용하는 모습입니다.

이를 통해, 객체의 비정형적인 형태를 더 잘 표현하고, 이미지의 미세한 구조를 더욱 정확하게 인식할 수 있게 됩니다.

> Deformable convolution 논문은 Standard CNN의 **마지막 세 개의 레이어에 대해서만** deformable convolution를 적용한다고 합니다.

## Multi-Head Attention in Transformers
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/6f1505c8-e4b5-4b72-adc4-2573a70aec3a)

$$MultiHeadAttn(z_q,x)=\Sigma^M_{m=1} W_m (\Sigma_{k \in \Omega_k} A_{mqk} \cdot W^{\prime}_m x_k)$$

- $$z_q$$: input feature of $$q^{th}$$ query.
- $$x$$: input feature map (input feature of key elements).
    - $$x_k$$: input feature map at $$k^{th}$$ key.
- $$M$$: number of attention heads.
- $$\Omega_k$$: the set of key elements.
- $$A_{mqk}$$: attention weight of $$q^{th}$$ query to $$k^{th}$$ key at $$m^{th}$$ head.
- $$W^{\prime}_m$$: input value projection matrix at $$m^{th}$$ head.
- $$W_m$$: output projection matrix at $$m^{th}$$ head.


****
# Problem Definition ✏
                Given an Transformer-based model for object detection 

                Return a more efficient model

                Such that it outperforms the original model in terms of detecting small objects and inference time while maintaining accuracy.

****
# Challenges and Main Idea💣
**C1)** 기존 DETR 모델을 학습 수렴 속도가 매우 더디다.

<span style="color:lightgreen"> **Idea)** **deformable attention module**은 모든 픽셀이 아닌, 특정 위치만을 선별하여 어텐션을 적용하여 학습 수렴 속도가 $$\times 10$$ 빠릅니다. </span>

**C2)** 기존 DETR 모델은 작은 물체에 대한 object detection 성능이 저조하다.

<span style="color:lightgreen"> **Idea)** **multi-scale deformable attention module**은 다양한 크기의 feature maps를 활용하여 작은 물체를 적절히 탐지합니다. </span>

****
# Proposed Method 🧿
## Deformable Attention
$$MultiHeadAttn(z_q,p_q,x)=\Sigma^M_{m=1} W_m (\Sigma_{k \in \Omega_k} A_{mqk} \cdot W^{\prime}_m x(p_q+\bigtriangleup p_{mqk}))$$

- $$p_q$$: 인풋 피처맵에서 현재 쿼리 픽셀의 위치.
- $$\bigtriangleup p_{mqk}$$: $$m$$번째 헤드에서 $$k$$번째 키에 대한 $$q$$쿼리의 offset.

수식은 별거 없습니다.
단지 기존 $$MultiheadAttn$$에서 인풋 피처맵의 수용 영역을 정하기 위해 offset$$(\bigtriangleup p_{mqk})$$을 부여하고 있습니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/8dfc2a81-896a-41d3-8eff-84bb6f3cc1db)

- Reference point: 현재 합성곱 연산이 진행되고 있는 타겟 픽셀.
- Sampling points: 합성곱 연산에서 사용할 변형된 수용 영역 픽셀들

기존 DETR의 self-attention처럼 모든 픽셀들에 대해 attention을 수행하는 것이 아니라, 특정 sampling points들에 대해서만 attention을 수행합니다.

여기서 deformable convolution와 달리, Deformable DETR은 **attention의 입력**에 대해서 독립적인 linear layer를 통과시켜 sampling offset을 도출하고 sampilng points를 구합니다.

또한, 기존 DETR처럼 일반적인 방식처럼 내적을 통해 구한 attention weights가 아닌, 독립적인 linear layer을 취한 뒤, 그 결과값에 softmax를 취하여 attention weights를 계산합니다.

## Multi-scale Features
기존 DETR은 단 하나의 scale의 피처맵을 사용해서 학습하기 때문에, 미세한 탐지가 요구되는 작은 객체들에 대한 다양한 특징들을 관찰하지 못해서 성능이 좋지 않습니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/b8631fa8-54f6-4743-9a77-df98de515a36)

하여 상기 이미지에서 Deformable DETR에서는 CNN으로 서로 다른 크기의 특징맵들을 얻은 뒤, 그들로 부터 각각의 sampling points를 참조합니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/205ba71a-11e6-464e-82a3-889b53fdbe56)

이 때 피처맵들의 크기가 다르기 때문에, 알맞게 각 피처맵에 대한 sampling point 픽셀 위치를 찾기 위해서 offset을 $$[0,1]$$ 범위로 scaling하는 normalization를 거치게 됩니다.

하여 상기 이미지처럼 각 크기의 피처맵에 대해 적절히 sampling points를 찾는 모습입니다.

## Multi-scale Deformable Attention
$$MSDeformAttn(z_q,\hat{p}_q,(x^l)^L_{l=1})=\Sigma^M_{m=1} W_m (\Sigma^L_{l=1} \Sigma_{k \in \Omega_k} A_{mlqk} \cdot W^{\prime}_m x(\phi_l (\hat{p}_q)+\bigtriangleup p_{mlqk}))$$

- $$\phi_l(\cdot)$$: $$l$$번째 피처맵에서 scaling 값.
- $$\bigtriangleup p_{mlqk}$$: $$m$$번째 헤드에서 $$l$$번째 피처맵의 $$k$$번째 키에 대한 $$q$$쿼리의 offset.

이 수식 또한 별거 없습니다.
다양한 크기의 피처맵을 활용하기 때문에, $$\Sigma^L_{l=1}$$로 각 피처맵에 대한 어텐션을 수행하는 모습입니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/5eb07c0e-f8ec-464b-8e3b-27ac18dcb325)

- 사용하는 특징맵의 개수$$(k)$$는 하이퍼 파라미터입니다.
    - 상기 이미지에서는 $$k=4$$입니다.
- 이전 레이어 층의 서로 다른 크기의 피처맵들로 부터의 sampilng points를 다음 층의 출력 픽셀 하나를 결정하기 위한 수용 영역으로 삼습니다.

## 성능 향상
저자는 성능 향상을 위해 두 가지 방법 **(1)Iterative Bounding Box Refinement**, **(2)Two-Stage Deformable DETR**을 추가로 사용합니다.
- **Iterative Bounding Box Refinement**: 박스 좌표를 조금 보정해주는 auxiliary 레이어를 각 디코더 레이어에 추가하면 <span style="color:lightgreen"> AP 점수가 기존 DETR에 비해 $$2$$ 정도 오릅니다 </span>. 
- **Two-Stage Deformable DETR**: object query의 위치를 랜덤하게 초기화 하지 않고, Encoder를 따로 학습시켜서 얻은 출력을 바탕으로 객체가 존재하는 위치 부근에 초기화 시킵니다. 이를 통해 <span style="color:lightgreen"> AP 점수가 기존 DETR에 비해 $$3$$ 정도 오릅니다 </span>.  
    - 인코더를 1차적으로 따로 학습시키기 때문에 Two-Stage 입니다.

****
# Experiment 👀
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/a8e19de1-37e4-495e-89de-e5922247a14e)

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/6008614c-cced-47b2-93fa-0a86987e1321)

이 글의 서두에 언급했듯이, 기존 DETR에 비해 학습 속도가 최대 $$\times 10$$ 가량 빠르게 나타나는 모습입니다.

****
# Open Reivew 💗
NA

****
# Discussion 🍟
- 본 논문에서는 object query의 개수를 300개로 설정한다. 이를 전부 활용하는 것은 "no object" 레이블이 많아져서 불필요한 학습을 많이하게 되어 효율이 별로일 것 같은데, 그 개수를 줄이는 알고리즘을 연구할만한 가치가 있을까?
- 디코더의 multi-head self-attention은 제거하여도 본 논문의 모델의 성능에 지장이 없는데, 아키텍처 자체를 더 효율화 할 수 있는 여지가 있을까?

****
# Major Takeaways 😃
- Deformable Attention
- Transformer

****
# Conclusion ✨
## Strength
## Weakness

****
# Reference