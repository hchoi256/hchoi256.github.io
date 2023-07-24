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
<!-- ![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/5eb07c0e-f8ec-464b-8e3b-27ac18dcb325) -->

- DETR 모델에서 Transformer 아키텍처의 한계 해결
    - slow convergence $$\rightarrow$$ **deformable attention**(key sampling points)로 해결.
    - 작은 물체 객체 탐지 성능 저조 $$\rightarrow$$ **multi-scale features**로 해결.
- $$\times 10$$ 배 빠른 학습 속도.

****
# Preliminaries 🙌
## DETR 모델 한계
- **느린 수렴 속도**
    - DETR은 초기화 단계에서 feature maps의 모든 pixels에 대해 uniform attention weights를 부여합니다. 이는 모든 픽셀이 동일한 중요도를 가진다고 가정하는 것인데, 실제로는 객체가 존재하는 영역과 배경이나 노이즈가 있는 영역 등에 대해 다른 중요도를 가지는 것이 보통입니다. <span style="color:orange"> 하여 객체의 위치를 다소 부정확하게 판단하게 되는 **sparse meaningful locations** 현상이 발생합니다 </span>.
    - Transformer 인코더에서 각 픽셀들은 서로 다른 픽셀들과의 attention weight를 계산한다.
        - Quadratic time complexity ($$N^2$$).
            - $$N$$: the number of pixels.
- **작은 물체에 대한 객체 탐지 성능 저조**
    - 작은 물체 탐지를 위해 **high-resolution feature maps**가 중요합니다.
        - <span style="color:orange"> DETR은 Transformer 기반의 모델로 CNN 기반 모델보다 더 복잡한 구조를 가지고 있기 떄문에, 높은 해상도의 특징 맵을 처리하는데 더 많은 메모리와 계산량이 필요하게 됩니다. </span>
    - 작은 물체 탐지를 위해 **multi-scale features** 활용이 중요합니다.
        - 작은 객체는 이미지에서 크기가 작고 미세한 구조를 가지는 경우가 많습니다. Multiscale 피처맵은 다양한 해상도를 가지며, 작은 객체를 더 잘 포착하기 위해 다양한 크기의 특징을 제공합니다.
        - <span style="color:orange"> DETR은 multi-scale features가 아닌 동일한 크기의 patch로 features를 생성합니다. </span>


## Deformable Convolution
- 객체의 형태를 고려하여 feature maps들을 더 정확하게 추출하기 위한 기법입니다.
    - **sparse spatial locations**을 통해 sparse meaningful locations 현상 해결 가능
        - sparse spatial locations: 특정 이미지나 피처맵에서 일부 픽셀 또는 위치만을 선택하는 것을 의미합니다.
    - **element relation modeling**이 약함
        - 입력 이미지의 pixels들 간의 상대적인 관계 모델링
            - 객체의 위치, 크기, 클래스 등을 파악하는 데에 용이합니다.
- 각 위치의 pixels들에 대해 filter를 적용할 때, 해당 위치를 중심으로 작은 변형(deformation)을 가하는 방식으로 동작합니다.
    - 객체의 비정형적인 형태를 더 잘 표현하고, 이미지의 미세한 구조를 더욱 정확하게 인식할 수 있게 됩니다.

> 기존의 일반적인 컨볼루션은 사각형 형태의 필터를 사용하여 이미지의 공간적인 특징을 인식하는데, 이는 일부 객체의 형태가 사각형이 아닌 **비정형적인 형태**일 때 문제가 될 수 있습니다.

****
# Problem Definition ✏
                Given an Transformer-based model for object detection 

                Return a compressed model

                Such that it outperforms the performance of the original model in terms of inference time while retaining accuracy.

****
# Challenges and Main Idea💣
**C1)** 기존 DETR 모델을 학습 수렴 속도가 매우 더디다.

<span style="color:blue"> **Idea)** **deformable attention module**은 모든 픽셀이 아닌, 특정 위치만을 선별하여 어텐션을 우선 적용합니다. </span>

**C2)** 기존 DETR 모델은 작은 물체에 대한 object detection 성능이 저조하다.

<span style="color:blue"> **Idea)** **multi-scale deformable attention module**은 다양한 크기의 feature maps를 활용하여 작은 물체를 적절히 탐지합니다. </span>

****
# Proposed Method 🧿
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/5eb07c0e-f8ec-464b-8e3b-27ac18dcb325)

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