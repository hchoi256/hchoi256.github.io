---
layout: single
title: "[논문분석] Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation (CVPR 2018)"
categories: AIPaperCV
tag: [Computer Vision, CAM, AffinityNet, Weakly Supervised Semantice Segmentation]
toc: true
toc_sticky: true
toc_label: "쭌스log"
author_profile: false
header:
    teaser: /assets/images/posts/ws.png
sidebar:
    nav: "docs"
---

<span style="color:sky"> [논문링크](https://arxiv.org/abs/1803.10464) </span>.

****
# 한줄요약 ✔
- 기존 WSSS는 CAM으로 GT를 생성하는데, CAM이 생성한 feature map은 local discriminative parts만을 강조하여 전체 이미지에 대해 robust한 map을 생성하지 못한다.
- AffinityNet은 CAM으로 얻은 이러한 local activation을 인접 픽셀들로 propagate하여 보다 potent한 semantic entity를 추출한다.
- AffinityNet은 오로지 image-level supervision을 따르며, 추가적인 annotations은 필요없다.

****
# Preliminaries 🍱
## CRF
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/0e53c74a-ff8d-4a4d-a0e3-98ee798752b9)

- CNN 기반 모델보다 더 detail하게 segmentation 특징 추출이 가능하다.
- CNN으로 얻은 feature map에 표현된 객체별 segmentation의 가장자리를 보다 다듬기 위해 사용.
    - 상기 이미지에서 CRF 출력은 details도 잘 잡아내는 모습이며, CNN 결과와 CRF 결과 간의 `KL-Divergence`을 Loss func.에 포함하여 semantic entity의 가장자리를 post-processing한다.

> 보다 자세한 공식 설명은 본 글에서는 각설한다.

## CAM
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/7f1ac706-7a16-42be-b7ec-8b22ff21a26b)

- CNN의 최종 layer 출력인 feature map에 channel-wise 계산법인 GAP를 적용하여 feature map의 spatial info.를 유지 (기존 CAM에서는 1차원 배열로 point-wise하게 flatten시킨 후 fc-layer의 인풋으로 활용한다).
    - 각 채널은 이미지에서 각 객체의 특징을 표현한다 (채널 개수=커널 개수).
    
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/49132aea-08ab-4392-9494-dab4c2ed9386)

- 이후, FC-layer에서 각 특징을 담고있는 feature map의 평균값을 인풋으로 받고, 각 인풋의 각 class에 대한 민감도를 softmax를 거쳐 weights로 표현한다. 

### Equation
$$Y^c=\Sigma_k w^c_k {1 \over Z} \Sigma_{i,j} A^k_{i,j}$$

- $$Y^c$$: class $$c$$에 대한 score (모델 예측 logit/output).
- $${1 \over Z} \Sigma_{i,j} A^k_{i,j}$$:  feature map $$k$$의 GAP 값.
    - $$Z$$: feature map $$k$$에서 pixel 개수.
- $$w^c_k$$: feature map $$k$$의 class $$$$c$$$$에 대한 민감도.
- $$A^k_{i,j}$$: feature map $$k$$의 $$$$i,j$$$$에 해당하는 pixel 값.

****
# Challenges and Main Idea💣
**C1)** <span style="color:orange"> CAM이 생성한 feature map은 local discriminative parts만을 강조한다 </span>.

**Idea)** <span style="color:lightgreen"> CAM의 local activation을 인접 픽셀들로 propagate하여 보다 potent한 semantic entity를 추출 </span>.

****
# Problem Definition ❤️
Given a model $$\mathcal{T}$$.

Return a model $$\mathcal{S}$$.

Such that $$\mathcal{S}$$ outweighs the segmentation performance of $$\mathcal{T}$$.

****
# Proposed Method 🧿
## Model Architecture
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/4ab0ff0c-0fd8-413e-b98b-02aa32509a4c)

1. **CAM**
- object area 강조하는 feature map $$\mathcal{F}$$ 추출.

2. **AffinityNet**
- entire image area에 대하여 segmentation을 반영하기 위해, $$\mathcal{F}$$ 에서 표현된 class별 픽셀 semantic 정보를 nearby area로 적절히 propagate하여 CAM의 quality를 높인다. 

3. **DNN (Seg Net)**
- 앞서 추출한 segmentation label을 기반으로 segmentation을 수행하는 network.

## AffinityNet
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/dd086a36-10bd-4b39-b555-de7ee7850c39)

<span style="color:yellow"> $$W_{ij}=exp^{- \Vert f^{aff}(x_i,y_i)-f^{aff}(x_j,y_j) \Vert_1}$$ </span>.

- $$f^{aff}$$: feature map에서 $$x_i, y_i$$ 위치에 놓인 픽셀의 affinity score.
    - 상기 이미지의 학습 가능한 network를 통해 계산된다.
        - 해당 network를 학습시키기 위해 semantic affinity label이 필요하다.
- $$W_{ij}$$: 서로 다른 픽셀들 간의 affinity score 차이 (= 연관성 정도).

### Semantic affinity label
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/fa4b29f7-35a9-4878-9de9-26475e2d1dca)

1. CAM으로 background를 표현하는 feature maps $$M_c$$와 $$M_{bg}(x,y)$$ 추출.
    1. <span style="color:yellow"> $$M_{bg}(x,y)=\{1-max_{c \in C} M_c(x,y)\}^{\alpha}$$; where $$M_c(x,y)=\bold{w}^T_c f^{cam}(x,y)$$ </span>.
        1. $$x,y$$: 픽셀 위치.
        2. $$C$$: class.
        3. $$f^{cam}$$: a feature vector before GAP.
        4. $$\bold{w}_c$$: classification weights.
        5. $$\alpha$$: background score 활성화 정도 조절.
        6. $$M_c$$: GAP 이후 얻은 CAM 출력.
2. Object confident area 획득 과정:
    1. $$\alpha$$를 감소시키면 $$M_{bg}$$가 증가되어 background score가 강조되어 feature map $$(c)$$를 얻는다.
    2. $$M_c$$ 에 dense CRF (dCRF)를 적용하여 나온 feature map $$(b)$$ 에서 객체별로 가장 높은 class 점수를 달성한 class label이 부여되어 semantic entities가 표현된다.
3. Confident background area 획득 과정:
    1. 앞서 구한 $$(b)$$ 와 $$(c)$$ 를 비교하여 확실히 background인 영역을 선별한다.
4. 그 외 나머지는 중립 영역으로 labeling.
5. Semantic affinity label 획득.

### Loss Function
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/b630a084-b2cb-4429-b10f-9c090bb8d900)

<span style="color:yellow"> $$\mathcal{L}=\mathcal{L}^+_{fg}+\mathcal{L}^+_{bg}+2\mathcal{L}^-$$ </span>

- $$\mathcal{L}^+_{fg}=-\frac{1}{\mathcal{P}^+_{fg}} \Sigma_{(i,j) \in \mathcal{P}^+_{fg}} log W_{ij}$$.
- $$\mathcal{L}^+_{bg}=-\frac{1}{\mathcal{P}^+_{bg}} \Sigma_{(i,j) \in \mathcal{P}^+_{bg}} log W_{ij}$$.
- $$\mathcal{L}^-=-\frac{1}{\mathcal{P}^-} \Sigma_{(i,j) \in \mathcal{P}^-} log (1-W_{ij})$$.

<span style="color:yellow"> $$\mathcal{P}=\{ (i,j) \vert d((x_i,y_i),(x_j,y_j)) < \gamma, \forall i \neq j \}$$ </span>

- $$\mathcal{P}^+=\{ (i,j) \vert (i,j) \in \mathcal{P}, W^*_{ij}=1$$.
    - $$W^*_{ij}=1$$: $$(i,i)$$ 과 $$(j,j)$$ 위치의 픽셀이 같은 label일 경우.
- $$\mathcal{P}^-=\{ (i,j) \vert (i,j) \in \mathcal{P}, W^*_{ij}=0$$.

## CAM with AffinityNet
<span style="color:yellow"> $$vec(M^*_c)=T^t \cdot vec(M_c)$$ </span>

- $$t$$: number of iterations.
- $$T=D^{-1} W^{\circ \beta}$$: Transition matrix.
    - $$D_{ii}=\Sigma_j W^{\beta}_{ij}$$.
    - $$W^{\circ \beta}$$: [hadamard power](https://ko.wikipedia.org/wiki/%EC%95%84%EB%8B%A4%EB%A7%88%EB%A5%B4_%EA%B3%B1).
    - $$\beta >1$$: 중요하지 않은 affinity 값을 무시하도록 해준다.

****
# Experiment 👀
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/0beac313-b575-47a6-9a71-bff751ee4102)

- $$(c)$$에서 소파 객체의 가장자리는 non-discriminative 하기 때문에 CAM이 segmentation을 제대로 수행하지 못한 모습이다. 하나 $$(d)$$에서 AffinityNet을 통해 소파의 가장자리까지 적절히 propagate하여 semantic entities를 생성한 모습이다.

그 외 다른 실험 결과들은 PASCAL Voc 2012에서 SOTA를 달성했다는 자료들이니 자세한 설명은 각설한다.

****
# Open Reivew 💗
NA

****
# Discussion 🍟
NA

****
# Major Takeaways 😃
NA

****
# Conclusion ✨
NA

****
# Reference
https://velog.io/@kowoonho/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Learning-Pixel-level-Semantic-Affinity-with-Image-level-Supervision-for-Weakly-Supervised-Semantic-Segmentation