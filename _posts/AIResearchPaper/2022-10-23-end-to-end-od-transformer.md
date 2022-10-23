---
layout: single
title: "[논문 분석] End-to-End Object Detection with Transformers (ECCV 2020)"
categories: AIPaperCV
tag: [Object Detection, Transformer]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/od.png
sidebar:
    nav: "docs"
---

[**논문**](https://link.springer.com/content/pdf/10.1007/978-3-030-58452-8_13.pdf)

****
# 배경 🙌
**Object detection**은 분류를 위한 cateogry box를 생성하는 과정이다.

현대 detectors들은 *간접적인 방법*(hand-designed = 사전작업)으로 객체 탐지를 구현했다: anchors, non-maximal suppression, window centers, 대용량 proposals, etc.

> **Non-Maximum Suppression**
>> ![image](https://user-images.githubusercontent.com/39285147/197414699-970639a6-076d-4b2b-b1de-763931c9082e.png)
>> object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법

이러한 간접적인 방법들은 *후처리*(오차 예측 및 제거)에 막대하게 영향을 받는다.

> 오차: 실제값 - 예측값

때문에 기존 모델들은 여러 가지 복잡한 예측 문제에 대한 객체 탐지 문제에서 한계를 나타낸다.

이러한 간접적인 pipeline(과정)에서의 surrogate tasks를 간소화하기 위해 등장한 것이 **direct set prediction 방법**이다.

> **Surrogate**: output이 정확히 측정될 수 없을 때, 대신 output 기준을 제공

이번 논문 주제인 **DETR(DEtection TRansformer)**은 direct set prediction(bipartite matching loss) 방법과 Transformer(non-autoregressive)을 결합한 방법이다.

상기 언급된 용어들에 대해 알아보자.

****
# INTRODUCTION 👀
 ![image](https://user-images.githubusercontent.com/39285147/197411640-e6c3de0f-b4f3-4665-ae05-6a0b45c90bf3.png)

## 배경지식
### Bipartite matching(이분매칭)
[*Bipartite Graph*]

![image](https://user-images.githubusercontent.com/39285147/197421855-3281f5d4-8b83-4983-b407-6f84649dbccc.png)

이분 그래프에서 A 그룹의 정점에서 B 그룹의 정점으로 간선을 연결 할 때, **A 그래프 하나의 정점이 B 그래프 하나의 정점만** 가지도록 구성된 것이 이분 매칭

상기 이분 그래프에서, 좌측은 이분 그래프이지만 이분 매칭은 아니고 우측은 둘다 맞다.

- ground truth boxes를 사용해 독립적 예측을 한다.
    - no match: "no object"
- uniquely assigns a prediction to a ground truth object
    - 객체에 대한 예측값 순열 불변(invariant) --> 객체별 parallellism 보장
        - 개별적으로 GT(ground truth) object와 예측 object에 대한 loss를 가지고 있어서 예측된 object의 순서에 상관이 없어 병렬화가 가능
    - 반대로 기존 RNN 모델 = autoregressive decoding(object 순서 O) --> 객체별 parallellism 보장 X
- set loss function(하기 참조)

> **Gound-truth**
>> ![image](https://user-images.githubusercontent.com/39285147/197421710-7405f615-8cfb-40b7-bd86-3ee8f7346a96.png)
>> 데이터의 원본 혹은 실제 값 표현

### Set loss function
- performs bipartite matching between predicted and ground-truth objects.
    - 다르게 말하면, 하나의 object에 대하여 각각 독립적으로 GT 및 예측값 이분 매칭 수행
- [Hungarian algorithm](https://gazelle-and-cs.tistory.com/29)

> **Hungarian algorithm**
>> 가중치가 있는 이분 그래프(weighted bitarted graph)에서 maximum weight matching을 찾기 위한 알고리즘

### COCO
- one of the most popular object detection datasets

### Transformer Decoder
**기존 Transformer** = autoregressive (output sequence를 ***하나하나*** 넣어주는 방식) 
- pairwise interactions between elements in a sequence
- duplicate predictions 제거 가능

****
# DETR Model ✒
- bipartite matching loss + transformers with (non-autoregressive) parallel decoding
- **하나의 CNN를 Transformer 아키텍쳐와 병합** --> **직접 예측** 가능
- extra-long training schedule
- auxiliary decoding losses in the transformer

> **자기회귀(AR; Autoregressive)**
>> 과거의 움직임에 기반 미래 예측

## Set Prediction
**Indirect Set Prediction**
- multilabel classification (postprocessing)
    - near-identical boxes 해결 어렵

**Direct Set Prediction**
- postprocessing-free
    - *global inference schemes* model interactions between all predicted elements to avoid redundancy.
- Auto-regressive sequence models (i.e., recurrent neural networks)
    - bipartite matching loss            
    - permutation-invariance

## Object Detection Set Prediction Loss
- bipartite matching loss + transformers (non-autoregressive) parallel decoding
- loss = 최적 bipartite matching(예측값 ~ GT)
    - 최적 bipartite matching = 예측값 ~ 실제값 매칭 방법 중 최저 비용을 갖는 매칭
    - Hungarian algorithm을 통해 효율적으로 찾을 수 있다

> **Hungarian algorithm**
>> ![image](https://user-images.githubusercontent.com/39285147/197422840-8c8770b5-895b-4c82-b967-da083a62c4df.png)
>> ![image](https://user-images.githubusercontent.com/39285147/197422872-acf77efd-3103-4008-921c-f62aa22a13fc.png)

> **Bounding box loss**
>> ![image](https://user-images.githubusercontent.com/39285147/197422932-1866e001-8086-4f89-a231-4582d8e304d2.png)
>> ![image](https://user-images.githubusercontent.com/39285147/197422984-634e754a-c7db-47fd-9eaa-2523296a2057.png)

## DETR Architecture
![image](https://user-images.githubusercontent.com/39285147/197422990-0d50e9ab-0866-40d2-9940-ff3ffb91fdde.png)

## Backbone
- feature extraction

## Transformer Encoder
- feature maps 생성 과정
- 기존 transformer encoder에 **positional encoding** 추가
    - 덕분에 autoregressive와 다르게 인풋 순서 상관 안 써도됨

## Transformer Decoder
- ***한번에*** N개의 obejct를 병렬 예측
    - 1) Input embedding
        - *object query(positional encoding)* 통해 표현
    - 2) N개의 object query는 디코더에 의해 output embedding으로 변환
    - 3) N개의 마지막 예측값들 산출
    - 4) self/encoder-decoder간 어텐션을 통해 각 object 간의 global 관계 학습

## Prediction FFN
- 최종 디텍션 예측
- 3개의 perceptron, ReLU, linea projection으로 구성
- Procedure
    - 1) FFN --> 상대적인 중앙값 예측
    - 2) linear layer --> softmax로 class 예측
        - 실제 object 외 class = ∅

## Auxiliary decoding losses
- 알맞은 개수의 object 예측 가능
- 각 decoder layer에 대해 prediction FFN & Hungarian loss 추가
    - 모든 prediction FFN은 같은 파라미터를 사용
- 다른 decoder layer의 input 정규화를 위해 layer norm 추가

> **FFN**
>> simple feed forward network(FFN)

****
# Experiments ✏

## 성능 비교: Faster R-CNN and RetinaNet

## Ablations

## DETR for Panoptic Segmentation

****
# 결과 ✔
## DETR 장점
- DETR 정확도 SOTA R-CNN 모델 능가
- 유연한 Transformer 아키텍쳐 --> Panoptic segmentation 성능 ↑
- 구현이 쉽다
- 큰 object 탐지 성능 ↑
    - global information performed by the self-attention
    - enabled by the non-local computations of the transformer
- 커스텀 layers 필요 없음 --> better reproduction
    - 때문에 DETR은 ResNet, Transformer 프레임 워크에서도 재사용 가능 

## DETR 한계
- Training 너무 오래 걸림
- 최적화 문제
- 작은 object 탐지 성능 ↓

****
# Reference 👓
[**GitHub Repository**](https://github.com/hchoi256/carla-research-project)

[*UW Madison CARLA Research Team*](https://cavh.cee.wisc.edu/carla-simulation-project/)

[*CARLA Simulator*](https://carla.readthedocs.io/en/latest/)

[End to End Object Detection with Transformers](https://velog.io/@long8v/End-to-End-Object-Detection-with-Transformers)

[*Hungarian Algorithm*](https://gazelle-and-cs.tistory.com/29)