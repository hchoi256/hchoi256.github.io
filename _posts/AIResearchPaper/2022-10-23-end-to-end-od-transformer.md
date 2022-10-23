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

What paper you have read?
- End-to-End Object Detection with Transformers (ECCV 2020)

What area or technology addressed in the paper that interested you the most? And can they be applied to Carla?
- DETR Accuracy > SOTA R-CNN
- panoptic segmentation
- DETR is straightforward to implement
- high performance on large dataset


****
# 배경 🙌
**Object detection**은 분류를 위한 cateogry box를 생성하는 과정이다.

[*non-maximal suppression*]

![image](https://user-images.githubusercontent.com/39285147/197414699-970639a6-076d-4b2b-b1de-763931c9082e.png)

현대 detectors들은 *간접적인 방법*(hand-designed = 사전작업)으로 객체 탐지를 구현했다: anchors, non-maximal suppression, window centers, 대용량 proposals, etc.

> **Non-Maximum Suppression**
>> object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법

이러한 간접적인 방법들은 *후처리*(오차 예측 및 제거)에 막대하게 영향을 받는다.

> 오차: 실제값 - 예측값

기존 모델들은 여러 가지 복잡한 예측 문제에 대한 객체 탐지 문제에서 한계를 나타낸다.

이러한 pipeline(과정)에서의 surrogate tasks를 간소화하기 위해 등장한 것이 **Direct set prediction 방법**이다.

> Surrogate 모델: output이 정확히 측정될 수 없을 때, 대신 output 기준을 제공하는 대리 모델

이러한 새로운 모델들은 여러 가지 복잡한 예측 문제 해결에 효과적이다.

그러한 방법의 대표적인 예시인 **DETR**에 대해 이번 논문에서 알아보자.

****
# INTRODUCTION 👀
 ![image](https://user-images.githubusercontent.com/39285147/197411640-e6c3de0f-b4f3-4665-ae05-6a0b45c90bf3.png)

DETR
- **하나의 CNN를 Transformer 아키텍쳐와 병합**하여 병렬처리를 통해 **직접적으로 예측** 수행
- extra-long training schedule
- auxiliary decoding losses in the transformer
- the conjunction of the bipartite matching loss and transformers with (non-autoregressive) parallel decoding

> **자기회귀(AR; Autoregressive)**
>> 과거의 움직임에 기반해서 미래를 예측하는 것 

## 기술 용어
### Bipartite matching
- ground truth boxes를 사용해 독립적 예측을 한다.
    - no match: "no object"
- uniquely assigns a prediction to a ground truth object
    - 예측값 순열 = invariant = 예측할 객체에 대해 고유함 --> parallellism 보장
    - 반대로 기존 RNN 모델 = autoregressive decoding
- Hungarian algorithm

### Self-attention
- pairwise interactions between elements in a sequence,
- duplicate predictions 제거

### Set loss function
- performs bipartite matching between predicted and ground-truth objects.

### COCO
- one of the most popular object detection datasets

### Transformer Decoder
**기존 Transformer** = autoregressive (output sequence를 하나하나 넣어주는 방식) 

**DETR 모델** = 한번에 N개의 obejct를 병렬 예측
- N개의 다른 결과 --> N개의 서로 다른 input embedding
    - Input embedding: "*object query(positional encoding)*"를 통해 표현
        - 1) N개의 object query는 디코더에 의해 output embedding으로 변환
        - 2) N개의 마지막 예측 산출
        - 3) self/encoder-decoder간 어텐션을 통해 각 object 간의 global 관계 학습

### Auxiliary decoding losses
모델이 맞는 개수의 object를 예측할 수 있도록 도움을 주는 auxiliary loss를 사용하였다. 우리는 각 decoder layer이후에 prediction FFN과 Hungarian loss를 추가하였다. 모든 prediction FFN은 같은 파라미터를 사용하였다. 우리는 다른 디코더레이어에서의 input을 정규화하기 위해 layer norm을 추가하였다.

****
# Related Work 🗂
## Set Prediction
**Indirect Set Prediction**
- multilabel classification (postprocessing)
    - near-identical boxes 해결 불가능

**Direct Set Prediction**
- postprocessing-free
    - *global inference schemes*
        - model interactions between all predicted elements to avoid redundancy.
    - Auto-regressive sequence models (i.e., recurrent neural networks)
        - bipartite matching loss            
        - permutation-invariance

**DETR**
- bipartite matching loss + transformers (non-autoregressive) parallel decoding

## Transformers and Parallel Decoding

## Object Detection


****
# DETR Model ✒
## Object Detection Set Prediction Loss

## DETR Architecture

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
- 대용량 dataset에 대한 성능 ↑
    - global information performed by the self-attention
    - enabled by the non-local computations of the transformer
- 커스텀 layers 필요 없음 --> better reproduction
    - 때문에 DETR은 ResNet, Transformer 프레임 워크에서도 재사용 가능 

## DETR 한계
- Training 너무 오래 걸림
- 최적화 문제
- small dataset에 대한 학습 성능 ↓

****
# Reference 👓
[**GitHub Repository**](https://github.com/hchoi256/carla-research-project)

[*UW Madison CARLA Research Team*](https://cavh.cee.wisc.edu/carla-simulation-project/)

[*CARLA Simulator*](https://carla.readthedocs.io/en/latest/)