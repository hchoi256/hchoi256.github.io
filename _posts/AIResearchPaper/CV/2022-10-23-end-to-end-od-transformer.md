---
layout: single
title: "[논문 분석] End-to-End Object Detection with Transformers (ECCV 2020)"
categories: AIPaperCV
tag: [Computer Vision, Object Detection, Transformer]
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
**Object detection**은 Classification과 Localization을 합친 것으로, 사진속 객체에 대해 클래스 분류와 bounding box 위치를 표현하는 것을 말한다.

이전 객체탐지 방법들은 사진속 객체가 존재할 것 같은 후보영역을 차출하기 위해 *간접적인 방법*으로 객체 탐지를 구현했다.
- Two-stage (속도 느림, 정확성 좋음):
    - **RCNN, Fast RCNN**: Selective Search
    - **Faster RCNN**: Sliding window, Anchor boxes, RPN
- One-stage (속도 빠름, 정확성 나쁨):
    - **YOLO**: grid division

> Non-maximal suppression: 객체별 가장 높은 예측 확률을 보인 bounding box를 기준으로 중복되는 다른 boxes를 제거하는 기법.  

> **Non-Maximum Suppression**
>> ![image](https://user-images.githubusercontent.com/39285147/197414699-970639a6-076d-4b2b-b1de-763931c9082e.png)
>>
>> object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법
>>
>> 객체별 가장 높은 예측 확률을 보인 bounding box를 기준으로 중복되는 다른 boxes를 제거하는 기법. 

이러한 간접적인 방법들은 *후처리*(오차 예측 및 제거)에 막대하게 영향을 받는다.

때문에 기존 모델들은 여러 가지 복잡한 예측 문제에 대한 객체 탐지 문제에서 한계를 나타낸다.

이러한 간접적인 pipeline(과정)에서 정확한 결과를 낼 수 없어 NMS같은 추가적인 후처리의 힘을 빌리는 surrogate tasks를 간소화하기 위해 등장한 것이 **direct set prediction 방법**이다.

> **Surrogate**: output이 정확히 측정될 수 없을 때, 대신 output 기준을 제공한다.

우리가 흔히 아는 집합(set)은 하나의 집합 안에 중복되는 요소가 없고 순서의 제약이 없는 것이 특징이다.

Direct set prediction은 이러한 집합의 특성을 이용하여, 하나의 객체에 대해 단 하나의 bounding box만 매칭되는 것을 도와 중복되는 box의 생성을 회피하여 NMS 같은 후처리의 의존성에서 벗어난다.
- 순서 제약 없음: 각 객체별 병렬성 보장
- 중복 회피: 객체별 독립적인 하나의 bounding box 가짐.

이번 논문 주제인 **DETR(DEtection TRansformer)**은 direct set prediction(bipartite matching loss) 방법과 Transformer(non-autoregressive)을 결합한 방법이다.

****
# INTRODUCTION 👀
 ![image](https://user-images.githubusercontent.com/39285147/197411640-e6c3de0f-b4f3-4665-ae05-6a0b45c90bf3.png)

## 배경지식
### Bipartite matching(이분매칭)
[*Bipartite Graph*]

![image](https://user-images.githubusercontent.com/39285147/197421855-3281f5d4-8b83-4983-b407-6f84649dbccc.png)

이분 그래프에서 A 그룹의 정점에서 B 그룹의 정점으로 간선을 연결 할 때, **A 그래프 하나의 정점이 B 그래프 하나의 정점만** 가지도록 구성된 것이 이분 매칭

상기 이분 그래프에서, 좌측은 이분 그래프이지만 이분 매칭은 아니고 우측은 둘다 맞다.

- 중복 회피
    - Transformer 학습 시, ground truth와 디코더의 인풋인 object query가 각각의 객체와 대응되는 독립적 예측을 가능케 한다.
    - object query: 디코더 인풋으로 인코더 출력으로부터 정보를 받아 사진속 각 객체에 대한 클래스와 box 위치 정보를 학습한다.
        - 디코더 인풋인 object qeury의 개수($N$)는 사전에 지정되는 하이퍼 파라미터로써 사진속에 존재할 것으로 생각되는 총 객체 개수보다 크게 잡는다(논문에서는 100개의 object query를 사용).
- 순서 제약 없음
    - uniquely assigns a prediction to a ground truth object
    - 객체에 대한 예측값 순열 불변(invariant) --> 객체별 parallellism 보장
        - 개별적으로 GT(ground truth) object와 예측 object에 대한 loss를 가지고 있어서 예측된 object의 순서에 상관이 없어 병렬화가 가능

> **Gound-truth**
>> ![image](https://user-images.githubusercontent.com/39285147/197421710-7405f615-8cfb-40b7-bd86-3ee8f7346a96.png)
>>
>> 데이터의 원본 혹은 실제 값 표현

> 참고로 기존 RNN 모델은 autoregressive decoding 기반이라 object 순서가 존재해서 객체별 parallellism이 보장되지 않는다.

### Set loss function
- 이분매칭(bipartite matching)을 예측값과 GT값에 대해 수행한다.
    - 예측값과 GT는 모두 (class, box 위치) 형식을 갖는다.
        - box 위치는 (x,y,w,h) 형식으로 제공된다; x와 y는 box의 중앙이고, w와 h는 해당 box의 각각 너비와 높이이다.
    - DETR 모델의 목적함수(Hungarian algorithm)로 사용된다.

> [Hungarian algorithm](https://gazelle-and-cs.tistory.com/29)
>
>> 가중치가 있는 이분 그래프(weighted bitarted graph)에서 maximum weight matching을 찾기 위한 알고리즘

> **COCO**: one of the most popular object detection datasets

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
![image](https://user-images.githubusercontent.com/39285147/197422840-8c8770b5-895b-4c82-b967-da083a62c4df.png)- bipartite matching loss + transformers (non-autoregressive) parallel decoding

- loss = 최적 bipartite matching(예측값 ~ GT)
    - 최적 bipartite matching = 예측값 ~ 실제값 매칭 방법 중 최저 비용을 갖는 매칭
    - Hungarian algorithm을 통해 효율적으로 찾을 수 있다

> **Hungarian algorithm**
>> ![image](https://user-images.githubusercontent.com/39285147/197422872-acf77efd-3103-4008-921c-f62aa22a13fc.png)

> **Bounding box loss**![image](https://user-images.githubusercontent.com/39285147/197450096-d32373ac-9af3-4085-b473-60372b5f0406.png)
>> ![image](https://user-images.githubusercontent.com/39285147/197422932-1866e001-8086-4f89-a231-4582d8e304d2.png)

> **IoU**
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
**기존 Transformer**
- autoregressive (output sequence를 ***하나하나*** 넣어주는 방식) 
- pairwise interactions between elements in a sequence
- duplicate predictions 제거 가능

**새로운 Transformer**
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
![image](https://user-images.githubusercontent.com/39285147/197423492-347a9b5f-f3d1-4555-b6b4-d3bb0679dc22.png)

- Faster RCNN과 비슷한 성능

$$AP: Average Precision$$
$$AP_50: IoU > 50(correct)$$

****
# 요약 ✔
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