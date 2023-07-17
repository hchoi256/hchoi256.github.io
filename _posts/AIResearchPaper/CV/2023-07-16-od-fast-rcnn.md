---
layout: single
title: "[논문 분석] Fast RCNN"
categories: AIPaperCV
tag: [Computer Vision, Object Detection, Fast RCNN]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/od.png
sidebar:
    nav: "docs"
---

[**논문**](https://arxiv.org/abs/1311.2524)

![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/5bcd93a8-6a4e-489b-ad8e-516fa960d025)

****
# 한줄요약 ✔
![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/5e742b19-1f41-4b5b-b0c8-81afd5cef645)

- 인풋 이미지를 **CNN을 1회 적용**하여 특징맵 추출 후 selective search로 region proposals 진행. 
- **RoI(Region of Interest)**: R-CNN과 달리 CNN 인풋으로 사용하기 위해 각 후보영역을 warping하는 과정 생략
- **Multi-task Loss**: 모델을 개별 학습 시키지 않고 end-to-end로 한 번에 학습.
    - 학습 및 detection 시간 크게 감소

****
# Introdcution 😉
![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/a765f8d7-5c0a-4b24-9f55-dee3ce1780a3)

R-CNN 모델은 2000장의 region proposals를 CNN 모델에 입력시켜 각각에 대하여 독립적으로 학습시기 때문에 많은 시간이 소요된다.

하지만, Fast R-CNN은 이러한 문제를 개선하여 단 1장의 이미지를 입력받아 CNN에 적용한 후, region proposals의 크기를 warp시킬 필요 없이 **RoI(Region of Interest) pooling**을 통해 고정된 크기의 feature vector를 fully connected layer에 전달한다.

**Multi-task loss**를 사용하여 모델을 개별적으로 학습시킬 필요 없이 한 번에 학습시킵니다. 이를 통해 학습 및 detection 시간이 크게 감소한다.

****
# Proposed Method 🧿
## Fast-R-CNN 작동 방식
1. [**Selective Search**](#selective-search): region proposals 사전 추출
2. [**CNN**](#pre-trained-network): 인풋 이미지를 CNN에 통과하여 feature map 획득
3. **RoI Projection**: 1에서 생성한 region proposals들을 feature map에 서브 샘플링 비율에 맞게 조정하여 projection.
4. [**RoI pooling layer**](#roi-pooling-layer): 동일한 크기의 인풋을 받는 FNN의 인풋으로써 사용하기 위해 생성한 서로 다른 RoI를 동일한 고정 길이의 특징으로 변환.
-  RoI 영역을 고정된 크기의 그리드로 분할하고, 각 그리드 셀 내에서 최댓값을 추출하여 피처맵을 생성합니다. 이렇게 추출된 최댓값들은 동일한 크기의 고정된 특징 벡터로 구성됩니다.
5. [FC Layers](#fc-layers)
- **첫 번째 FC(Fully-Connected) 레이어**: 분류를 위한 FC 레이어, 활성화 함수 포함
- **두 번째 FC(Fully-Connected) 레이어**: 바운딩 박스 회귀를 위한 FC 레이어, 활성화 함수 포함
7. [클래스 예측 (각 클래스의 확률)](#class-prediction)
8. [바운딩 박스 회귀 (각 바운딩 박스의 좌표 조정 값)](#bounding-box-regressor)
9. [NMS](#non-maximum-supression)

## Selective Search
            Input : image
            Process : Selective search
            Output : 2000 region proposals 

## Pre-trained Network
            Input : 224x224x3 sized image
            Process : feature extraction by VGG16
            Output : 14x14x512 feature maps

![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/586be0c5-1f2b-4711-833f-4ac4d5b71912)

- Fast R-CNN은 피처맵 추출을 위해 VGG16 모델을 사용한다.
    - VGG16의 마지막 Max polling layer $$\rightarrow$$ RoI pooling layer 교체
    - VGG16의 마지막 fc layers를 class 예측과 box 회귀 결과를 리턴하도록 수정.
    - VGG16가 image와 Region proposals라는 두 가지 입력을 갖도록 수정
    - conv layer3까지의 가중치값은 고정(freeze)하고 fine tuning.
        - 논문의 저자는 fc layer만 fine tuning했을 때보다 conv layer까지 포함시켜 학습시켰을 때 더 좋은 성능을 보인다.
 
> R-CNN은 CNN부분에서 AlexNet를 사용했다.

## RoI Pooling Layer
            Input : 14x14 sized 512 feature maps, 2000 region proposals
            Process : RoI pooling
            Output : 7x7x512 feature maps

![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/aee8c60d-4cd9-4bf0-8b9d-a315714502cd)

Feature map에서 region proposals에 해당하는 관심 영역(Region of Interest)을 지정한 크기의 grid로 나눈 후 max pooling을 수행하는 방법이다.

미리 지정한 크기의 sub-window에서 max pooling을 수행하여 region proposal의 크기가 서로 달라도 고정된 크기의 feature map 추출 가능하다.

1. 입력 이미지가 CNN을 통과하여 feature map을 생성.
2. Selective search로 얻어진 region proposal은 feature map에 projection되어 RoI를 생성.
3. $$h \times w$$ 크기의 RoI는 $$h \over H$$ $$\times$$ $$w \over W$$ 크기의 sub-window로 분할.
- $$H$$와 $$W$$는 사전에 설정한 하이퍼 파라미터.
4. Max pooling: 각 sub-window에서 최대값인 output을 연결하여 최종 output을 생성.

## Fine-tuning for Detection
Fast R-CNN는 end-to-end 학습이기에 network의 모든 가중치를 학습할 수 있다.

### Hierarchical Sampling
R-CNN 모델은 학습 시 region proposal이 서로 다른 이미지에서 추출되고, 이로 인해 학습 시 연산을 공유할 수 없다는 단점이 있다.

이를 해결하고자 Hierarchical Sampling 방법으로 연산과 메모리 공유를 가능케한다.

$$N$$개의 이미지를 sampling하고 각 이미지에서 $$R \over N$$개의 RoI를 sampling.
- $$R$$: mini-batch size.

동일한 이미지에서 RoI는 순전파와 역전파 과정에서 연산량과 메모리가 공유된다.

그리고 $$N$$을 작게 설정할 수록 mini-batch 연산량은 더욱 줄어든다.

가령, $$N=1,R=128$$ 일 때, 이미지 당 128개의 RoI를 sampling.

하지만, $$N=2,\ R=128$$는 이미지당 64개의 RoI가 sampling되고 RoI 연산량과 메모리가 그만큼 공유되기 때문에, $$N=1$$일 때보다 64배 빠르다.

### End-to-end 학습 가능
이전 R-CNN에서는 SVM, bounding box regressor, network 총 3개의 학습을 별도로 진행해야 한다.

하지만, Fast R-CNN은 클래스 예측을 위한 SVM을 softmax layer로 대체하고 또 다른 output layer를 만들어 bounding box regressor를 한 가지 신경망으로 통합했다.

하여 신경망을 학습하면 softmax layer, bounding box regressor도 동시에 학습된다.

이를 위해 Fast R-CNN은 **Multi-task loss**를 활용한다.

## FC Layers
            Input : 7x7x512 sized feature map
            Process : feature extraction by fc layers
            Output : 4096 sized feature vector

## Class Prediction
            Input : 4096 sized feature vector
            Process : class prediction by Classifier
            Output : (K+1) sized vector(class score)

## Bounding Box Regressor
            Input : 4096 sized feature vector
            Process : Detailed localization by Bounding box regressor
            Output : (K+1) x 4 sized vector

## Multi-task Loss
            Input : (K+1) sized vector(class score), (K+1) x 4 sized vector
            Process : calculate loss by Multi-task loss function
            Output : loss(Log loss + Smooth L1 loss)

![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/ab765f63-5e37-4ba9-ba6a-fb9c7a778397)

- $$L_{cls}$$: class 예측 loss.
    - $$p$$: $$(K+1)$$개의 모델의 예측 클래스 점수.
    - $$u$$: 정답 객체 클래스 점수; 배경 클래스는 $$u=0$$.
- $$L_{loc}$$: bounding box에 대한 loss으로 배경 클래스는 무시한다.
    - $$v$$: $$u$$ 클래스에 해당하는 true bounding box.
    - $$t^u$$: $$u$$ 클래스에 해당하는 bounding box regression offset.
- $$\lambda$$: 두 task loss의 균형을 조절하는 하이퍼 파라미터.

이 손실함수를 통해 하나의 네트워크에서 object class와 bounding box를 동시에 학습할 수 있다.

> ![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/5b0e15ad-82f8-482f-ad3c-bc26658533ac)
>
>> L1 loss는 L2 loss보다 이상치에 덜 민감하다. L2 loss는 민감하여 기울기 폭발을 일으킬 수 있다.

## Mini-batch Sampling
SGD mini-batches는 가령 2개의 이미지에서 각 64개의 RoI를 sampling하여 총 128 mini-batch size를 이용한다.

원래는 이미지당 2000개의 RoI가 발생하는데 하기 기준을 통해 64개의 RoI를 선별한다.
- RoI의 25%: region proposal이 true bounding box와 IoU > 0.5 이상인 것에서 무작위 추출.
- RoI의 75%: RoI는 $$0.5 > IoU > 0.1$$인 것에서 추출.

## Non Maximum Supression
![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/1faf4a1f-0c23-4643-abbc-67bb07ea8c8f)

R-CNN을 통해 얻게 되는 2000개의 bounding box를 전부 다 표시할 경우우 하나의 객체에 대해 지나치게 많은 bounding box가 겹친다.

하여 각 객체별 최적의 bounding box를 선택하는 **Non maximum supression** 알고리즘을 적용한다.

1. Linear SVM에서 bounding box별로 지정한 confidence scroe threshold 이하의 box를 우선 제거.
2. 나머지 bounding box를 confidence score에 따라 내림차순 정렬.
3. 살아남은 boxes 중 confidence score가 가장 높은 box를 기준으로 다른 box와의 IoU값을 조사하여, IoU threshold 이상인 box를 모두 제거.
4. 3의 과정을 반복하여 남아있는 box만 선택한다.

****
# Conclusion ✨
## Strength
- R-CNN보다 높은 detection quality(mAP).
- Multi-task loss를 사용해 single stage 학습 (end-to-end).
    - End-to-end 학습에 따른 네트워크의 모든 layer들을 갱신(update)할 수 있다.
- Feature caching을 위한 disk storage가 필요 없다.

## Weakness
- 인풋 이미지 하나당 2,000개의 region proposal 추출하기 떄문에 학습 및 추론 속도가 느리다
- 한 네트워크에서 3가지의 모델(CNN, SVM, Box Regressor)을 사용하기 때문에 구조와 학습 과정이 복잡하고, end-to-end 학습 불가능.  

****
# Reference