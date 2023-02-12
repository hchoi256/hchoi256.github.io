---
layout: single
title: "초거대AI 멀티모델(MultiModal)이란?"
categories: LargeAI
tag: [HyperscaleAI, MultiModal, Transformer]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/largeai.png
sidebar:
    nav: "docs"
---

****
# 한줄요약 ✔
**Multimodal Learning**. 단일 모달 데이터, 가령 이미지, 음성, 텍스트 중 한 가지 형태만을 학습에 이용하는 한계를 극복하고자 여러 모달의 데이터를 사용해 주어진 문제를 해결하는 모델 구축 방법론이다.

주로 인간의 행동 인식이나 감정 인식 등의 문제에서 활발히 연구되고 있는 분야이며, 대체로 단일 모달보다 우수한 성능을 입증한다.

멀티모달 딥러닝은 각 모달에 적합한 딥러닝 구조를 사용하여 특징벡터를 추출하고, 이를 기반으로 각 모달을 통합한다.

추출된 특징벡터를 분석 목적에 따라 어떻게 통합할지에 대한 다양한 연구들이 존재한다.

그 중에서도, 시간과 비용이 많이드는 labeling 과정을 생략한 `self-supervised learning` 기반 멀티모달 연구가 가장 각광받고 있다.

Transformer가 추출한 각 모달의 특징벡터를 `contrastive learning` 기반으로 학습시키며, 각 모달의 특징에 맞는 손실함수(NCE loss / MIL-NCE loss)를 사용하였다.

> 모달(Modal): 한 데이터의 형태

****
# Introduction 🙌
최근 딥러닝 알고리즘의 발전 및 computing 성능 확보에 따라, AI 시스템들이 인간 수준을 웃돌고 있다.

![image](https://user-images.githubusercontent.com/39285147/218303545-ef191a40-38c0-447f-9f5e-f846603f6ec4.png)

일반적으로 단일 모달의 경우, 인간 행동/감정 인식 문제를 해결하기 위하여 사람 이미지 데이터를 감정 분류를 수행하는 CNN 모델에게 집어넣는다.

하지만, 상기 task들의 경우 단순히 이미지만 가지고 정확한 감정 분류가 어려울 수 있다.

![image](https://user-images.githubusercontent.com/39285147/218303655-f627a451-a0ec-469c-a8c5-ef9e60d0c980.png)

얼굴 표정뿐만 아니라 음성 데이터 또한 감정 분류에 지대한 영향을 줄 수 있기 때문이다.

![image](https://user-images.githubusercontent.com/39285147/218303731-1b7ec64b-db07-419e-8edb-bcfa3f6dc2d4.png)

하여 여러 종류의 데이터를 고려하여 학습하고자 한 개념이 바로 **멀티모달**이다.

****
# Multimodal Deep Learning 💣
![image](https://user-images.githubusercontent.com/39285147/218304174-ee7d956e-12bf-4425-aabb-82f64e391052.png)

- 각 모달에 적합한 DL 구조 기반 특징 벡터 추출
- 두 가지 모달 통합 방식:
    - (1) [**Feature Concatenation**](#1-feature-vector-concatenation-✏)        
        - 이미지 데이터는 CNN, 텍스트 데이터는 RNN 네트워크를 통해 특징 벡터를 추출한다.

        ![image](https://user-images.githubusercontent.com/39285147/218304065-5d613180-0add-4cec-9cf0-97cc75e945f1.png)

    - (2) **Ensemble Classifier**
        - 각 모달 데이터 별 classifier의 output labels를 가지고 ensemble(voting, etc.) 진행하여 final output 예측.

        ![image](https://user-images.githubusercontent.com/39285147/218303986-d8ae18c9-5cb6-4c9e-8844-b8d7a893a0d8.png)

Feature Concat 방법이 대중적이며, 이 방식 기반 멀티모달 연구가 현재까지 어떻게 진행되어 왔는지 그 논문들의 역사를 핵심만 간단히 살펴보자.

****
# Feature Vector Concatenation ✏
## Fusion Network
[논문링크: Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks](https://arxiv.org/abs/1703.10893)

- 해당 논문은 이미지+음성 통합 데이터를 통한 음성 향상으로 음성 신호의 노이즈 최소화하는 것을 목표로 한다.
- 각 모달 별 특징 벡터를 잘 추출한 후, FC에서 분류를 수행하는 것을 기본으로 한다.

[*AVDCNN 구조*]

![image](https://user-images.githubusercontent.com/39285147/218304420-a6361c55-6149-4479-81bf-4e70fb247164.png)

이미지, 음성 데이터에 대한 각 network가 추출한 feature vector들을 concat하여 그림에서 `Merged Layer`을 만든다.

이후, FCL로 이어져 하기 손실함수를 기반으로 모델이 학습하며 최적의 parameters를 찾는다.

![image](https://user-images.githubusercontent.com/39285147/218304522-b7227a5b-1f63-4269-9aab-37eeddb5325a.png)

> CNN에 대한 자세한 이해는 [여기](https://github.com/hchoi256/ai-terms/blob/main/README.md)를 참조하길 바란다.

[*AVDCNN 모델 피라미터 개수*]

![image](https://user-images.githubusercontent.com/39285147/218304649-4f2f286f-ac83-41e1-abec-2bdcc6038d03.png)

상기 표에서 `Merged Layer`가 갖고있는 뉴런의 개수가 2804개인 모습이다.

대체로 FCL가 진행될수록 뉴런 개수가 FC2까지 줄어들며 학습이 진행된다.

> Fully Connecter Layer
>
>> ![image](https://user-images.githubusercontent.com/39285147/218305000-55820718-4661-4c6a-bb84-b342ed3ef11f.png)
>>
>> CNN/Pooling 결과를 인풋으로 받아 가중치 적용을 통해 정의된 라벨로 분류하는 구간이다.

또 다른 feature vector concatenation 기반 연구들로는 차량 사고에 큰 영향을 주는 운전자의 스트레스 수준을 분석하는 논문이 있다.


해당 논문은 운전자의 스트레스에 영향을 주는 ECG 신호, 차량 데이터, 상황 데이터 등 다양한 모달 형태의 데이터 기반 딥러닝 학습을 통해 우수한 성능을 이끌어낸 바 있다.

![image](https://user-images.githubusercontent.com/39285147/218305213-e3a95783-18be-4ae6-a5b3-0ebc514ca21a.png)

해당 task는 time series가 주요한 영향을 끼칠 수 있기에, network에 **LSTM**을 FC 직전 추가하여 시간 연속성 특성을 얻는 모습이다.

> LSTM에 대한 자세한 이해는 [여기](https://github.com/hchoi256/ai-terms/blob/main/README.md)를 참조하길 바란다.

## Transformer
[논문링크: Vatt: Transformers for Multimodal Self-supervised Learning](https://arxiv.org/abs/2104.11178)
- 해당 논문은 주어진 Label에 없는(unlabeled dataset) 대규모 데이터셋에 대한 최적의 multimodal feature 추출을 목표로 한다.

> Self-supervised Learning에 대한 자세한 이해는 [여기](https://github.com/hchoi256/ai-terms/blob/main/README.md)를 참조하길 바란다.

[*VATT Transformer 구조*]

![image](https://user-images.githubusercontent.com/39285147/218306210-15c95a76-a03e-4f3a-8930-6e5e71024c56.png)

각 모달 데이터 토큰화 이후, Linear Projection을 통해 나온 값을 Encoder 입력으로 사용하는 모습이다.

![image](https://user-images.githubusercontent.com/39285147/218306284-6ce92b69-54bb-4b76-80b7-42a4afdb7676.png)

- Transformer에서 추출된 여러 모달의 특징 벡터를 `contrastive learning` 기반으로 학습한다.
    - Video-audio pair: NCE-loss
    - video-text pair: MIL-NCE

가령, 바람 영상은 바람 소리와의 유사도가 높을 것이고, 이 경우 NCE-loss가 낮게 나타날 것이다.
- NCE-loss는 손실함수로써, 그 값이 크다는 것은 모델의 분류 결과가 정확하지 않다는 것을 의미한다.

> `Contrastive Learning`
>
>> 비슷한 것들끼리는 유사도가 높고, 다른 것들은 유사도가 낮게끔 전개하는 방식이다.

****
# Reference
[1] Hou, J. C., Wang, S. S., Lai, Y. H., Tsao, Y., Chang, H. W., & Wang, H. M. (2018). Audio-visual speech enhancement using multimodal deep convolutional neural networks. IEEE Transactions on Emerging Topics in Computational Intelligence, 2(2), 117-128.

[2] Rastgoo, M. N., Nakisa, B., Maire, F., Rakotonirainy, A., & Chandran, V. (2019). Automatic driver stress level classification using multimodal deep learning. Expert Systems with Applications, 138, 112793.

[3] Akbari, H., Yuan, L., Qian, R., Chuang, W. H., Chang, S. F., Cui, Y., & Gong, B. (2021). Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. arXiv preprint arXiv:2104.11178.