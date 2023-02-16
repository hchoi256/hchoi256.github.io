---
layout: single
title: "DenseNet, SqueezeNet, MobileNet 이해하기"
categories: LightWeight
tag: [Quantization, DenseNet, SqueezeNet, MobileNet]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/qntzn.png
sidebar:
    nav: "docs"
---

# INTRO ✨
글에 들어가기 앞서 ResNet에 대한 완벽한 이해가 수반됐다는 전제를 알린다.

****
# DenseNet 👀
![image](https://user-images.githubusercontent.com/39285147/219291172-0c3e7970-445f-428a-bf37-ffa21d787670.png)

-  H() 함수: BN, ReLU, 3x3 conv

![image](https://user-images.githubusercontent.com/39285147/219298388-34d0afd9-76e2-48c6-b39e-989720ec89c0.png)

**ResNet** 
- `Residual connection`을 사용하여, function 이전 값을 `identity mapping`을 통해 **더해준다**.

**DenseNet**
- 이전 레이어를 모든 다음 레이어에 직접적으로 연결 $$\rightarrow$$ 정보 흐름(information flow)향상.
- `Residual connection`을 사용하여, function 이전 레이어 값들을 모두 **concatenate**하여 bottlenect 레이어 뒤 기존 불필요한 `1x1 conv 확장` 대신 직접 차원을 증가시킨다.
- Concat하여 늘어나는 정도를 **growth rate(k)**로 조절한다.
    - Growth rate는 각 레이어가 전체에 어느 정도 기여를 할지 결정한다.

## Pre-activation
![image](https://user-images.githubusercontent.com/39285147/219300565-be1a670d-e620-43e5-b610-fb8fd170aa88.png)

**Pre-activation**을 고려하는 이유는 원래의 ResNet에 있는 ReLU가 진정한 identity mapping 개념을 방해하기 때문에 ReLU 순서를 바꿀 수 있는지 확인하기 위함이다.

Weight/Activation/Batch Normalization의 순서 관련한 문제이다.
- **Original**: Weight 먼저
- **Pre-activation**: BatchNorm 먼저

BatchNorm $$\rightarrow$$ ReLU $$\rightarrow$$ Convolution.

## Bottle Nect Architecture
![image](https://user-images.githubusercontent.com/39285147/219301127-626a58ba-ba27-45c8-b632-94f38352904a.png)

마찬가지로 1x1 conv (= bottle neck 구조)를 사용하여 dimension을 reduction한 뒤 output들을 concatenate한다.

## Experiment
![image](https://user-images.githubusercontent.com/39285147/219304419-dc26436d-a318-4243-ad1e-558620c30c5d.png)

****
# SqueezeNet 🎄
![image](https://user-images.githubusercontent.com/39285147/219304582-97d4c8c1-fa35-4899-b52d-82d4b4b4e106.png)

## Fire Module
하이퍼 피라미터:
- `s1x1`: squeeze layer에서 1x1 filter 수
- `e1x1`: expand layer에서 1x1 filter 수
- `e3x3`: expand layer에서 3x3 filter 수

### Squeeze Layer
**목표 1: 3x3 filter로 입력되는 입력 채널의 수를 감소시킨다.**
- 3x3 filter의 conv layer 연산량은 $$(입력 채널) \times (필터 수) \times (필터 크기)$$.
- 입력 채널을 감소하면 3x3 filter 연산량 감소.
- s1x1 < (e1x1 + e3x3)로 설정하여 squeeze layer의 channel수가 expand layer의 channel수보다 작게 설정

1x1 conv layer를 사용하여 channel reduction (원하는 채널 수로 줄이기)

1x1 filter들의 출력값은 하나로 합쳐져서 expand로 전달된다.

### Expansion Layer
1x1 conv layer와 3x3 conv layer 함께 병렬 사용

Padding을 사용하여, 두 layer의 output size가 서로 일치하도록 맞춰준다

### Squeeze Ratio (SR)
![image](https://user-images.githubusercontent.com/39285147/219305188-5fb1071e-7981-420b-af03-66c414e6efe5.png)

Expand layer 앞에 있는 squeeze layer의 filter 수로써, expand layer의 filter에 대한 비율이다.

가령, SR = 0.75이고 expand layer의 필터 개수가 4개라면, squeeze layer 개수는 3개이다.

![image](https://user-images.githubusercontent.com/39285147/219309581-95d4538c-6d91-4173-9f73-53ba66b3b740.png)

## Bypass
![image](https://user-images.githubusercontent.com/39285147/219307842-a5fa0947-e8e7-44bc-9bdf-5305f75bb1b7.png)

- **single bypass**: 기존 ResNet
- **complex bypass**: 입력 채널수와 출력 채널수가 다른 경우, conv1x1 추가로 채널수 조정 

결과적으로 single bypass가 조금 더 나은 성능을 보인다.

> bottleneck 문제
>
>> squeeze layer의 파라미터 수는 expand layer보다 작아서, 적은 양의 정보가 squeeze layer를 통과한다고 생각한다. 이러한 차원 감소는 모델을 가볍게 해주지만 정보 손실을 유발하므로, bypass를 추가하여 정보 손실을 막는다.

# Experinment
![image](https://user-images.githubusercontent.com/39285147/219308760-32ae48e4-87f7-47ca-8adc-a70bcdfce9e5.png)


****
# MobileNet 🌷

****
# Reference