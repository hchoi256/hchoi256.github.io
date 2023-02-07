---
layout: single
title: "[논문분석] Binarized Neural Networks"
categories: LightWeight
tag: [Model Compression, Light-weight, Binarization]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/qntzn.png
sidebar:
    nav: "docs"
---

[논문링크: Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/pdf/1602.02830.pdf)

****
# 한줄요약 ✔
- **Bit-wise Operation**: Binarizes only weigths and activations with -1 or 1 at run-time *except for the first layer*
- Input value of the first layer does not need binarization
    - Data representation << Model internal representation
        - i.e., Input: 채널 3개 (RGB), Model: 512 多

****
# Introduction 🙌
현대에 이르러 깊은 신경망을 쌓는 Deep Neural Network (DNN)가 폭발적으로 증가함에 따라, 높은 사양의 GPU에 대한 수요 또한 급증하였다. 이러한 large model은 mobel devices 같은 low-power devices 에서는 이용 불가능하여 대중들은 **경량화** 기법에 대해 관심을 갖기 시작한다. Quantization (양자화) 방법으로 여러 가지가 있지만, 해당 논문에서는 **binarization** 이진화 기법을 통한 양자화를 구현한다.

이진화란 값을 참/거짓, -1/+1 처럼 두 개중 하나의 값으로 결과를 반환하는 이분법적인 접근 방식이다. 하여 그러한 결과값들이 필요로 하는 bit 개수는 오로지 1개이기 때문에, 극단적으로 bit 개수를 줄임으로써 inference 성능을 향상시킨다.

****
# Challenges 💣
- `Training` Requires **more epoches** than the original NN in terms of the same accuracy
    - 대신, 하나의 operation에 대한 연산 속도 및 memory size가 적다.
- Activation function 으로 사용된 함수인 htahn의 형태적 특성(기울기가 0이 되는 부분 존재) 때문에 **Gradient Vanishing** 현상 여전히 잔재

****
# Definition ✏
            `Given` a pre-trained FP32 model
            `Returns` a (mixed-)binarized model
            `Preserving` the accuracy of the original model with higher inference speed

****
# Proposed Method 🧿
## Binarization Functions
Binarization을 구현하는 두 가지 방법이 있다: **(1) Deterministic binarization**과 **(2) Stochastic binarization**.

### Deterministic Binarization
![image](https://user-images.githubusercontent.com/39285147/217169809-8a51f70e-07a2-48b2-bf7e-97eb83b5bc10.png)

우리가 흔히 아는 `Sign()` 비선형 활성화 함수이다. Input이 양수이면 +1, 음수이면 -1을 배출한다. 해당 함수를 hidden units으로 활용하며 network의 비선형성을 유지한다.

### Stochastic Binarization
![image](https://user-images.githubusercontent.com/39285147/217170358-1ff28116-8b51-4c4f-b03a-dd74eb37803a.png)

![image](https://user-images.githubusercontent.com/39285147/217170433-b50c4038-0312-456d-8669-e1e01dce02c3.png)

`(2)`의 방법은 이름 그대로 stochastic, 즉 random으로 양자화를 위한 bit 개수를 생성한다. $$\sigma(x)$$ 함수는 랜덤으로 $$[0, 1]$$ 범위의 확률를 배출한다. 그러한 확률값을 전달받아 -1/+1 output을 결정하는 H/W를 따로 필요로 한다는 한계가 존재하여, 해당 논문에서는 S/W 차원에서 접근이 더 수월한 `(1)` 방법으로 풀이한다.

## Differentiable Deterministic Quantization
![image](https://user-images.githubusercontent.com/39285147/217172970-1941f2ea-8b5b-431c-abf8-ae1623b184eb.png)

            Binarize()
            - Weight: Sign function
            - Activation: Htanh function
            - Input: N/A

`(1)` Deterministic 접근법에서 소개된 $$Sign()$$ 함수의 미분을 조사해보면, 대부분의 영역에서 그 미분값이 0이 되기 때문에 역전파가 불가능하다. 역전파 과정에서 Activation의 미분값을 우리가 활용하는데, $$Sign()$$ 함수의 미분값이 대부분 0이다. 어떻게 미분 가능하고, Deterministic 하게 변환할 수 있을까?

![image](https://user-images.githubusercontent.com/39285147/217171474-a027ac93-5baf-4950-95a1-25ef372c433e.png)

해당 논문은 *Hard Tanh* 비선형 활성화 함수 ($$1_{|r|<=1}$$)를 사용하여 **STE(Straight-Through Estimator)**를 구현하여 상기 문제를 해결한다.

$$g_q=Sign(r)$$

$$g_r=g_q1_{|r|<=1}$$

STE는 뉴럴 네트워크에 threshold operation이 있을 때 gradient 전달을 효과적으로 해주는 방법으로, 가령 상기 문제에 대해서는 $$-1< x < 1$$ 인 기울기 값에 대해서만 BackPropagation이 일어나게 하고, 그 외의 값들은 전부 0으로 초기화한다. 이를 **Saturation**으로 볼 수 있는데, Performance 성능 악화에 기여하는 부분을 Saturation 상태라 간주하여 학습에서 제외한다.

![image](https://user-images.githubusercontent.com/39285147/217173042-b2f0e912-5977-4f6e-b2d9-89a1cdc03afa.png)

상기 이미지는 Network 동작 원리를 한 눈에 파악하기 좋은 시각 자료이다. 순전파 과정에서의 Binarize() 함수를 보면, Activation에 대해서만 Htanh 함수를 적용한다. 그 이유는 Activation의 미분값이 역전파의 가중치 업데이트 과정에서 필요하기 때문이다. 하여 $$Sign()$$ 함수는 그 미분값이 대부분의 지역에서 0이 된다는 점에서, Activation Binarization에 대해서 올바른 쓰임이 아닐 것이다.

## SBN(Shift based batch normalization)
![image](https://user-images.githubusercontent.com/39285147/217180006-bc5326b1-04b4-4b59-9820-a6b01d00bd24.png)

BN 과정은 분산값 구하는 과정을 수반하는데, 여기서 행렬 곱셈(제곱) 연산이 사용된다. 거진 모든 피라미터가 bit-wise arithmatic으로 전개되는 Binarized NN 모델에서는 행렬 곱셈 연산이 불필요하다. 대신, 우리는 손쉽게 bit-wise 연산자를 활용하여 주어진 input에 대한 거듭제곱 값을 구할 수 있을 것이다.

> 100 << 2 $$\rightarrow$$ 10000


## SAM(Shift based AdaMax)
ADAM Optimizer Multiplication多

****
# Experiment 👀


****
# Conclusion ✨
## Strengths
- Drastically **reduce memory size** during forward pass
- **Power efficiency**
    - Accesses and replaces most arithmetic operations with bit-wise operations
- Available on-line

****
# Reference