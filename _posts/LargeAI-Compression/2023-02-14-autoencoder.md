---
layout: single
title: "Auto Encoder(오토 인코더) 이해하기"
categories: LightWeight
tag: [Compression, AutoEncoder]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/ml-thumbnail.jpg
sidebar:
    nav: "docs"
---

****
# 오토인코더 🙌
![image](https://user-images.githubusercontent.com/39285147/219285508-1c792bbb-0494-400f-8785-59efc18cf000.png)

**오토인코더**는 입력이 들어왔을 때 최대한 compression 시키고, 다시 본래의 입력 형태로 복원 시키는 신경망이다.
- `Latent vector`: 압축 과정에서 추출한 의미 있는 데이터로, Encoder가 자동적으로 찾아주는 사전에 미리 알 수 없는 데이터
- `Encoder`: 데이터 압축, laten vector 추출
- `Decoder`: 데이터 복원, latent vector 잘 찾기 위한 도우미 역할

![image](https://user-images.githubusercontent.com/39285147/219285860-5727f83a-24b8-443e-bb0f-c5c8cd14d8c9.png)

가령 상기 그림은 Auto Encoder 결과를 시각화한 이미지이다.
- y축: 두께 변화
- x축: 기울기 변화

하여 이 모델의 `latent vector`는 두께, 회전, Class로 정의된다.

****
# 오토인코더 응용분야 💣
## 1) Unsupervised Learning (= Self-Supervised Learning)
**Unsupervised Learning**은 사전 label 데이터 없이 학습하기 때문에, 데이터 자체에 숨겨져 있는 패턴을 발견하는 것이 목표이다.

> ![image](https://user-images.githubusercontent.com/39285147/219286692-b5c94f5f-6714-4f75-b08d-a8c1a5f293e5.png)
>
>> MNIST 데이터셋에서 compressed data(= latent vectors)를 찾기 위한 Auto Encoder 학습 방식은 사전 label 정보가 없기에 Unsupervised learning이다. 

## 2) Maximum Likelihood Density Estimation
## MLE 배경지식
![image](https://user-images.githubusercontent.com/39285147/219286983-51ae79c8-642b-4105-8240-eac821e67936.png)

상기 이미지에서, 오코인코더를 통해 7이라는 input 숫자를 축소하여(`Encoder`) 다시 7이라는 숫자로 재현(`Decoder`)할 수 있다.

Decoder의 복원 능력 검증을 위하여 입력된 7과 재현된 7의 사진의 차이를 확인해보면, 픽셀 단위로 재현된 숫자 이미지이기 때문에 그 차이가 매우 클 것이다 (**비효율적!!**).

그 차이를 효율적으로 좁히기 위한 방법 중 하나가 **MLE(Maximum Likelihood density estimation)**이다.

## MLE
MLE는 입력과 출력이 같아질 **확률**이 최대가 되길 바란다.

> 기존 예시는 단순히 입력과 출력 자체가 같아지기를 바라는 관점이다.

관측된 샘플들을 존재하게 할만한 가능성(확률)을 최대화 시키는 값($$\theta$$)을 찾는 것이 목적이다.

> $$\theta$$: 최적의 parameter.

간단히, 최적의 parameter를 찾는 과정이다,

> 우도(Likelihood)와 얽힌 MLE에 관한 보다 자세한 내용은 [여기](https://github.com/hchoi256/ai-terms)를 참조하자.

## 3) Manifold learning
**Manifold learning**의 한 가지 역할 중 하나는 바로 **차원 축소**인데, 굉장히 큰 차원을 저차원으로 줄일 수 있다.

![image](https://user-images.githubusercontent.com/39285147/219286983-51ae79c8-642b-4105-8240-eac821e67936.png)

오코인코더 관점에서, manifold learning을 이용하면 상기 예제에서 숫자 7의 이미지에 대한 latent vector, 두께, 회전, Class로 차원 축소가 가능하다.

가령, 숫자 7 이미지가 MNIST 데이터(28x28)라면, 784 차원으로 표현 될 수 있을 것이다.

이를 두께, 회전, Class로 총 3개의 차원으로 축소할 수 있는 것도 데이터를 잘 표현할 수 있는 서브 스페이스(i.e., 여기서는 두께, 회전, class)를 찾았기 때문이다.

하여 오코인코더에서 Encoder 부분에 대해 효과적으로 적용 가능한 알고리즘이 되겠다.

****
# Reference
[오코인코더](https://videolectures.net/deeplearning2015_vincent_autoencoders/)