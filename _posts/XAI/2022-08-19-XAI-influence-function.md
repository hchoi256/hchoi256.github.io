---
layout: single
title: "설명 가능한 AI (XAI): Perturbation Map"
categories: XAI
tag: [XAI, Perturbation Map, LIME, RISE]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/xai.png
sidebar:
    nav: "docs"
---

모델의 정확한 구조나 계수는 모르는 상태에서 **그 모델에 대한 입출력만 가지고** 있는 경우 설명하는 방법이다.

입력 데이터를 조금씩 바꾸면서 그에 대한 출력을 보고, 그 변화에 기반해서 설명한다.

# Local Interpretable Model agnostic Explanations (LIME)
![image](https://user-images.githubusercontent.com/39285147/185716845-b0c63b9d-cab3-4394-be1f-9c0ccd636606.png)

상기 모형에서, 빨간색 포인트 주변에서는 **선형 모델**로 근사하여 표현하겠다는 의미이다.

어떤 분류기가 DL 모델처럼 매우 복잡한 *비선형적 특징*을 가지고 있어도 주어진 데이터 포인트에 대하여 아주 **Local**하게는 다 **선형적인 모델로 근사화**가 가능하다.

입력 데이터를 조금씩 바꾸면서 그에 대한 출력을 보고, 이렇게 나온 입출력 Pair(purturbed된 이미지, 출력확률)들을 간단한 **선형 모델**로 근사하여 설명한다.

![image](https://user-images.githubusercontent.com/39285147/185717162-dc7bf7f0-5eb3-437a-a404-ee97ab1826f1.png)

LIME은 해석 요소, **super pixels**를 교란시켜서 **local interpretation model**을 얻는 방식으로 동작한다.

## Super Pixel
![image](https://user-images.githubusercontent.com/39285147/185717221-41fa234d-29fa-468a-a594-ba00328ffba3.png)

**Super pixel**이란 유사성을 지닌 픽셀들을 일정 기준에 따라 묶어서 만든 하나의 커다란 픽셀을 말한다.
- May not capture correct regions.

## 장단점
**장점**
- **주어진 입력과 그에 대한 출력**만 얻을 수 있다면 **어떤 모델에 대해서도 다 적용할 수 있는 설명 방법**으로, **Blackbox**에 대한 해결이 가능하다.

**단점**
- **Computationally expensive** (forward propagation 多)
- Hard to apply to certain kind of models when the underlying model is **still locally non linear**.
- 객체 분류가 아니라 이미지 **전체 분류에는 취약하다**.
    - ![image](https://user-images.githubusercontent.com/39285147/185717070-c46652fa-7856-4436-8e21-f420a5a3137c.png)

# Randomized Input Sampling for Explanation (RISE)
![image](https://user-images.githubusercontent.com/39285147/185717478-0298b1e6-8d5a-42dc-a824-81451b9d07a5.png)

LIME과 비슷하게 여러 번 입력을 perturb해서 설명을 구하는 Black-box 설명 방법이다.

여러 개의 랜덤 마스킹이 되어 있는 입력에 대한 **출력 스코어**(= 확률)를 구하고, 이 마스크들에 가중치를 둬서 평균을 냈을 때 나오는 것이 출력값인 Map이다.

## 장단점
![image](https://user-images.githubusercontent.com/39285147/185717611-df50da1a-c6b4-4534-ba95-74ae1d9f9dfc.png)

**장점**
- Much clear saliency map

**단점**
- **High computational complexity** (LIME보다 더 많은 randomly generated masked images 필요로 한다)
- **Noisy due to sampling** (# masked images에 따라 설명이 달라진다)

이제, 본격적으로 [다음 글](https://hchoi256.github.io/xai/XAI-perturbation/)에서 XAI 기법들에 대하여 학습해보자.
