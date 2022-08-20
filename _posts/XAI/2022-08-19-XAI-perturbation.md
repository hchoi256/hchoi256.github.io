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

# 대리 분석(surrogate analysis)
![image](https://user-images.githubusercontent.com/39285147/185720735-07e8c018-f7b9-416d-b411-7590bc726878.png)

XAI에서 대리 분석이란, 설명하고자 하는 원래 모델이 지나치게 복잡해서 해석하기 어려울 때, 해석 가능한 대리 모델(surrogate model)을 사용하여 기존의 모델을 해석하는 기법을 말한다. 

상기 모형에서, 설명이 불가능한 blackbox 모델을 대리 모델인 logistic regression 모델의 회귀 계수로 설명할 수 있다.

모델 **어그노스틱(model-agnostic)**한 것이 특징이고, **global/local** 대리 분석 방법으로 나뉜다.

이제 Local 대리 분석 방법인 LIME을 살펴보자.

# Local Interpretable Model agnostic Explanations (LIME)
![image](https://user-images.githubusercontent.com/39285147/185719899-ae369fe4-a82b-4565-9897-3eadf6c032bd.png)

모든 예측 모델에 대한 결과를 해석 가능하고 신뢰할 수 있는 방법으로 설명하는 새로운 기법을 제공하는 알고리즘이다.
- i.e., 채무 불이행할 가능성이 높다는 판단에 대한 근거(설명/해석)를 제시.

**설명하고 싶은 예측 값 근처**에 대해서만 해석 가능한 모델(i.e., 선형 모델)을 학습시키는 방법이다.
- *설명하고 싶은 예측 값 근처*: 강아지 사진을 강아지라고 예측하는 모델이 있다면, 여기서 예측 값은 '강아지'이다.

> *해석*: 각 예측을 내림에 있어 어떤 피처(= 이미지 특징)가 사용되었는지에 대한 설명을 제공한다는 의미.

손실함수로 *대리 함수*를 적합하고, 그 결과를 사용해 모델이 **개별 샘플(= 강아지가 포함된 사진을 강아지라 예측할 때, '강아지')에 대해 왜 그러한 판단을 내렸는지**를 유추해볼 수 있다.

## 직관
![image](https://user-images.githubusercontent.com/39285147/185719520-6be6e6cc-c6c6-483f-9fe8-e20ecf750ca9.png)

설명하려는 데이터(굵은 빨간 십자가)의 살짝 옆에만 본다면, 그 주변만 근사한 선형 함수를 만들어낼 수 있다.

어떤 분류기가 DL 모델처럼 매우 복잡한 *비선형적 특징*을 가지고 있어도 주어진 데이터 포인트에 대하여 아주 **Local**하게는 다 **선형적인 모델로 근사화**가 가능하다.

이렇게 만든 간단한 선형 함수를 보면 이 예시에서 **어떤 변수(= 이미지 특징)가 중요한 역할**을 하는지 알 수 있고, 이것이 LIME에 **Local**이라는 말이 붙는 이유이다.

> Local: 한 개인, 혹은 한 샘플에 내려진 판단이 어떻게 내려진 것인지를 분석한다.

## 작동 원리
![image](https://user-images.githubusercontent.com/39285147/185719561-6ab4f8c1-21a1-4122-a5f4-d58120c01aff.png)

입력 데이터를 조금씩 바꾸면서(perturb) 그에 대한 출력을 보고, 이렇게 나온 입출력 Pair(purturbed된 이미지, 출력확률)들을 간단한 **선형 모델**로 근사하여 설명한다.

![image](https://user-images.githubusercontent.com/39285147/185717162-dc7bf7f0-5eb3-437a-a404-ee97ab1826f1.png)

1. **Permute data**: 새로운 fake dataset 생성

2. **Calculate distance between permutation and original observation (= similarity score)**: 원래 데이터와 새로 만든 데이터가 얼마나 다른지 측정

3. **Make prediction on new data using complex model**: 새로운 데이터를 블랙박스 모델에 넣어 라벨 예측

4. **Pick m features best describing the complex model outcome from the permuted data**: 예측을 도출해내기 위해 필요한 정보가 가장 많은(informative) 피처 m개 도출

5. **Fit a simple model to the permuted data with m features and similarity scores as weights**: 상기 과정에서 구한 피쳐 m개로 새로운 데이터를 선형 모델에서 학습

6. **Feature weights from the simple model make explanation for the complex models local behavior**: 여기서 구한 모델의 기울기(coef)는 local scale에서 해당 observation에 대한 설명

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

이제, 본격적으로 [다음 글](https://hchoi256.github.io/xai/XAI-influence-function/)에서 Influence function-based 기법들에 대하여 학습해보자.
