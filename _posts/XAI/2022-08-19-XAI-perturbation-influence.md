---
layout: single
title: "설명 가능한 AI (XAI): Perturbation Map & Influence Map"
categories: XAI
tag: [XAI, Perturbation Map, LIME, RISE, Inception]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/xai.png
sidebar:
    nav: "docs"
---

****
# Perturbation Map

모델의 정확한 구조나 계수는 모르는 상태에서 **그 모델에 대한 입출력만 가지고** 있는 경우 설명하는 방법이다.

입력 데이터를 조금씩 바꾸면서 그에 대한 출력을 보고, 그 변화에 기반해서 설명한다.

## 대리 분석(surrogate analysis)
![image](https://user-images.githubusercontent.com/39285147/185720735-07e8c018-f7b9-416d-b411-7590bc726878.png)

XAI에서 대리 분석이란, 설명하고자 하는 원래 모델이 지나치게 복잡해서 해석하기 어려울 때, 해석 가능한 대리 모델(surrogate model)을 사용하여 기존의 모델을 해석하는 기법을 말한다. 

상기 모형에서, 설명이 불가능한 blackbox 모델 대신, 그 대리 모델인 logistic regression 모델로 학습하여 얻은 회귀 계수를 가중치로써 활용하여 특정 피처에 대한 중요도를 설명할 수 있다.

모델 **어그노스틱(model-agnostic)**한 것이 특징이고, **global/local** 대리 분석 방법으로 나뉜다.

이제 Local 대리 분석 방법인 LIME을 살펴보자.

## Super Pixel
![image](https://user-images.githubusercontent.com/39285147/185717221-41fa234d-29fa-468a-a594-ba00328ffba3.png)

**Super pixel**이란 유사성을 지닌 픽셀들을 일정 기준에 따라 묶어서 만든 하나의 커다란 픽셀을 말한다.
- May not capture correct regions.

## Local Interpretable Model agnostic Explanations (LIME)
![image](https://user-images.githubusercontent.com/39285147/185719899-ae369fe4-a82b-4565-9897-3eadf6c032bd.png)

**대리 분석 방법**을 사용해서 모든 예측 모델에 대한 결과를 해석 가능하고 신뢰할 수 있는 방법으로 설명하는 새로운 기법을 제공하는 알고리즘이다.

**설명하고 싶은 예측 값 근처**에 대해서만 **국소적으로** 해석 가능한 모델(i.e., 선형 모델)을 학습시키는 방법이다.

> *해석*: 각 예측을 내림에 있어 어떤 피처(= 이미지 특징)가 사용되었는지에 대한 설명을 제공한다는 의미.

### 직관
![image](https://user-images.githubusercontent.com/39285147/185719520-6be6e6cc-c6c6-483f-9fe8-e20ecf750ca9.png)

설명하려는 데이터(굵은 빨간 십자가)의 살짝 옆에만 본다면, 그 주변만 **근사한 선형 함수**를 만들어낼 수 있다.

어떤 분류기가 DL 모델처럼 매우 복잡한 *비선형적 특징*을 가지고 있어도 주어진 데이터 포인트에 대하여 아주 **Local**하게는 다 **선형적인 모델로 근사화**가 가능하다.

[*설명력 공식*]

![image](https://user-images.githubusercontent.com/39285147/185719561-6ab4f8c1-21a1-4122-a5f4-d58120c01aff.png)

- *L*: 손실함수
- *f*: 원본 모델 (black box 모형)
- *g*: 선형 모댈
- *π*: Super pixel
- *Ω(g)*: 모델 복잡도 (Regularization)

설명력은 설명 모델(선형 모형)의 예측이 얼마나 원본 모델 f(e.g. xgboost model)의 예측과 가까운지를 바탕으로 손실을 측정한다.

이 때, 그러한 선형 형태를 띄는 국소적 부분에 대하여 손실함수로 *대리 함수*를 적합하고, 그 결과를 사용해 모델이 **개별 샘플에 대해 왜 그러한 판단을 내렸는지**를 유추해볼 수 있다.

![image](https://user-images.githubusercontent.com/39285147/185717162-dc7bf7f0-5eb3-437a-a404-ee97ab1826f1.png)

이를 통해, **어떤 변수(= 이미지 특징)가 blackbox 모델의 예측 결과에서 중요한 역할**을 하는지 알 수 있고, 이것이 LIME에 **Local**이라는 말이 붙는 이유이다.

> Local: 한 개인, 혹은 한 샘플에 내려진 판단이 어떻게 내려진 것인지를 분석한다.

### 작동 원리
입력 데이터를 조금씩 바꾸면서(perturb) 그에 대한 출력을 보고, 이렇게 나온 입출력 Pair(purturbed된 이미지, 출력확률)들을 간단한 **선형 모델**로 근사하여 설명한다.

1. **Permute data**: 입력 데이터 주변에서 새로운 fake dataset 생성

2. **Calculate distance between permutation and original observation (= similarity score)**: 원래 데이터와 새로 만든 데이터가 얼마나 다른지 측정 (*이후, 선형결합 모델 fitting에서 가중치로 사용된다*)

3. **Make prediction on new data using complex model**: 새로운 데이터를 블랙박스 모델에 넣어 라벨/카테고리 예측

4. **Pick m features best describing the complex model outcome from the permuted data**: 그러한 예측을 도출해내기 위해 필요한 가장 많은 정보를 가진 피처 m개 도출 (예측에 주요한 피처만 다룬다)

5. **Fit a simple model to the permuted data with m features and similarity scores as weights**: 상기 과정에서 구한 피쳐 m개로 새로운 데이터를 선형 모델에서 학습 (2에서 구한 차이를 선형모델에 가중치로 부여한다)

6. **Feature weights from the simple model make explanation for the complex models local behavior**: 선형 모델에서 도출된 기울기(coef)는 관심있는 관측값에 대한 블랙박스 모형의 예측을 설명한다.

### 장단점
**장점**
- **주어진 입력과 그에 대한 출력**만 얻을 수 있다면 **어떤 모델에 대해서도 다 적용할 수 있는 설명 방법**(*Model Agnostic*)으로, **Blackbox**에 대한 해결이 가능하다.

**단점**
- **Computationally expensive** (forward propagation 多)
- 국소적으로 바라본 부분이 선형이 아니라 여전히 **locally non linear**.
- 객체 분류가 아니라 이미지 **전체 분류에는 취약하다**.
    - ![image](https://user-images.githubusercontent.com/39285147/185717070-c46652fa-7856-4436-8e21-f420a5a3137c.png)

## Randomized Input Sampling for Explanation (RISE)
![image](https://user-images.githubusercontent.com/39285147/185717478-0298b1e6-8d5a-42dc-a824-81451b9d07a5.png)

LIME과 비슷하게 여러 번 입력을 perturb해서 설명을 구하는 Black-box 설명 방법이다.

여러 개의 랜덤 마스킹이 되어 있는 입력에 대한 **출력 스코어**(= 확률)를 구하고, 이 마스크들에 가중치를 둬서 평균을 냈을 때 나오는 *Saliency Map*을 출력값으로 내보낸다.

> Saliency Map: 이미지에서 중요한 부분을 강조하는 heatmap으로써 출력을 내보낸다.

### 장단점
![image](https://user-images.githubusercontent.com/39285147/185717611-df50da1a-c6b4-4534-ba95-74ae1d9f9dfc.png)

**장점**
- Much clear saliency map

**단점**
- **High computational complexity** (LIME보다 더 많은 randomly generated masked images 필요로 한다)
- **Noisy due to sampling** (# masked images에 따라 설명이 달라진다)

****
# Influence function-based
![image](https://user-images.githubusercontent.com/39285147/185722555-aabac63f-ef87-4bd5-90d4-69a2c5cac6f2.png)

해당 테스트 이미지 분류에 있어서 Training image 샘플 하나를 제외하고 학습했을 때 Classification Score가 얼만큼 변할지 근사하는 함수이다. 

가장 큰 영향력을 행사한 Training 샘플을 설명으로 제공한다.

> 통계에서 비슷한 메커니즘을 갖는 [우도비 검증](https://github.com/hchoi256/ai-terms)이라는 것을 참조하자.

## SVM vs. Inception
![image](https://user-images.githubusercontent.com/39285147/185722593-1f43bcff-395b-492a-b636-3bc6fdf3d14f.png)

**SVM**
- Training image 중에서 Test image와 **색깔**이 비슷하고, 해당 Training image를 학습에서 제외 후 결과적으로 Classification Score 변화가 가장 큰 Training image를 찾는다.

**Inception(CNN)**
- 실제 Test image(열대어)와 비슷한 Training image를 직접 찾는다 (제대로된 특징들을 더 잘 뽑아내서 학습한다).

****
# Reference
["Why Should I Trust You?" Explaining the Predictions of Any Classifier](http://sameersingh.org/files/papers/lime-kdd16.pdf)

이제, 본격적으로 [다음 글](https://hchoi256.github.io/xai/XAI-quantitative/)에서 설명 방법들끼리 비교하는 방법론들에 대하여 학습해보자.
