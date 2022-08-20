---
layout: single
title: "설명 가능한 AI (XAI): Quantitative Metrics & Sanity Check/Robustness"
categories: XAI
tag: [XAI, Quantitative Metrics, Sanity check, Robustness]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/xai.png
sidebar:
    nav: "docs"
---

![image](https://user-images.githubusercontent.com/39285147/185745690-c6c421c2-0fdd-4c19-997f-f3d3cab2d5be.png)

**설명 방법**들끼리 비교하는 방법론에 대한 연구이다.

# Quantitative Metrics

## Human based visual assessment
![image](https://user-images.githubusercontent.com/39285147/185745724-9f837c6c-8f9d-4735-80f2-33bac525e8f4.png)

'**사람들이 직접**' XAI 방법들이 만들어낸 설명을 보고 비교 평가하는 것이다.

### AMT (Amazon Mechanical Turk) Test
모델이 내놓은 예측 결과에 대해 사람이 어떤 예측인지 맞추는 방식으로 평가한다.

### 단점
Obtaining human assessment is **very expensive**.

## Human annotation
![image](https://user-images.githubusercontent.com/39285147/185745756-3aca4753-baa9-47d9-a03a-6b83dc2ef5cb.png)

Some metrics employ **human annotations** (localization and semantic segmentation) as **a ground truth**, and compare them with interpretation

### Pointing game
![image](https://user-images.githubusercontent.com/39285147/185745816-da164df6-dc1a-4c4c-8fe4-4f678e40cc99.png)

**Bounding box**를 활용해서 평가하는 방법으로, 테스트 이미지에서 중요하다 생각되는 부분이 사람이 지정한 bounding box 내부에 위치하면 좋은 지표이다.

상기 모형에서 수치가 클수록 human annotations와 predictions의 차이(= accuracy)가 크다,

### Weakly supervised semantic segmentation
![image](https://user-images.githubusercontent.com/39285147/185745890-e5b9fe46-f56f-4f2d-8f60-13ca960cf562.png)

어떤 이미지에 대해서 Classification label 만 주어져 있을 때, 그것을 활용하여 픽셀별로 객체의 Label을 예측하는 ***Semantic segmentation***을 수행하는 방법이다.
- Setting: Pixel level label is **not given during training**
- This metric measures the **mean IoU** between interpretation and semantic segmentation label
- mIoU of GradCAM is 49.6%

픽셀별 정답 label이 다 주어져 있지 않기 때문에 '*Weakly Supervised*'이라 불린다.
- 가령, 상기 모형의 ground-truth 이미지에는 오토바이 부분만 하늘색으로 정답 label이 주어졌다.

> IoU(Intersection over Union): 정답 Map과 이렇게 만들어낸 Segmentation map이 **얼마나 겹치는지**를 평가하는 Metric이다.

> Semantic Segmentation: 사진에 있는 모든 픽셀을 해당하는 (미리 지정된 개수의) class로 분류하는 것.
>
>> ![image](https://user-images.githubusercontent.com/39285147/185745935-052f3404-a47f-4071-9210-6f7d4d58a96b.png)

### 단점
- Hard to make the human annotations
- Such localization and segmentation labels are not a ground truth of interpretation (해석의 근거가 아니다)

## Pixel perturbation
![image](https://user-images.githubusercontent.com/39285147/185746125-1daaed5f-f9a6-404f-b1e3-ed0b7e3f4c7a.png)

If we remove an important area in image, the **logit value for class would be decreased**.

### AOPC (Area Over the MoRF Perturbation Curve)
![image](https://user-images.githubusercontent.com/39285147/185746152-23cf524b-2d89-4bf8-8e69-f2567015dd1c.png)

AOPC는 **MoRF (Most Relevant First) order 순서에 따라 입력 patch 제거할 때, logit value 감소율**을 측정한다.

주어진 이미지에 대해 XAI가 설명을 제공하면, 그 제공한 설명의 중요도 순서대로 각 픽셀들을 정렬할 수 있을 것이고, 그 순서대로 픽셀을 **교란**하였을 때, 원래 예측한 분류 스코어 값이 얼마나 빨리 바뀌는지를 측정하는 것이다.

> 중요도 정렬 알고리즘: Gradients or Sensitivity heatmaps, Guided Backprop, Integrated Gradients, Classific SmoothGrad, SmoothGrad²(SQ-SG), VarGrad

> 교란: 중요하다 생각하는 부분을 랜덤한 픽셀로 변환하는 작업이다.

### Insertion and Deletion
![image](https://user-images.githubusercontent.com/39285147/185746270-cbf098e0-00aa-4085-8629-ce8912d7bc1f.png)

**Deletion curve**
- x axis: the percentage of the **removed** pixels in the MoRF order,
- y axis: the class probability of the model (분류 성공 확률)

**Insertion curve**
- x axis: the percentage of the **recovered** pixels in the MoRF order, *starting from gray image* (무의 상태에서 이미지 특징들 하나하나 복원해가면서 )
- y axis: the class probability of the model (분류 성공 확률)

만약, Deletion curve에서 어떠한 Output을 예측하기 위해 활용된 중요한 이미지 특징을 제거하면 '분류 성공 확률' 낮아질 것 이다.

### 장단점
**장점**
- **사람의 직접적인 평가나 Annotation을 활용하지 않으면서**도 객관적인, 정량적인 평가 지표를 얻을 수 있다는 데 장점이 있다.

**단점**
- 주어진 입력의 데이터를 지우거나 추가하면서 모델의 출력값을 보는데, 이 과정이 **ML의 주요 과정을 위반**하기도 한다.
- 어떤 픽셀을 지우고 **랜덤 픽셀**로 대체했을 때, 해당 이미지는 모델을 학습시킨 **이미지 분포와 달라져서(Data Distribution Shift)** 출력 스코어의 변화가 정확하지 않을 수 있다.
    - 가령 랜덤하게 지운 모양이 동그라미라서, 이 모양 때문에 풍선이라고 모델이 착각할 수도 있다.

> *Distribution Shift*
>
> 중요한 특징점이 지워진 데이터는 기존 Model이 학습한 Training Dataset과는 완전히 다른 Distribution을 가진다. 이 때, Model의 성능저하가 'Distribution Shift'에서 온 것인지 '중요한 특징점이 삭제'되어 그런 것인지 알수 없다는 한계가 존재한다.

> ***ROAR***로 상기 단점들을 해결한다.

## ROAR
![image](https://user-images.githubusercontent.com/39285147/185746925-7de31aec-b101-4203-92d0-43a993253289.png)

XAI이 생성한 이미지에서 중요한 픽셀들을 지우고 나서, **중요한 픽셀(정보)가 지워진 데이터를 활용해서 모델을 재학습한 뒤 정확도가 얼마나 떨어졌나 평가하는 방법**이다.
- 따라서, 모델의 성능저차가 중요한 픽셀 정보 제거에 따른 결과인지, 'Distribution Shift'에서 온 것인지 판별할 수 있다.

상기 모형에서 재학습을 하지 않으면 'Data Distribution Shift' 현상 때문에 거의 선형적으로 Model의 성능이 떨어지는 것을 볼 수 있지만, 재학습을 했을 경우 정말 중요한 정보가 지워졌을 때만 성능이 떨어지는 모습이다.

### 장단점
**장점**
- Insertion이나 AOPC에 비해 더 정확한 판단이 가능하다.

**단점**
- Retraining everytime is **computationally expensive!** (픽셀 지운 후 모델을 매번 재학습해야한다)

> 'Data Distribution Shift' 현상을 극복하기 위해서는 재학습에 필요한 시간과 높은 연산량을 감수해야한다.

# Sanity checks / Robustness

## Model randomization

## Adversarial attack

## Adversarial model manipulation
