---
layout: single
title: "Meta, Transfer, Multi-task, Continual Learning 차이"
categories: LightWeight
tag: [Meta Learning, Transfer Learning, Multi-task Learning, Continual Learning]
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
# Introduction 🙌
대부분의 앱들은 높은 질의 Data 수집 및 이를 Training할 Computational Power가 부족하여, 주어진 환경에 맞게 Re-Train하는 알고리즘들이 부각되고 있다.

하여 다른 Task 수행을 위한 AI 사전 학습 모델로 적은 Dataset을 가지는 또 다른 Task도 잘 수행할 수 있도록 학습시키는 방식이 대두되고 있다.

해당 방식은 Data의 양이 적고, HW 한계를 타파하는 장점을 갖고 있다.

****
# Multi-task Learning ✏
- 각 task 별 최적의 피라미터를 공유하는 하나의 거대 모델
- 새로운 dataset이 들어오면 가장 적합한 task 찾기 위한 학습

****
# Meta Learning 😜
- Few-shot Learning
- *Learning-to-Learn*
- 기학습 모델을 사용해서 전이학습보다 더 적은 dataset에서 빠르게 최적화하는 generalization 방식
    - `Model-based Approach`
    - `Metric-based model`: 저차원 공간에 새로운 데이터 맵핑하여 가장 가까운 class로 분류
    - `Optimization-based Approach`: 다수 task의 generalized model 피라미터를 새로운 task 모델의 초기 피라미터 값으로 이용 $$\rightarrow$$ 최적 피라미터 더 빠르게 검색.
        - `Model-Agnostic Meta-Learning (MAML)`: 분류 이외 강화학습 등 다양한 알고리즘 적용 가능

## Model-Agnostic Meta-Learning (MAML)
$$Model-Agnostic (모델과\ 상관없이)$$

![image](https://user-images.githubusercontent.com/39285147/218943327-ec845b73-171e-47b4-8853-797429e38793.png)

![image](https://user-images.githubusercontent.com/39285147/218943056-ebad0c8d-ed9f-4cc1-bdba-30b93f9440b6.png)

상기 그림에서 $$\theta$$가 가리키는 point는 Task 1, 2, 3에 대한 최적은 아니다.

하지만, Task 1, 2, 3 어느 곳으로든 빠르게 이동할 수 있는 point이기 때문에 $$\theta$$는 최상의 초기 시작점일 것이다.

그 시작점은 다수 task로부터 평균적으로 2회의 Gradient Descent만 발생하면 도달할 수 있는 거리에 있도록 설정한다.

이후, 새로운 Task에 맞는 최적의 $$\theta^*$$를 찾아가는 방식으로 Gradient Descent를 진행한다.

따라서, 새로운 task가 적은 dataset에 대하여 수행되어도 우리는 최적의 시작점 $$\theta^*$$에서부터 시작하기 때문애 few-shot learning에 성공한다.

****
# Transfer Learning 🥰
- Few-shot Learning
- 다른 task 기학습 모델을 사용해서, 다른 task 수행하는 적은 dataset 기반 fine-tuning 알고리즘 적용

## Catastrophic Forgetting
기학습 모델에서 얻은 weight들이 어떤 Correlation이 있는지 모르는 Deep Learning에서 새로운 Task를 배우는데 정확하게 Fine Tuning을 위해 weight를 섣부르게 바꾸면, 기존 Task 정보와 관련된 weight, 즉 정보를 잃어버리는 현상이다.

## Knowledge Distillation
- 같은 task 기학습 모델에서, 더 적은 dataset 공유하는 작은 모델로 지식(가중치) 전달

****
# Continual Learning 🌷
전이학습의 고질병 **Catastrophic forgetting (Semantic Draft)** 문제 해결을 위해 나온 알고리즘

## (1) Elastic Weight Consoliation (EWC)
Fine Tuning으로 weight를 섣부르게 건드리면, 기학습 모델이 학습한 Task 정보를 잃어버리게 된다.

하여 새로운 task에 대하여 weight를 조금씩 건드려보자는 방식이다.

기학습 모델의 weight 중에 중요한 weight에는 Regularization Term을 추가해서 살짝만 가중치를 수정한다.

![image](https://user-images.githubusercontent.com/39285147/218947973-02ffeb4c-9930-4dbf-9983-d0c017b92d15.png)

![image](https://user-images.githubusercontent.com/39285147/218950098-5407ffd1-005f-4dc7-ae79-2fdae1c1223c.png)

$$F_i$$라는 함수는 Fisher Information Matrix로 어떤 Random Variable의 관측값으로부터, 분포의 parameter에 대해 유추할 수 있는 정보의 양이다.

`L2 Reularization`과 `No Penalty 알고리즘`에서, 기존 Task A (no penalty, 파란색)가 적은 Error를 갖는 구간을 벗어나버리지만, EWC는 그 중간 지점을 교묘하게 잘 찾아가는 것을 볼 수 있다.

다르게 말하면, Task A와 B에 모두를 고려한 최적의 weight를 찾는 방식이다.

## (2) Dynamically expandable networks (DEN)
![image](https://user-images.githubusercontent.com/39285147/218950462-633f040a-1713-4454-bddf-35a7ee8c8a33.png)

다양한 문제를 풀려면, Neural Network의 Capacity가 증가해야 하기 때문에, Node 수가 증가시키는 방식에 대한 Appraoch이다.

DEN은 동적으로 Node 수를 증가시키면서, 새로운 Task에 적응해나가는 방식이다.

하기 3가지 서로 다른 process를 통해 표현될 수 있다.

> 각 Process에 따른 loss function은 (2)[https://arxiv.org/abs/1708.01547]에 잘 정리되어 있다.

### (1) Selective Re-training
Re-training을 할 주요한 weight를 선별해서 update하자는 방식이다.

Catastrophic forgetting 현상을 방지하고자, 기존의 weight들을 저장했다가 이후 다시 재활용한다 (Split and Duplication).

### (2) Dynamic Expansion
중요한 weight만으로는 Target Task에 대한 충분한 성능이 안 나올 수 있다.

Model의 Capacity가 부족하기 때문에, 노드를 추가해야 하는 상황을 말한다.

### (3) Split and Duplication
Catastrophic forgetting 현상 방지 차원에서 기존의 weight가 Threshold 이상으로 바뀌면 복사한 기존의 weight를 옆에 붙여넣는다.

1단계에서 update할 weight를 적절히 뽑지 않았으면, 3번에서 복사해서 추가될 노드가 과다하여 비효율적이거나, 혹은 Catastrophic forgetting 현상이 재발생한다. 

****
# Reference 💕
[1] J. Kirkpatrick, et al., “Overcoming catastrophic forgetting in neural networks,” Proc. Nat. Acad. Sci., vol. 114, no. 13, pp. 3521–3526, 2017.

[2] Yoon, J., Yang, E., Lee, J., Hwang, S.J. "Lifelong learning with dynamically expandable networks", ICLR, 2017