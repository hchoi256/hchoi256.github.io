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

## Knowledge Distillation
- 같은 task 기학습 모델에서, 더 적은 dataset 공유하는 작은 모델로 지식(가중치) 전달

****
# Continual Learning 🌷
- 전이학습의 고질병 **Catastrophic forgetting (Semantic Draft)** 문제 해결을 위해 나온 알고리즘

****
# Conclusion ✨


****
# Reference 💕