---
layout: single
title: "[논문분석] AlphaTuning: Quantization-Aware Parameter-Efficient Adaptation of Large-Scale Pre-Trained Language Models"
categories: Others
tag: [Model Compression, AlphaTuning, Quantization, Pruning]
toc: true
toc_sticky: true
toc_label: "쭌스log"
author_profile: false
header:
    teaser: /assets/images/posts/qntzn.png
sidebar:
    nav: "docs"
---

[논문링크](https://arxiv.org/pdf/2210.03858.pdf)

****
# 한줄요약 ✔

****
# Background 🍱
Self-supervised Learning의 등장으로, 데이터 확보가 쉬워짐에 따라 증가한 데이터 개수만큼 모델의 사이즈(parameter 개수) 또한 증가시킬 수 있게 되었다.

현 시점에서 Transformer의 막강한 power을 덕택에 AI 전반, 특히 NLP 분야에 많은 발전이 잇따르고, 거대 언어모델에 대한 수요가 증가하였다.

하여 본 논문 또한 언어모델 압축을 target으로 삼는다.

모델압축 외에, zero/few-shot learning 기반 기학습 모델의 성능 향상의 전적에도 불구하고, 기학습 모델 training 이후 fine-tuning이라는 새로운 adaptation 과정은 downstream task 성능 향상에 있어서 필수적이다.

> 각 downstream task마다 독립적인 각각의 adaptation 성능 결과를 보인다.

하여 다양한 downstream task에 범용적으로 좋은 성능을 끌어낼 수 있는 parameters만을 fine-tuning 대상으로 삼는 것이 핵심이다 (`parameter-efficient adaptation techniques`).
- Adapter modules, low-rank adapatino, prefix-tuning, etc.

하지만, 본 논문은 trainiable parameter 개수 자체는 상기 방식들로 줄일 수 있으나, 그들이 기학습 모델에 비해 fine-tuned model의 inference 성능이 그다지 좋지 않다는 점을 강조한다.

또한, 현존하는 모델압축 기술들은 반대로 parameter-efficient adapation 기법을 활용하지 않다는 점을 강조한다.

가령, 기존 압축 기법인 QAT의 경우, fine-tuning과 모델압축을 함께 사용한 접근법이지만, 기학습 모델 만큼의 memory storage를 필요로 한다.

****
# Introduction 🙌
최근 초거대 언어 모델에 대한 관심도가 나날이 증가하는 추세 속에서, 어떻게 모델의 추론 속도를 정확도를 해치지 않으면서 높일 수 있는 지에 대한 많은 연구가 이뤄지고 있다.

이러한 관심에도 불구하고, 해당 분야는 아직 밝혀지지 않은 것들이 너무 많다.

하여 본 논문은 `model compression`과 `parameter-efficient adaptation` 기술을 결합한 새로운 방식인 compression-aware parameter-efficient 기법인 **AlphaTuning**을 제시한다.

> `Parameter-efficient tuning`
>
>> 더 적은 파라미터만을 학습하여 downstream task에 대해서 fine-tuning과 비슷한 성능을 내는 것이다.

해당 기술은 기학습 모델에 대한 post-training quantization과 양자화된 몇몇의 parameters들에 대해서만 target task에 fine-tuning시킨다.

AlphaTuning은 binary-coding quantization을 수행하며, adaptation 단계에서는 binary 값들을 freeze하고 scaling factors들만 fine-tuning을 진행한다.

> `Binarization` 개념이 어색하다면, [여기](https://hchoi256.github.io/aipaperlightweight/xnor-net/)를 참조하길 바란다.

본 논문이 제시한 AlphaTuning 기법은 GPT-2와 OPT 적용 시, 4-bit quantization 활용 10배 압축 및 1000배의 parameter 개수 감소 효과를 보여준다.

> GPT는 다들 알고 계실테고, OPT는 Meta에서 선보인 초거대 언어 모델로 가장 최신 버전의 해당 모델은 1750억개 매개변수를 이용한다.

![image](https://user-images.githubusercontent.com/39285147/220860698-05776ad1-d788-422f-a4e0-5cf52b53fcef.png)

모델압축과 parameter-efficient adaptation을 수행하는 접근법은 두 가지가 있다.

상기 이미지에서, 기존 접근법은 $$A \rightarrow C \rightarrow D$$ 순서에 해당하며, 많은 trainable parameters 및 PTQ 시 downstream tasks 성능 감소가 한계로 꼽힌다.

하여 본 논문은 새로운 접근법 $$A\rightarrow B \rightarrow D$$을 제시하고, 해당 접근법을 **AlphaTuning**이라 명칭한다.

AlphaTuning은 (1)주어진 parameters을 `binary values`와 `scaling factors`로 factorization을 수행한다.
- scaling factors는 quantization formats에서 아주 작은 부분만을 차지한다.

이후, (2)binary values는 freeze하고, 작은 메모리 부분만을 차지하는 scaling factors에 대해서만 fine-tuning을 진행하여 추론 속도를 accelerate한다.

$$A \rightarrow B$$ 과정은 QAT 대신 PTQ를 수행한다; QAT는 방대한 데이터셋에 대해 훈련 시 computational overhead가 엄청나다.

<span style="color:red">

QAT 경우 overhead 줄일 수만 있다면, PTQ를 대체해도 좋을까?

</span>

****
# Problem Definition ✏
                Given a large pre-trained language model

                Return a quantized model

                Such that it outperforms the performance of the original model in terms of inference time while retaining accuracy.

****
# Challenges and Main Idea💣
## C1
- Accelerating a large LM using binarization is accompanied by a non-trivial reduction in accuracy.

## C2
- How can we wisely remove many redundant parameters in the adaptation phase?

## C3
- What is the ace in the hole for the combination of model compression and parameter-efficient techniques without sacrificing memory storage?

## Idea
- Freezes all binarized parameters while just fine-tuning its scaling factor.

****
# Proposed Method 🧿


****
# Major Takeaways 😃

****
# Open Reivew 💗

****
# Experiment 👀

****
# Conclusion ✨

****
# Reference