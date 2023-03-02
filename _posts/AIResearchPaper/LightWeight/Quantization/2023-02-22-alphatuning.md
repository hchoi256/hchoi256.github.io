---
layout: single
title: "[논문분석] AlphaTuning: Quantization-Aware Parameter-Efficient Adaptation of Large-Scale Pre-Trained Language Models"
categories: AIPaperLightWeight
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

A $$\rightarrow$$ B 과정은 QAT 대신 PTQ를 수행한다; QAT는 방대한 데이터셋에 대해 훈련 시 computational overhead가 엄청나다.

<span style="color:yellow"> QAT 경우 overhead 줄일 수만 있다면, PTQ를 대체해도 좋을까?</span>

****
# Related Work 😉
## Large-Scale Language Models and Quantization
기학습 Transformer 언어모델은 기존의 NLP 디자인 및 배포 방식을 전면적으로 변화시켰다.

최근에, 초거대 언어모델에 대한 확장된 접근성은 새로운 자연어 처리의 시대를 열었고, few-shot learning과 parameter-efficient adapation 같은 기술의 재발견을 끌어낸다.

Quantization은 근본적인 초거대 언어모델에 대한 공간 및 계산 시간 효율 해결책으로 손꼽히고 있지만, 기존 방법들은 양자화된 상태에서 제한된 영역과 task adapationability를 제공하는 한계를 지니고 있었다.

## Parameter-Efficient Adaptation of LMs
초거대 모델들이 판을 치는 현 시점에서 언어모델을 효율적으로 downstream task에 adapting하는 것은 이 사회의 최고 관심사이다.

한 가지 전도유망한 접근방식은 (1)`in-context learning (ICL)`라는 것인데, 이것은 주어진 prompt 대한 패턴들로 부터 배우고 예측하는 언어모델이다.

해당 기법은 초거대 언어모델들에 대하여 parameter-tuning 없이 합리적인 few-shot 성능을 끌내고, 수많은 확장 연구들이 탄생해왔다.

또 다른 방식으로는 (2)parameter-efficient LM adapation을 위해 외부 혹은 부분적으로 내부 parameters (i.e., continuous prompt embeddings)를 이용하는 것인데, 이것은 특정 prompt prefixes가 더 나은 특정 LM 행동 방식에 관여할 수 있다는 아이디어에서 착안한다.

> `Continuous/soft prompts`
>
>> Additional learnable parameters injected into the model

과거 연구들은 **discrete** prompt token space를 조사했지만, 이후 **continuous** work embedding space를 최적화하는 것이 더 나은 결과를 보여줬다.

> Prompt tuning과 관련한 더 자세한 내용은 [P-tuning](https://velog.io/@seopbo/GPT-Understands-Too) 기법을 찹조하기 바란다.
>
>> `P-tuning`은 기학습 언어모델의 모든 weight를 fine-tuning하지 않고, `continuous prompt embeddings`만 tuning하는 방법이다.

(3)또 다른 연구로는 새로운 parameters들을 Transformer blocks이나 부분적으로 기존 parameters을 훈련시키기도 하며, (4)마지막으로는 parameter-efficient fine-tuning 방식과 관련된 모든 기존 접근 방식들을 통합하기도 했다.

****
# Problem Definition ✏
                Given a large pre-trained language model

                Return a fine-tuned model after quantizing the PLM

                Such that it outperforms the performance of the quantized model after the adaptation in terms of inference time while retaining accuracy.

****
# Challenges and Main Idea💣
**C1)** Accelerating a large LM using binarization is accompanied by a non-trivial reduction in accuracy.

**C2)** How can we wisely remove many redundant parameters in the adaptation phase?

**C3)** What is the ace in the hole for the combination of model compression and parameter-efficient techniques without sacrificing memory storage?

**Idea)** Freezes all binarized parameters while just fine-tuning its scaling factor.

****
# Proposed Method 🧿
## Quantization for AlphaTuning
본 논문은 느리고, 비싼 QAT 기법 대신, PTQ 기법 중 `Binary Coding Quantization (BCQ)`라 불리는 binarization 기법을 활용한다.

Binary 양자화는 극단적인 lower precision을 취함으로써, 극강의 압출률을 달성할 수 있지만, 정확성을 많이 잃기 마련이다.

### BCQ Format
![image](https://user-images.githubusercontent.com/39285147/221071718-3299a693-964b-4baa-8a7c-6cfb6fd5f507.png)

                q 증가할수록, 정확도 상승 | g 증가할수록 압축률 손해

- Weight vectors: $$w \in \mathbb{R}^g \approx \Sigma^{q}_{i=1}\alpha_i b_i$$.
    - 1 $$q$$: the number of quantization bits.
    - 2 $$\alpha \in \mathbb{R}$$ a scaling factor to be shared by $$g$$ weights.
    - 3 $$b \in \{-1,+1\}^g$$: a binary vector.
    - 4 $$g$$: (hyper-parameter) a group size or the number of weights sharing a common sacling factor.

여기서 $$\alpha,\ B_i$$는 하기의 간단한 미분을 통한 수식 연산으로 도출할 수 있다.

![image](https://user-images.githubusercontent.com/39285147/221073463-80dccef8-ae3a-4227-8dd2-72bc9e82cf7f.png)

하여 $$q=1$$의 경우 상기 이미지처럼 손쉬운 연산으로 $$\alpha,\ B_1$$를 도출 가능하고, 나머지 경우는 `greedy approximation` 같은 heuristic methods를 통해 전개한다.
- `Greedy approximation`: 상기 식에서 $$q>1$$의 경우, $$q=1$$ 경우의 $$\alpha,\ B_1$$ 값 먼저 구하고 나서, $$q=2$$ 경우의 피라미터 구하는 식으로 단계별 전개.

#### Row-wise Quantization
![image](https://user-images.githubusercontent.com/39285147/221074535-d2ede1e0-4c23-40b6-9732-efbb4adc2db6.png)

For $$W \in \mathbb{R}^{h_out \times h_in},\ g=h_in$$.

<span style="color:yellow"> CNN에서 Depth-wise convolution처럼 row-wise 대신 group-wise처럼 달리하면 연산량이 더 감축되지 않을까? </span>

Binarization: $$W \approx \Sigma^{q}_{i=1}diag(\alpha_i)*B_i$$.

![image](https://user-images.githubusercontent.com/39285147/221074822-86f3287b-2aa1-4f82-a6f8-3f73392bcbb6.png)

- Input X는 양자화 X: $$B$$가 BCQ 적용되었기에, Input은 굳이 양자화하지 않아도 이진화 전용 XNOR 연산으로 복잡한 FP 연산을 피하기 가능.
- Activation은 양자화 진행 X 
    - <span style="color:yellow"> 더 나은 quantization 수준을 위해 activation은 양자화에서 제외한다 하였으나, 사실 binary가 gradient를 표현하지 못해서가 아닐까? </span>

> [XNOR 연산](https://hchoi256.github.io/aipaperlightweight/xnor-net/)

### Transformer Quantization
[*Medium-sized GPT-2 model withhidden size($$h$$) of 1024*]

![image](https://user-images.githubusercontent.com/39285147/221075842-1f66a2bc-4460-4703-a745-bc8ba9d551df.png)

상기 이미지에서 Transformer의 weights가 상당한 memory footprint를 차지하는 모습이다.

하여 weights에 대해서 양자화 및 fine-tuning을 진행하고자 한다.

ACD로 이어지는 AlphaTuning 구조에서 scaling factor만 각기 다른 downstream task에 fine-tuning 되는 모습이다.

여기서, $$h=1024$$임을 감안하면, scaling factor의 row의 크기는 $$q=[1~4]$$인데 이것만 fine-tuning하게 되면 inference 시 행렬 곱셈 연산에서 상당한 압축 효과를 보일 것이다.

## AlphaTuning: Efficient Fine-Tuning of Quantized Models
### AlphaTuning Principles
- **Fine-tunes** scaling factor(= affine parameter)
- **Freezes** biases, binary values B, and those of the normalization layer and embedding layer

#### Training Algorithm
![image](https://user-images.githubusercontent.com/39285147/221127362-968e9b8b-771b-46e7-89cd-8858d622d964.png)

- .$$\mathbb{I}$$: $$h_{out}$$-long all-ones vector.
- .$$g_L$$: group size.

순/역전파 모두 quantized values 기반 학습을 수행한다.

`(3)` 역전파 과정에서 Chain Rule 기반 편미분으로 피라미터들의 미분식을 도출할 수 있고, `(4)` $$g_L$$로 $$\alpha$$ updates가 크게 변동하는 현상을 최소화하여 성능의 안정성를 도모한다.

하여 scaling factor에 대한 fine-tuning을 downstream task에 대해 진행하게 된다.

### AlphaTuning for GPT-2
[*GPT-2 medium and larnge on WebNLG dataset*]

![image](https://user-images.githubusercontent.com/39285147/221097697-1b331cbd-c140-4be0-a548-4b48013aef57.png)

![image](https://user-images.githubusercontent.com/39285147/221127662-b7ccffe5-883f-436e-b7ac-3685df2a7040.png)

![image](https://user-images.githubusercontent.com/39285147/221128028-ca00202d-dad8-4dbe-b34b-8276084241b8.png)

![image](https://user-images.githubusercontent.com/39285147/221129587-7228534d-7689-4856-912a-dd3aac2c6f2e.png)

상기 이미지들은 GPT-2 M, L 모델 성능 지표로써, AlphaTuning이 Figure 1) $$A \rightarrow C \rightarrow D$$ 방식인 LoRA 및 단순 FT보다 BLEU 정확성은 거진 근사하면서 압축률을 크게 낮춘 모습이다.
- 학습가능한 parameters 크기 감소.
- BLUE 지표 경쟁 모델들에 근사

결과적으로 AlphaTuning with the 3-bit quantization가 가장 성능이 우수한 것으로 나타난다.

![image](https://user-images.githubusercontent.com/39285147/221132720-3477e84b-c2d8-4c7b-bfa0-961a2059e05b.png)

- `Learning rate`와 `weight decay factor`는 본 논문에서 직접 best 값을 찾아 고정값으로 이용하였고, 나머지 모든 hyper-parameters는 **(Hu et al., 2022) for WebNLG** 논문에서 지정한 값을 고정으로 사용한다.

#### Hyper-Parameter Selection
![image](https://user-images.githubusercontent.com/39285147/221125934-7b4d747f-07d2-44ee-8b01-80314d84ecad.png)

모델 학습에 활용되는 다른 hyper-parameter 구성에 대한 실험도 진행되었다. 

앞서 언급했던 것처럼, $$\alpha_i$$는 greedy methods를 통해 순차적으로 구해진다.
- Linear decay learning rate w/o dropout.

<span style="color:yellow"> scaling factors에 대해서도 threshold를 부여하여 기준치 미달 node들은 dropout 처리해도 되지않을까? </span>

> 모든 $$\alpha$$를 한 번에 학습하는 것은 **Table 2**에서 볼 수 있듯 marginal gains만을 얻으니, greedy methods 써도 무방하다는 주장인 것 같다.

<span style="color:yellow"> Alternating vs. Greedy; 절대적으로 Greedy가 더 좋다고 말할 수 있나? </span>

각 시도마다 5번 째 epoch에서 test scores을 기록하고, random seed를 바꿔서 마치 cross validation처럼 총 5번의 시도에서 얻은 test scores들의 기대값을 구한다.
- 각 seed는 사전에 고정되었다.

<span style="color:yellow"> 5번 epoch로 충분한가? </span>

<span style="color:yellow"> Hyper-Parameter 세팅이 달라지면 AlphaTuning의 성능이 역전될수도 있지 않을까? </span>

****
# Experiment 👀
## GPT-2 Models on DART and E2E
![image](https://user-images.githubusercontent.com/39285147/221133940-8f8a3722-0892-43d2-8459-bd9c42d4ce89.png)

## OPT Models on MNLI and SAMSum
![image](https://user-images.githubusercontent.com/39285147/221133978-a3f60d71-3b5b-46d8-be44-031f13b7c397.png)

****
# Open Reivew 💗
TBD

****
# Major Takeaways 😃
- First successful compression-aware parameter-efficient adaptation method
- Only scaling factors (0.1% of the model size) are enough for successful adaptations
- High scores even under 4-bit quantization throughout various LMs and downstream tasks

****
# Conclusion ✨
## Strength
- Stable performance on **various downstream tasks**
- **Significant infernece boost** with a binary neural network

## Weakness
- GPT-2, 1.3B OPT보다 더 큰 초거대 언어모델의 경우에는 성능이 달라질 수 있다 (실험환경 한계).
    - 본 논문에서는 일반적으로 모델 사이즈가 클수록 압축률이 크고 및 정확도 감소률이 적다는 믿음에 의지한다. 
- AlphaTuning은 full FT 기법보다 추론 속도가 느리게 나타난다.
    - 본 논문은 이 한계를 AlphaTuning 학습 방법론에 대한 정보 부족을 한계로 꼽는다.
- 그 외 포스트에서 <span style="color:yellow"> 노란색 </span>으로 표시된 자가질문들 또한 약점이 될 수도 있다.

****
# Reference
[P-tuning](https://velog.io/@seopbo/GPT-Understands-Too)

[BCQ](https://arxiv.org/pdf/2206.09557.pdf)

[XNOR 연산](https://hchoi256.github.io/aipaperlightweight/xnor-net/)