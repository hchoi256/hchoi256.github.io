---
layout: single
title: "[논문분석] Weaker Than You Think: A Critical Look at Weakly Supervised Learning"
categories: AIPaperNLP
tag: [NLP, Weakly-supervised Learning, Semi-supervised Learning]
toc: true
toc_sticky: true
toc_label: "쭌스log"
author_profile: false
header:
    teaser: /assets/images/posts/ai-thumbnail.jpg
sidebar:
    nav: "docs"
---

[논문링크](https://arxiv.org/pdf/2305.17442.pdf)

****
# 한줄요약 ✔
- Weakly-supervised Learning은 주로 noisy labels가 많다.
  - 여전히 fully-supervised model 보다 성능이 안좋은 이유.
- <span style="color:orange"> 기존 WS 방법들은 과장되었다 </span>.
  - **Clean validation samples**:
    - Correct labels를 가진 데이터로 검증하는 것; Early stopping, meta learning 등의 이유로 사용.
    - <span style="color:lightgreen"> 차라리 이것을 training dataset에 포함시켰으면 성능이 더 좋게 나오더라 </span>.
      - 이것을 training data로 사용하여 fine-tuning한 모델이 이를 validation set으로 활용한 최신 WSL 모델보다 성능이 좋더라.. (*Figure 1*)
        - WSL 모델들은 단순 weak labels 보다는 성능 좋긴함..
- <span style="color:orange"> 기존 모델은 단순히 linguistic correlation이 비슷한 것끼리 엮이도록 학습하였는데 이것은 **편향**때문에 일반화 성능 저하될 수 있다 </span>.
  - <span style="color:lightgreen"> 긍정적인 것에 부정적인 레이블을 강제하여 further tuning → 일반화 성능 향상 </span>.
- Contributions:
  - Validation samples를 training data로써 활용하여 fine-tuning.
    - Revisiting the true benefits of WSL.

****
# Preliminaries 🍱
## Weak Supervision
- definition
  - proposed to ease the annotation bottleneck in training machine learning models.
  - uses weak sources to automatically annotate the data
- drawback
  - its annotations could be noisy (some annotations are incorrect); causing poor generalization
  - solutions
    - to re-weight the impact of examples in loss computation
    - to train noise-robust models using KD
      - equipped with meta-learning in case of being fragile
    - to leverage the knowledge from pre-trained LLM
- datasets
  - WRENCH
  - WALNUT

## Realistic Evaluation
### Semi-supervised Learning
- often trains with a few hundred training examples while retaining thousands of validation samples for model selection
  - 레이블 사람이 수동으로 달아주는 것 한계
- discard the validation set and use a fixed set of hyperparameters across datasets
  - 일일히 각 데이터셋에 최적 하이퍼파라미터 조합을 사람이 찾음;
- prompt-based few-shot learning
  - sensitive to prompt selection and requires additional data for prompt evaluation
    - few-shot learning에 모순!
  - prompt 예시: 다음 리뷰가 긍정적인지 부정적인지 판단하십시오:”
- recent work
  - fine-grained model selection 생략
  - Number of validation samples strictly controlled

### 의의
- To our knowledge, no similar work exists exploring the aforementioned problems in the context of weak supervision.

****
# Challenges and Main Idea💣
**C1)** <span style="color:orange"> 기존 WS 방법들은 데이터 활용이 과장되었다. </span>

**Idea)** <span style="color:lightgreen"> 차라리 이것을 training dataset에 포함시켰으면 성능이 더 좋게 나오더라 </span>

**C2)** <span style="color:orange"> 기존 WS 방법들은 학습 방법에 편향이 존재한다. </span>

**Idea)** <span style="color:lightgreen"> 긍정적인 것에 부정적인 레이블을 강제하여 further tuning → 일반화 성능 향상 </span>

****
# Problem Definition ❤️
**Given** a pre-trained model on $$D_w$$ ~ $$\mathcal{D}_n$$.

**Return** a model.

**Such that** it generalizes well on $$D_{test}$$ ~ $$\mathcal{D}_c$$.

****
# Methodology 👀
## Setup
### Formulation
- $$\mathcal{X}$$: feature.
- $$\mathcal{Y}$$: label space.
  - $$\hat{y}_i$$: labels obtained from weak labeling sources; could be different from the GT label $$y_i$$.
- $$D={(x_i,y_i)}^N_{i=1}$$.
  - $$D_c$$: clean data distribution.
  - $$D_w$$: weakly labeled dataset.
  - $$\mathcal{D}_n$$: noisy distribution.
- The goal of WSL algorithms is to obtain a model that generalizes well on $$D_{test} ∼ D_c$$ despite being trained on $$D_w ∼ D_n$$.
- baseline: $$RoBERTa-base$$.

### Datasets
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/49678179-8970-4b82-b37f-4512e2011e37)

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/62fdabd9-7077-4f7c-833e-3685d0f3229d)

- eight datasets covering different NLP tasks in English

### WSL baselines
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/554c8620-e38b-495e-a36a-b28b60a2571a)

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/74975c44-6e64-46ac-9c81-fe3abe4f428b)

- $$FT_W$$: standard fine-tuning approach for WSL.
- $$L2R$$: meta-learning to determine the optimal weights for each noisy training sample.
- $$MLC$$: meta-learning for the meta-model to correct the noisy labels.
- $$BOND$$: noise-aware self-training framework designed for learning with weak annotations.
- $$COSINE$$: self-training with contrastive regularization to improve noise robustness further.

y축(*Relative performance improvement over weak labels*):

$$G_{\alpha}={(P_{\alpha}-P_{WL}) \over P_{WL}}$$

- $$P_{\alpha}$$: the performance achieved by weak labels.
- $$P_{WL}$$: a certain WSL method.

**⇒ Without clean validation samples, existing WSL approaches do not work.**

<span style="color:yellow"> 과연 정말 clean data가 없으면 WSL 성능이 안 좋을 수밖에 없나? </span>.

## Clean Data
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/e141050b-eae2-40a9-a6cc-49ab6580f9d2)

**⇒ a small amount of clean validation samples may be sufficient for current WSL methods to achieve good performance**

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/675f3182-cc92-49b8-894f-20254ce3c29d)

**⇒ the advantage of using WSL approaches vanishes when we have as few as 10 clean samples per class**

## Continuous Fine-tuning (CFT)
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/a6034902-43dc-4028-8ee0-c83a72599219)

- **CFT**
  - In the first phase, we apply WSL approaches on the weakly labeled training set, using the clean data for validation.
  - In the second phase, we take the model trained on the weakly labeled data as a starting point and continue to train it on the clean data.

**⇒ the net benefit of using sophisticated WSL approaches may be significantly overestimated and impractical for real-world use cases.**

- 그냥 FT 한 것만으로도 기존 WSL 방법들 상당한 성능 향상 (even when # clean data = low)
- L2R의 Yelp 데이터셋 결과의 경우 CFT 이후 오히려 성능이 떨어진 모습인데, 이것은 L2R가 validation loss를 사용하여 파라미터를 업데이트하기 때문에 검증 샘플의 가치가 큰 영향을 주지 않았을지도..

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/fb8c45c6-5c48-4887-9914-2959edf88873)

**⇒ Pre-training on more data clearly helps to overcome biases from weak labels.**
- pre-training provides the model with an inductive bias to seek more general linguistic correlations instead of superficial correlations from the weak labels

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/ef30d31e-ab38-4528-a989-31c7d9249168)

**⇒ contradictory samples play a more important role here and at least a minimum set of contradictory samples are required for CFT to be beneficial**

****
# Open Reivew 💗
NA

****
# Discussion 🍟
NA

****
# Major Takeaways 😃
NA

****
# Conclusion ✨
## Strength
- If a proposed WSL method requires extra clean data, such as for validation, then the simple FTW+CFT baseline should be included in evaluation to claim the real benefits gained by applying the method.

## Weakness
- it may be possible to perform model selection by utilizing prior knowledge about the dataset
- For low-resource languages where no PLMs are available, training may not be that effective
- We have not extended our research to more diverse types of weak labels

****
# Reference
NA