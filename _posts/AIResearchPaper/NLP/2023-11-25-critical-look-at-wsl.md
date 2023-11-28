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

<span style="color:lightgreen"> ???? </span>

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
# Proposed Method 🧿
**Given** a pre-trained model on $D_w$ ~ $\mathcal{D}_n$.

**Return** a model.

**Such that** it generalizes well on $D_{test}$ ~ $\mathcal{D}_c$.

****
# Setup 👀
## Formulation
- $\mathcal{X}$: feature.
- $\mathcal{Y}$: label space.
  - $\hat{y}_i$: labels obtained from weak labeling sources; could be different from the GT label $y_i$.
- $D={(x_i,y_i)}^N_{i=1}$.
  - $D_c$: clean data distribution.
  - $D_w$: weakly labeled dataset.
  - $\mathcal{D}_n$: noisy distribution.
- The goal of WSL algorithms is to obtain a model that generalizes well on $D_{test} ∼ D_c$ despite being trained on $D_w ∼ D_n$.
- baseline: $RoBERTa-base$.

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
## Weakness

****
# Reference
NA