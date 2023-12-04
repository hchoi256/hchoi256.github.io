---
layout: single
title: "[논문분석] "
categories: AIPaperCV
tag: [Computer Vision, Weakly-supervised Learning]
toc: true
toc_sticky: true
toc_label: "쭌스log"
author_profile: false
header:
    teaser: /assets/images/posts/dm.png
sidebar:
    nav: "docs"
---

<span style="color:sky"> [논문링크](https://arxiv.org/pdf/2006.11239.pdf)  </span>

****
# 한줄요약 ✔
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/e3064880-f540-49aa-b458-1ea29a24dce4)

- Noise를 점점 더해가는 forward process $$q$$, 그리고 noise($$X_T$$)로 부터 data($$X_0$$)를 조금씩 복원하는 reverse process $$p$$를 학습한다.
    - $$p$$: Image generator.
    - $$q$$: 학습 대상 X.

****
# Preliminaries 🍱
## Markov Chain
- 특정 상태의 확률은 오직 바로 직전 과거의 상태에 의존한다.

## Generative Models
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/a739a6b1-52b8-43ee-82dd-8b301c3733d2)

****
# Challenges and Main Idea💣
NA

****
# Problem Definition ❤️

****
# Proposed Method 🧿
## Forward Process
<span style="color:yellow"> $$q(x_{1:\Tau} \vert x_0) := \prod^{\Tau}_{t=1} q(x_t \vert x_{t-1})$$ </span>

- Markov Chain:
    - $$q(x_{\Tau}):=q(x_{\Tau} \vert x_{\Tau-1},x_0)$$.
        - $$x_0$$에서 시작하여 $$x_{\Tau}$$의 상태는 바로 직전 시점인 $$x_{\Tau-1}$$에 기인한다.
- <span style="color:yellow"> $$q(x_t \vert x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t \bf{I})$$ </span>.
    - $$\beta_t$$: $$t$$ 시점에서 가할 noise 정도.
    - $$\mathcal{N}$$: $$\sqrt{1-\beta_t}$$ 만큼 $$x_{t-1}$$ 시점의 정보를 반영하고, $$\beta_t$$ 만큼 noise가 부여된 $$x_t$$ 시점의 정규 분포.
- <span style="color:yellow"> $$x_t=\sqrt{\alpha_t} x_{t-1}+\sqrt{1-\alpha_t} \epsilon_{t-1}=\ldots=\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$$ </span>.
    - $$q(x_t \vert x_0)=\mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\bf{I})$$.
        - $$\bar{\alpha}_t=\prod^t_{i=1} \alpha_i$$.
        - $$\alpha_t=1-\beta_t$$.
    - $$\epsilon$$:  standard variance.


## Backward Process
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/43f897b5-8888-47ab-b649-82b5e890e89c)

<span style="color:yellow"> $$p_{\theta}(x_{0:\Tau}) :=p(x_{\Tau}) \prod^{\Tau}_{t=1} p_{\theta}(x_{t-1} \vert x_t)$$ </span>

- $$p(x_{\Tau})=\mathcal{N}(x_{\Tau};0,\bf{I})$$; 시작 시점.
- $$p_{\theta}(x_{t-1} \vert x_t) := \mathcal{N}(x_{t-1}; \bf{\mu}_{\theta}(x_t, t), \Sigma_{\theta}(x_t,t))$$
    - $$\bf{\mu}_{\theta}$$*,* $$\Sigma_{\theta}$$: 학습 가능한 파라미터.

## Loss Function
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/58fb35b4-f535-462f-91b3-5ea52fb31afc)

****
# Experiment 👀
NA

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
NA

****
# Reference
[DDPM](https://process-mining.tistory.com/182)