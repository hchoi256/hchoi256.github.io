---
layout: single
title: "[논문 분석] Swin Transformer (ICCV 2021)"
categories: AIPaperCV
tag: [Computer Vision, Transformer, Swin Transformer]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/swin.png
sidebar:
    nav: "docs"
---

[논문링크](https://arxiv.org/abs/2103.14030)

<!-- <span style="color:blue"> ???? </span> -->

****
# 한줄요약 ✔
- **다양한 Vision Task**에 적합한 구조.
- **Local Window**를 적용하여 inductive bias 완화.
- **Patch Merging**을 통해 레이어 간 계층적 구조를 형성하여 이미지 특성을 고려합니다.

> **이미지 특성**: 해상도 혹은 전체 이미지에 존재하는 객체의 크기 및 형태.

****
# Preliminaries 🍱
## ViT 모델
![image](https://github.com/hchoi256/ai-boot-camp/assets/39285147/8204eabe-c472-4f9b-b289-f2c22c8f41b3)

ViT 논문에 대한 설명은 [여기](https://hchoi256.github.io/aipapercv/vit/)를 참조해주세요.

## Local Window
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/45a5c7cf-98fe-4104-951c-8ab940a73f95)

- $$H$$: 이미지 세로 길이.
- $$W$$: 이미지 가로 길이.
- $$C$$: 이미지 채널수.
- $$P_h$$: 패치 세로 길이.
- $$P_w$$: 패치 가로 길이.
- $$N$$: 패치 개수.
- $$M$$: local window 크기.
- $$n$$: local window 개수.



****
# Problem Definition ✏
                Given a 2D image dataset

                Return a more efficient Transformer-based Vision model on the dataset

                Such that it outperforms the performance of the original model in terms of inference time while retaining accuracy.

****
# Challenges and Main Idea💣
**C1)** <span style="color:orange"> 기존 ViT 모델은 Classification Task를 풀기 위한 모델로 제안되었습니다.</span>

**C2)** <span style="color:orange"> 기존 ViT 모델은 텍스트와 달리 이미지를 위한 특성이 없습니다. </span>

**C3)** <span style="color:orange"> 기존 ViT 모델은 입력 Token의 개수가 증가함에 따라 Transformer 구조상 quadratic한 Time Complexity를 갖습니다. </span>

**Idea)** <span style="color:lightgreen"> 본 논문의 Swin Transformer는 **Local Window** 및 **Patch Merging**을 도입하여 상기 문제들 해결합니다. </span>

****
# Proposed Method 🧿
##



****
# Experiment 👀

****
# Open Reivew 💗

****
# Discussion 🍟

****
# Major Takeaways 😃

****
# Conclusion ✨
## Strength
## Weakness

****
# Reference
[DSBA Lab](https://www.youtube.com/watch?v=2lZvuU_IIMA)