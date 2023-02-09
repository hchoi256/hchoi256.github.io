---
layout: single
title: "CPU, GPU(CUDA), NPU, TPU 이해"
categories: Study
tag: [CPU, GPU, NPU, TPU, CUDA]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/gpu.jfif
sidebar:
    nav: "docs"
---

****
# 한줄요약 ✔
- 처리방식
    - 직렬: CPU
    - 병렬: 나머지 GPU, TPU, NPU

****
# Introduction 🙌
현대에 이르러 인공지능 모델의 사이즈가 급증하면서, AI 모델 학습이 필수적인 더 나은 하드웨어 칩에 대한 수요 또한 증가하는 추세입니다.

다양한 종류의 하드웨어 칩이 현재 시장에서 활용되고 있으나, 딥러닝 학습과 관련하여 대표적으로 사용되왔던, 그리고 새로운 차세대 강자로 각광받는 칩들에 대해 알아봅시다.

****
# CPU(Centralized Processing Unit) 💣
## CPU 구조
[*폰 노이만 아키텍쳐*]

![image](https://user-images.githubusercontent.com/39285147/217796521-ee649b42-a210-4fdb-ada2-4981ca9ed28a.png)

- 현재 CPU는 폰노이만 구조에 기반
- 유연한 **직렬계산 능력** (이전-다음 계산 결과의 연계)
    - ALU 연산 이후 Memory 저장 필수
    - 한 번에 하나의 transaction 순차적 처리

## CPU 연산
### Instruction Set Architecture(ISA)
각기 다른 종류의 CPU는 실행 가능한 기계어 모음(Instruction Set Architecture, ISA)을 갖습니다.

가령, 하기 오직 3개의 register를 갖는 간단한 ISA가 있다고 가정해봅시다.

                명령어      설명
                LDR	        데이터 로드 (--> CPU)
                STR	        데이터 이동 (CPU --> Memory) 
                ADD	        Addition

                LDR reg1, #1;
                LDR reg2, #2;
                ADD reg1, reg2, reg3;
                STR reg3, [0x00040222h];

1. $$1 \rightarrow\ R1$$.
2. $$2 \rightarrow\ R2$$.
3. $$R1+R2\ \rightarrow\ R3$$ (ALU).
4. $$R3 $$\rightarrow$$ 0x00040222h$$.

> `Register`: CPU 내부 저장소
>
> `Memory`: CPU 바깥 저장소

****
# GPU ✏

****
# NPU 🧿


****
# TPU 👀


****
# Conclusion ✨


****
# Reference