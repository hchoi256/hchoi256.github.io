---
layout: single
title: "CPU, GPU(CUDA), FPGA, NPU, TPU 이해"
categories: Study
tag: [CPU, GPU, NPU, TPU, FPGA, CUDA]
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
$$CPU\ 범용\ 계산기$$

## CPU 특징
[*폰 노이만 아키텍쳐*]

![image](https://user-images.githubusercontent.com/39285147/217796521-ee649b42-a210-4fdb-ada2-4981ca9ed28a.png)

- 현재 CPU는 폰노이만 구조에 기반
- 유연한 **직렬계산 능력** (이전-다음 계산 결과의 연계)
    - ALU 연산 이후 Memory 저장 필수
    - 한 번에 하나의 transaction 순차적 처리
- 평균 이상의 효율 $$\rightarrow$$ 컴퓨터, mobile devices, 임베디드 장비들의 main processor 역할
    - 복잡한 명령어세트를 실행하기 위한 정교한 아키텍쳐
        - 복잡한 명령어 세트나 많은 수의 레지스터 및 복잡한 캐시 구조를 가진다
    - 똑똑한 만큼 개별 계산당 비용이 크다

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

1. `1 CLOCK`: $$1 \rightarrow\ R1$$.
2. `2 CLOCK`: $$2 \rightarrow\ R2$$.
3. `3 CLOCK`: $$R1+R2\ \rightarrow\ R3$$ (ALU).
4. `4 CLOCK`: $$R3\ \rightarrow\ 0x00040222h$$.

> `Register`: CPU 내부 저장소
>
> `Memory`: CPU 바깥 저장소

`4 CLOCK`이 순차적으로 지난 후, 최종 addition의 결과가 CPU 외부 Memory의 0x00040222h 주소값 위치에 저장될 것입니다.

****
# Graphic Processing Unit(GPU) ✏
$$대규모\ 병렬\ 곱셈\ 계산기$$

![image](https://user-images.githubusercontent.com/39285147/217838136-115071f3-7445-4c14-93f5-3bdd6aa85c7d.png)

CPU 처리 기반에서는 고사양 게임을 모니터 화면에 표현하는 동안 키보드 입력이 안되는 문제가 발생할 수 있습니다.

이는 CPU가 직렬처리 방식에 기반하고 있어서, 모니터 화면이 다 표현될 때까지 키보드 입력 처리가 awaiting 되기 때문인데요.

이러한 한계점을 타파하고자 등장한 것이 바로 병렬처리 방식으로 동작하는 GPU입니다.

## GPU 특징
- CPU 코어의 복잡한 구조를 단순화 $$\rightarrow$$ 개별 계산 비용 최소화
- 단순한 형태의 코어 대량 집적 $$\rightarrow$$ 단순 연산 병렬 수행
- 전력소모, 내구성, 기대수명 ↓ $$\rightarrow$$ 말단 장치에서 상용화 한계점
- 고비용

## GPU 쓰임
### 그래픽 처리
GPU는 단순한 대량 계산을 복잡한 계산에 유리한 CPU로부터 독립시키기 위해 고안된 Co-processor입니다.

하여 GPU는 주로 그래픽 처리에 많이 활용되어 NVIDIA 같은 업체가 그 대표적인 예시입니다.

그래픽 처리에 필요한 계산에는 동일한 형태의 부동 소수점 곱셈 대량 수행이 전부입니다.

### AI 딥러닝 학습
AI 추론이나 학습을 할 때 핵심적으로 필요한 연산이 바로 행렬 합성곱(convolution) 연산입니다.

합성곱 연산 비중에 따라, 모델의 training/inference 성능이 결정됩니다.

#### 합성곱 연산
![image](https://user-images.githubusercontent.com/39285147/217868536-382aa0d1-61f4-4657-9ba6-fa54d734c9b1.png)

**Feature Extraction**이란 이미지에서 필터가 지정한 방향에 따른 경계선 모양을 대략적으로 알아내는 방법입니다.

이 때 사용되는 연산 기법이 바로 합성곱 연산입니다.

추출된 경계선 모양 Feature Map을 적절하게 변형하여 알고 있는 정답에 근접하도록 각 필터의 가중치를 수정해 나가는 것이 딥러닝 모델 학습 과정입니다.

Filter(커널)은 이미지 특징을 추출하기 위한 변수입니다.

딥러닝 학습을 한다는 것은 많은 필터의 계수를 변경하여 정답에 가깝게 만드는 과정입니다.

> 이 글의 취지에 어긋나는 합성곱에 대한 더 자세한 설명은 생략한다.

## CUDA
- GPU에게 곱셈을 어떻게 처리하고 결과 정리할 것인지 지시
    - 병렬계산 알고리즘 구현 가능
- C기반 GPU 계산 스케줄링 SDK

## cuDNN
- 기본적인 딥러닝 primitive 편리하게 구현

****
# Field Programmable Gate Array(FPGA) 👀
![image](https://user-images.githubusercontent.com/39285147/218299733-f29d110a-e6ca-4d90-8991-10607d81399f.png)

- 프로그래밍 가능한 논리소자 배열의 일종
- 원하는 의도애 맞게 논리회로 동작 가능
    - Logic 설계가 이루어지지 않은 형태로 납품 후 개발자가 직접 설계

## FPGA 장점
- 전력 소모량 ↓, Latency ↓, throughput ↑
    - 처음부터 Neural Network 모델 학습에 적합하도록 FPGA 구성 가능

## FPGA 단점
- 높은 가격, 사치스러운 비효율적이게 뛰어난 성능

****
# Neural Processing Unit(NPU) 🧿
## NPU 특징
- GPU에서 확장된 **신경망 학습 특화** AI 연산 병렬처리
- 딥러닝 네트워크의 각 layer를 실리콘으로 구현한 칩셋
    - ANN의 뉴런들(Convolution, FCL, etc.) 하드웨어적으로 구현
    - 하여 SW 유연함에 있어서 한계를 가짐
- GPU, cuDNN 사용할 때와 같은 performance 네트워크를 구성 불가능
    - NPU에서 구현하지 않은 DNN 뉴런은 하드웨어 가속 불가 $$\rightarrow$$ CPU 직렬연산 수행 $$\rightarrow$$ CPU 사용에 따른 성능 저하
- 스마트폰에 대부분 사용 (영상, 이미지, 음성인식 AI 스마트폰 기술)

> 만약 CPU가 FC 지원하지 않는다면, 해당 네트워크 효율 저하
>
>> ![image](https://user-images.githubusercontent.com/39285147/218299038-d097d82c-e84c-4313-96c6-3dc657b7f11c.png)

## NPU 장점
- 신경망 AI 연산 처리에 용이
- 저비용

## NPU 단점
- GPU보다 범용성 ↓
- 신경망 노드를 하드웨어적으로 구현했기에, HW 구조상 다른 AI 알고리즘 습득 한계
    - HW 구조 다른 알고리즘 구현 시, 오히려 더 큰 시간 및 비용 소모

## NPU 목적
- 기존 딥러닝 비용 최소화
    - 기존 GPU 시스템 = 비쌈, 휴대 불가능, 전력 소모 상당
    - 말단 device, 가령 휴대폰에 GPU 넣기 불가능

> Smart Phone에는 **AP(Application Processor)칩**이라는 GPU + CPU 통합 on-device가 있다.

> **Edge computing/inference**: 말단 devices에서 추론 및 학습 수행하는 것

## GPU + NPU 구조
- GPU: 모델 학습 가속
- NPU: 말단 장비에서 저전력 추론

> 점차 모델 학습 가속도 NPU 병렬처리로 바뀌는 추세

****
# TensorFlow Processing Unit(TPU) 👀
- A type of NPU by Google
- 기존 NPU와 크게 다르지 않으나, `Tensorflow lite` 호환성 ↑
- Bfloat16 기반
- Systolic matrix multiplication
- Very high throughput

****
# Conclusion ✨
![image](https://user-images.githubusercontent.com/39285147/218299539-ee113c3f-8c58-4c22-8e3e-d721e0800199.png)

****
# Reference