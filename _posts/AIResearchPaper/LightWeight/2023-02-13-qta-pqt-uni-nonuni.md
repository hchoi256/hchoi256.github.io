---
layout: single
title: "Quantization 기본기 다지기"
categories: LightWeight
tag: [Quantization, Uniform, Non-uniform, QAT, PTQ]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/qntzn.png
sidebar:
    nav: "docs"
---

****
# Uniform vs Non-uniform 🙌
![image](https://user-images.githubusercontent.com/39285147/218532636-6bfaf954-949f-4ff8-8530-051745fbec47.png)

## Calibration
[*Calibrating Activations*]

![image](https://user-images.githubusercontent.com/39285147/218538899-6969ada3-6011-4450-b196-4c5fef820d9b.png)

Activation의 Calibration은 상당히 까다롭다.

Input은 Offline에서 알 수 없는 변수라서 input 이후 Calibration이 진행되어야 한다.
- Input 개수 ↑, 학습/추론 이미지의 분포 유사도 ↑ $$\rightarrow$$ Calibration 효율 ↑.

> `Max`: 원소값중 가장 큰 값의 절대값
>
> `Entropy`: 양자화된 값과 기존 FP 값의 차이를 최소로 만드는 값 
>
> `Percentile`: 입력값의 분포를 백분위수로 나타내서 0.01% 등 사용자가 설정한 값

## Uniform(균일 양자화)
![image](https://user-images.githubusercontent.com/39285147/218531666-6fe2cb58-736c-4fad-b449-2138d85f9ccc.png)

- FP Input, 가중치를 INT, 고정소수점 형태로 변환
- step size 균일
- 대칭적 양수/음수 구조
- `Round-off Error`: 양자화 스텝 크기 절반
    - 짝수 or 홀수 step 양자화

### (1) Affine Transfrom
![image](https://user-images.githubusercontent.com/39285147/218533664-d0255d17-a412-43c3-aa51-e9405c16ab99.png)

- 비대칭(Asymmetric)
- `(2)` 보다 정확도 ↑, Cost 효율 ↓

$$f(x)=s\times x+z$$

$$s=\frac{2^b-1}{\alpha-\beta}$$

$$z=-round(\beta \times s)-2^{b-1}$$

$$z$$: zero-point, 변환 전 0의 위치가 변환 후 어느 점으로 대응되는지 표현

$$s$$: scaling factor

$$quantize(x,b,s,z)=clip(round(s\times x+z),-2^{b-1},2^{b-1}-1)$$

$$dequantize(x_q,s,z)=\frac{x_q-z}{s}$$

> `Fake Image/Quantization`: 양자화 변환 후 값들이다.

### (2) Scale Transform
![image](https://user-images.githubusercontent.com/39285147/218533713-ffa45cd5-e3e4-43e9-8074-7e2d231e4f3d.png)

- 대칭(Symmetric)
- `(1)` 보다 정확도 ↓, Cost 효율 ↑

$$f(x)=s\times x$$

$$s=\frac{2^{b-1}-1}{\alpha}$$

$$quantize(x,b,s)=clip(round(s\times x),-2^{b-1}+1,2^{b-1}-1)$$

$$dequantize(x_q,s)=\frac{x_q}{s}$$

## Non-uniform(비균일 양자화)
- step size 비균일
    - 입력 신호 레벨 ↓ $$\rightarrow$$ 양자화 계단 간격 ↓
    - `Code Book`: 맵핑 방식을 결정하는 사용자가 정해놓은 Rule 

                음수: 0
                0~1: 1
                >= 1: 2

****
# Post Training Quantization ✏

****
# Quantization Aware Training 💜


****
# Reference
## Quantization Granularity
- **Activation**: tensor quantization.
- **Weights**: tensor or chennel quantization.
