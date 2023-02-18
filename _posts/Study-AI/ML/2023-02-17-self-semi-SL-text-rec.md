---
layout: single
title: "Self/Semi-supervised Learning for Scene Text Recognition"
categories: Others
tag: [Self/Semi Supervised Learning, Scene Text Recognition]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
# header:
#     teaser: /assets/images/posts/qntzn.png
sidebar:
    nav: "docs"
---

****
# Preliminaries ✔
# Scene Text Spotting 
![image](https://user-images.githubusercontent.com/39285147/219853577-8745d4bb-7183-49df-b325-b2622e2a9ceb.png)

- 일상 이미지 안의 글자 검출 및 인식
    - (1) **Scene Text Detection**: 이미지 글자 인식
    - (2) **Scene Text Recognition**: 이미지 글자 출력

> **Optical Character Recognition (OCR)**
>
>> 규격화된 인쇄체 문자 인식
>>
>> ![image](https://user-images.githubusercontent.com/39285147/219853660-3ff10cc9-83be-4b26-a1db-f8561831a324.png)

****
# Scene Text Recognition (STR) 👌
## Sequence Prediction Task
![image](https://user-images.githubusercontent.com/39285147/219853744-8a79b08b-144f-41bc-9052-eaeb5413bd8b.png)

- 하나의 입력값에 대해 여러 개의 순차적 출력값
    - 전체 문자열 단위가 아닌, 각 글자 단위 학습
- Input: 이미지 / Output: 글자(label)

## STR 구성
![image](https://user-images.githubusercontent.com/39285147/219854050-61aa9f58-dcf2-43d8-95d2-2ed37210c4a9.png)

Encoder
- 1) **Transformation**: 인풋 이미지 정렬 (Affine)
- 2) **Feature Extraction**: 정렬된 이미지 to Visual Feature 추출 (ResNet)
- 3) **Sequence Modeling**: Visual Feature to Context Feature 변환 (LSTM)

Decoder
- 4) **Prediction**: Context Feature 기반 이미지 글자 예측 (Attention)

## STR 한계
- Labeling 비용 ↑
    - 2단계 절차: Detection + Labeling(표기)
- 각 나라 언어마다 input 데이터 수집 필요 

### STR 해결 (1): Synthesized Data
![image](https://user-images.githubusercontent.com/39285147/219854165-58ef7a12-7f8c-4e28-8f7d-0420d73acc4e.png)

최근 연구에서는 모델 학습에 synthesized data 활용하지만, 하기 문제점들이 여전히 존재한다.

- `Synthesized data`: 인간이 만들어낸 인위적인 데이터 (부자연스러움)
- Test set 일반화 성능 저하 가능성 ↑

### STR 해결 (2): Unlabeled Data
![image](https://user-images.githubusercontent.com/39285147/219854262-a1e02a84-986f-46c9-8354-17aa47c9f121.png)

- 소수 Labeled 데이터 존재할 때, Unlabeled 데이터 함께 활용.
    - Self-supervised Learning
    - Semi-supervised Learning
- Unlabeled data $$\rightarrow$$ 데이터 수집 비용 ↓.
- 실제 이미지 기반 $$\rightarrow$$ 일반화 성능 저하 가능성 ↓. 

> Unsupervised Learning: supervision이 아에 없는 학습.

****
# Self-supervised Learning 🙌
- `Pretext Task`: 문제를 해결하도록 훈련된 네트워크가 다른 downstream task에 쉽게 적용할 수 있는 어떤 시각적 특징을 배우는 단계 (Supervision)
- `Contrastive Learning`: 주어진 input 데이터에 Pos/Neg Pair 정의 후, 데이터 관계(비슷한 데이터는 유사도 높음) 통해 특징 학습
- `Non Contrastive Learning`: Neg Sample 정의 X, Pos Pair로만 학습 

> Pretext Task 예시: *Context Prediction*
>
>> ![image](https://user-images.githubusercontent.com/39285147/219855901-51013415-18d8-4b03-a718-876cc7ea7327.png)

> 상기 용어들에 대한 보다 자세한 설명은 하기 [References](#reference)에서 관련 concept 참조 요먕

![image](https://user-images.githubusercontent.com/39285147/219854407-4547feee-6097-4c62-87cd-e30b1767d5d2.png)

대부분 레이블이 없는 데이터셋을 이용해서 주어진 영상으로부터 임의의 패치에 대한 공간적 특성을 정확하게 학습한 뒤, feature extraction layers를 downstream task를 위한 모델로 weights는 freeze시킨 채 transfer learning하여 소량의 labeled 데이터셋을 이용해서 학습 과정을 거치는 전략이다.

입력 데이터 변형 $$\rightarrow$$ 기존 input 데이터에 지도 주는 방식으로 데이터 특징을 학습한다.
- **Stage 1**: Unlabeled 데이터로 Pretraining
- **Stage 2**: Labeled 데이터로 Fine-tuning

## Self-supervised Learning-based Scene Text Recognition
[[논문] Sequence-to-Sequence Contrastove Learning for Text Recognition](https://arxiv.org/abs/2012.10873)

![image](https://user-images.githubusercontent.com/39285147/219864470-36f5ad65-bc86-4cb1-8b20-018576cb9865.png)
- Text Recognition에 Self-supervised Learning의 Constrastive Learning 적용
- 문자인식에 Unlabeled 데이터를 함께 활용 가능한 자기지도학습 Framework 제안
    - Contrastive Learning 활용

일반적인 자기지도학습 STR 적용시 하기 한계 존재한다:
- 기존 Data Augmentation (RandAugment) Sequence 해침
- STR 모델의 Sequential 특징(출력값에 sequence 존재) 반영 어려움

하여 일반적인 이미지 분류 문제와 다르게 여러 개의 Sequential한 출력값 반영이 필요하다.

****
# Semi-supervised Learning 🙌
- `Pseudo-Labeling Method`: Unlabeled 데이터 예측결과 활용하여 가짜로 레이블링 후 labeled 데이터처럼 활용
- `Consistency Regularization Method`: 데이터 및 모델에 변형 후에도 예측 일관성 갖도록 학습
- `Hybrid Method`: 여러 준지도학습 알고리즘 아이디러 혼합 활용 학습.

![image](https://user-images.githubusercontent.com/39285147/219855596-24bea078-25c1-4e7d-9c01-647e468773fb.png)

- Unlabel 데이터에 대한 모델 예측결과로 Label을 임의로 만들어주어 학습
- Labeled/Unlabeled 데이터 함께 활용 학습

****
# Conclusion ✨


****
# Reference
[Self-supervised Learning](https://greeksharifa.github.io/self-supervised%20learning/2020/11/01/Self-Supervised-Learning/)