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

![image](https://user-images.githubusercontent.com/39285147/219908547-cdb74ac3-77a5-4377-a135-e881d4c56d93.png)

**Stage 1 (Pretraining Phase)**:
- `Data Augmentation`: 이미지 sequence 해치지 않도록 aug 수행
- `Base Encoder`: Context feature 추출
- `Projection Head`: 이미지 representation 퀄리티 향상 
- `Instance Mapping Function`: 이미지 sequences를 sub-words로 변환하고, 각 contrastive loss 산출
- `Contrastive Loss`

> ![image](https://user-images.githubusercontent.com/39285147/219908598-20d1c262-0bbc-43c1-8442-586c1a75bb6d.png)

**Stage 2 (Fine-tuning Phase)**:
- 일반적인 자기지도학습처럼 feature extractor를 freeze 후, decoder만 학습
    - Stage 1에서 얻은 Base Encoder 정보 활용하여 decoder 학습

### Experiment
![image](https://user-images.githubusercontent.com/39285147/219908695-11b2adce-6f05-4de2-b042-2d8ab3dee058.png)

- *Window-to-instance* 방식이 대체로 우수한 성능 보임
- Seq 고려한 Contrastive Learning이 문자 인식에서 좋은 성능 보임
- instance 개수:
    - 너무 많은 instance mapping 수행 $$\rightarrow$$ misalignment pair 문제 발생.
    - 너무 적은 instance mapping 수행 $$\rightarrow$$ negative pair 개수 감소.

****
# Semi-supervised Learning 🎆
- `Pseudo-Labeling Method`: Unlabeled 데이터 예측결과 활용하여 가짜로 레이블링 후 labeled 데이터처럼 활용
- `Consistency Regularization Method`: 데이터 및 모델에 변형 후에도 예측 일관성 갖도록 학습
- `Hybrid Method`: 여러 준지도학습 알고리즘 아이디러 혼합 활용 학습.

![image](https://user-images.githubusercontent.com/39285147/219855596-24bea078-25c1-4e7d-9c01-647e468773fb.png)

- Unlabeled 데이터에 대한 모델 예측결과로 Label을 임의로 만들어주어 학습
- Labeled/Unlabeled 데이터 함께 활용 학습

## Semi-supervised Learning-based Scene Text Recognition
[[논문분석] Pushing the Performance Limit of Scene Text Recognizer without Human Annotation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Pushing_the_Performance_Limit_of_Scene_Text_Recognizer_Without_Human_CVPR_2022_paper.pdf)

![image](https://user-images.githubusercontent.com/39285147/219908893-a3c14cb7-6481-4f9a-bff9-bae0d26daed4.png)

- STR에 `Consistency Regularization` 적용
    - `Consistency Regularization`: 동일한 이미지에서 다르게 변형된 이미지를 입력으로 받더라도, 동일한 결과 갖도록 학습
- STR에 합성 데이터+실제 Unlabeled 데이터 함께 활용하는 준지도학습

일반적인 준지도학습을 STR 적용 시, 하기 한계점 존재:
- 합성이미지 ~ 실제이미지 데이터 분포 차이로 학습률 저하
- 글자 간 Misalignment 문제 발생하여 동일하지 않은 글자끼리 consistency regularization 수행되어 학습 방해

### Model Architecture
![image](https://user-images.githubusercontent.com/39285147/219908957-e62a71ef-47c9-409d-9d41-93ebffd927ce.png)

- `Encoder`: 입력 이미지 feature 추출
- `Decoder`: 이미지 단위 feature에서 글자 단위 feature 생성
- `Classifier`: 글자 단위 feature에서 각 글자들을 예측

![image](https://user-images.githubusercontent.com/39285147/219909589-d202fd8c-a6eb-4264-bbda-a78e63400a43.png)

#### Supervised Branch
![image](https://user-images.githubusercontent.com/39285147/219908976-2a765cad-44dd-4313-b43a-aa3105fde208.png)

- Labeled 데이터(합성데이터) 활용 학습
    - cross entropy 사용
- Labeled 글자들 decoder의 입력 글자로 활용
- 학습된 weights들 unsupervised branch의 online model에 공유

#### Unsupervised Branch
![image](https://user-images.githubusercontent.com/39285147/219909227-dfc89594-9cef-489b-b2b4-9e63fd42786b.png)
![image](https://user-images.githubusercontent.com/39285147/219909455-aec2b168-68d1-44fe-b300-91fa7a065803.png)

- Unlabeled 데이터 활용 학습
- Online 모델, Target Model 모두 Asymmetric    
    - Target Model의 글자 별 예측 확률 활용 Noisy 데이터 필터링
        - Threshold보다 글자별 예측 확률의 가중합이 작으면 학습에서 활용하지 않음 (상기 이미지 Score: 0.5814)
- **Character-level Consistency Regularization**: 하나의 이미지 두 번 증강 후 두 예측 값이 글자 단위로 유사해지도록 학습 (KL Divergence)
    - *Online Model*: Strong augmentation 이미지 입력 / Encoder + Decoder + Projection + Classifier / Weight Decay
    - *Target Model*: Weak Augmentation 이미지 입력 / Encoder + Decoder + Classifier / Stop Gradient (EMA 활용) 
- Autoregressive decoder: 이전 출력값 활용
    - *Target Model*: Autoregressive하게 이전 시점값 활용
    - *Online Model*: Target Model의 이전 시점 값 활용

> **EMA**: $$\theta_t=\alpha \theta_t + (1-\alpha)\theta_{\alpha}$$.

![image](https://user-images.githubusercontent.com/39285147/219909528-5251a630-4d79-43ea-ada6-f714568993bc.png)

#### Domain Adaptation
![image](https://user-images.githubusercontent.com/39285147/219909579-7ada65c8-b0cb-40a6-8bc3-dbbfda8ffcf9.png)
![image](https://user-images.githubusercontent.com/39285147/219909575-d9104325-aed7-42f3-a2d0-091fd09d9e9e.png)

합성 데이터 ~ 실제 데이터 도메인 차이 최소화
- Supervised ~ Unsupervised Branch의 Target Model의 Vision feature에서 각각 공분산 행렬을 구한 후, 이들의 차이를 통해 Domain Shift 최소화

### Summary
- **1) Supervised Branch Loss 산출**
    - 예측값 ~ 레이블 활용 교차 엔트로피
- **2) Unsupervised Branch Loss 산출**
    - 데이터 증강 2회 후 Online/Target Model 입력
    - Encoder ~ Classifier 통과 후 각 Model 예측 수행
    - Target Model 예측 확률값 통해 Score 점수 산정 후 Threshold 비교하여 Noisy 데이터 학습 미반영
    - Noisy 아니라면, 글자 단위 Consistency Loss 산출 (Context Information 공유)
- **3) Domain Adaptation Loss 산출**
    - Supervised Branch와 Unsupervised Branch의 Target Model 활용 Loss 산출
- **4) Overall Loss 산출**
- **5) Weight Update**

### Experiment
![image](https://user-images.githubusercontent.com/39285147/219909733-4f28dcda-f9f8-4590-97f1-696da6319885.png)

- 해당 연구 준지도학습 >> 기존 지도학습
- 해당 연구 준지도학습 vs. 타 준지도학습
    - depends!

****
# Self&Semi-supervised Learning 🍞
[[논문분석] Multimodal Semi-Supervsied Learning for Text Recognition]

> 최근 연구 흐름: STR에 Unlabeled 데이터 활용 연구는 Vision Feature만 고려됨 / STR은 학습 위한 Labeled 데이터 매우 부족

![image](https://user-images.githubusercontent.com/39285147/219916155-6734372f-dc70-4534-9254-6628e4c91db3.png)

- **STR에 Semi, Self 모두 적용**
    - Self: Contrastive Learning
    - Semi: Consistency Regularization
- **Vision/Language 모두 고려한 Multimodal 모델**
    - Vision Model Pretraining: Constrastive Learning + Supervised Loss
    - Language Model Pretraining: Masked Language Model (MLM)으로 사전학습

> **MLM**: 특정 Text token을 가리고 가려진 부분의 text token 맞추는 방식 / unlabeled data 활용 large text corpus 사전학습

- **Fine-tuning & Fusion Model Training**
    - 각 Modality별 Prediction
    - 각 Modality별 Consistency Regularization

상기 용어들 모두 사전에 다루었던 내용이므로, 잘 읽어보면 이해될 것이다.

****
# Reference
[Self-supervised Learning](https://greeksharifa.github.io/self-supervised%20learning/2020/11/01/Self-Supervised-Learning/)

[Self/Semi-supervised Learning for Scene Text Recognition](http://dmqm.korea.ac.kr/activity/seminar/388)