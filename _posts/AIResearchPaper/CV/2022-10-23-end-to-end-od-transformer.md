---
layout: single
title: "[논문 분석] End-to-End Object Detection with Transformers (ECCV 2020)"
categories: AIPaperCV
tag: [Computer Vision, Object Detection, Transformer]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/od.png
sidebar:
    nav: "docs"
---

[**논문**](https://link.springer.com/content/pdf/10.1007/978-3-030-58452-8_13.pdf)

****
# INTRO 🙌
**Object detection**은 Classification과 Localization을 합친 것으로, 사진속 객체에 대해 클래스 분류와 bounding box 위치를 표현하는 것을 말한다.

이전 객체탐지 방법들은 사진속 객체가 존재할 것 같은 후보영역을 차출하기 위해 *간접적인 방법*으로 객체 탐지를 구현했다.
- Two-stage (속도 느림, 정확성 좋음):
    - **RCNN, Fast RCNN**: Selective Search
    - **Faster RCNN**: Sliding window, Anchor boxes, RPN
- One-stage (속도 빠름, 정확성 나쁨):
    - **YOLO**: grid division

## Non-Maximum Suppression (NMS)
![image](https://user-images.githubusercontent.com/39285147/197414699-970639a6-076d-4b2b-b1de-763931c9082e.png)

- Object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 후처리 기법.
- 객체별 가장 높은 예측 확률을 보인 bounding box를 기준으로 중복되는 다른 boxes를 제거하는 후처리 기법. 

학습 단계에서 후처리 기법인 NMS를 사용하지 않고 모델을 훈련한다면, 모델이 겹치는 bounding box를 처리하는 방법을 학습할 수 없으며, 이는 최종적인 감지 성능에 영향을 미칠 수 있습니다.

~~이러한 간접적인 방법들은 *후처리*(오차 예측 및 제거)에 막대하게 영향을 받는다.~~

~~때문에 기존 모델들은 여러 가지 복잡한 예측 문제에 대한 객체 탐지 문제에서 한계를 나타낸다.~~

이러한 간접적인 pipeline(과정)에서 정확한 결과를 낼 수 없어 NMS같은 추가적인 후처리의 힘을 빌리는 surrogate tasks를 대신하고자 등장한 것이 **direct set prediction 방법**이다.

> **Surrogate**: output이 정확히 측정될 수 없을 때, 대신 output 기준을 제공한다.

우리가 흔히 아는 집합(set)은 하나의 집합 안에 중복되는 요소가 없고 순서의 제약이 없는 것이 특징이다.

Direct set prediction은 이러한 집합의 특성을 이용하여, 하나의 객체에 대해 단 하나의 bounding box만 매칭되는 것을 도와 중복되는 box의 생성을 회피하여 NMS 같은 후처리의 의존성에서 벗어난다.
- **순서 제약 X**: 각 객체별 병렬성 보장
- **중복 X**: 객체별 독립적인 하나의 bounding box 가짐.

이번 논문 주제인 **DETR(DEtection TRansformer)**은 direct set prediction(bipartite matching loss) 방법과 Transformer(non-autoregressive)을 결합한 방법이다.

****
# Preliminaries 👀
## Bipartite matching(이분매칭)
### Bipartite Graph란?
[*Bipartite Graph*]

![image](https://user-images.githubusercontent.com/39285147/197421855-3281f5d4-8b83-4983-b407-6f84649dbccc.png)

이분 그래프에서 A 그룹의 정점에서 B 그룹의 정점으로 간선을 연결 할 때, **A 그래프 하나의 정점이 B 그래프 하나의 정점만** 가지도록 구성된 것이 **이분 매칭**이다.

상기 이분 그래프에서, 좌측은 이분 그래프이지만 이분 매칭은 아니고 우측은 둘다 맞다.

이분 그래프에서의 이분 매칭에는 다음과 같은 특징이 있다.
- **중복 X**
    - Transformer 학습 시, ground truth와 디코더의 인풋인 object query가 각각의 객체와 대응되는 독립적 예측을 가능케 한다.
        - *Object query*: 디코더 인풋으로 인코더 출력으로부터 정보를 받아 사진속 각 객체에 대한 클래스와 box 위치 정보를 학습한다.
            - 디코더 인풋인 object qeury의 개수($$N$$)는 사전에 지정되는 하이퍼 파라미터로써 사진속에 존재할 것으로 생각되는 총 객체 개수보다 크게 잡는다(논문에서는 100개의 object query를 사용).
- **순서 제약 X**
    - 객체에 대한 예측값 순열 불변(invariant) $$\rightarrow$$ 객체별 parallellism 보장.
        - 개별적으로 GT(ground truth) object와 예측 object에 대한 loss를 가지고 있어서 예측된 object의 순서에 상관이 없어 병렬화가 가능

> **Gound-truth**
>> ![image](https://user-images.githubusercontent.com/39285147/197421710-7405f615-8cfb-40b7-bd86-3ee8f7346a96.png)
>>
>> 데이터의 원본 혹은 실제 값 표현

> 참고로 기존 RNN 모델은 autoregressive decoding 기반이라 object 순서가 존재해서 객체별 parallellism이 보장되지 않는다.

## Set loss function
Set loss function은 이분 매칭을 통해 매칭된 결과를 바탕으로 예측값과 Ground-truth 간의 차이를 측정하는 손실 함수입니다.
- 예측값과 GT는 모두 $$(class,\ box\ 위치)$$ 형식을 갖는다.
    - box 위치는 $$(x,y,w,h)$$ 형식으로 제공된다; $$x$$와 $$y$$는 box의 중앙이고, $$w$$와 $$h$$는 해당 box의 각각 너비와 높이이다.

이분 매칭으로 얻은 매칭 결과를 바탕으로 객체의 클래스 예측과 bounding box 예측에 대한 오차를 계산하며, 이를 종합하여 최종적인 Set loss를 계산합니다.
- DETR 모델의 목적함수([Hungarian algorithm](https://gazelle-and-cs.tistory.com/29))로 사용된다.
    - Hungarian algorithm: 가중치가 있는 이분 그래프(weighted bitarted graph)에서 maximum weight matching을 찾기 위한 알고리즘

Set loss를 최소화하는 것은 모델의 객체 감지 성능을 향상시키는데 도움이 되며, 이를 위해 이분 매칭이 중요한 역할을 합니다.

****
# DETR Model ✒
![image](https://user-images.githubusercontent.com/39285147/197411640-e6c3de0f-b4f3-4665-ae05-6a0b45c90bf3.png)

- Bipartite matching loss + transformers with (non-autoregressive) parallel decoding
- **하나의 CNN를 Transformer 아키텍쳐와 병합** $$\rightarrow$$ **직접 예측** 가능.
    - 직접 예측 가능: 입력으로 주어진 이미지나 텍스트를 모델에 주입하면, 모델이 각각의 클래스 레이블이나 텍스트의 번역 등을 직접적으로 예측할 수 있다는 것.
- Extra-long training schedule
- Auxiliary decoding losses in the transformer

> **자기회귀(AR; Autoregressive)**: 과거의 움직임에 기반 미래 예측.

## Set Prediction
**Indirect Set Prediction**
- multilabel classification (postprocessing)
    - 사진속 객체가 애매하게 겹쳐있는 영역의 near-identical boxes들은 분류 문제 해결이 어렵다.

**Direct Set Prediction**
- postprocessing-free
    - *global inference schemes* model interactions between all predicted elements to avoid redundancy.
- Non-autoregressive sequence models (i.e., recurrent neural networks)
    - bipartite matching loss            
    - permutation-invariance

## Object Detection Set Prediction Loss
![image](https://user-images.githubusercontent.com/39285147/197422840-8c8770b5-895b-4c82-b967-da083a62c4df.png)

- bipartite matching loss + transformers (non-autoregressive) parallel decoding
- $$\mathcal{L}_{match}$$: $$\mathcal{L}_{Hungarian}$$으로 구한 최적 bipartite matching(예측값 ~ GT).
    - Hungarian algorithm: 최적 bipartite matching을 탐색.

### Hungarian algorithm
![image](https://user-images.githubusercontent.com/39285147/197422872-acf77efd-3103-4008-921c-f62aa22a13fc.png)
- $$\mathbb{1}_{\{c_i \neq \emptyset \}}$$: 클래스 $$c_i$$가 존재하면 $$1$$, 아니면 $$0$$.
- $$\hat{p}_{\hat{\sigma}(i)}(c_i)$$: 클래스 $$c_i$$을 예측할 확률.
- $$\mathcal{L}_{box}(b_i,\hat{b}_{\hat{\sigma}(i)})$$: bounding box 손실값.
    - $$b_i$$: i 번째 GT 정답값의 bounding box $$(x,y,w,h)$$.
    - $$\hat{b}_{\hat{\sigma}(i)}$$:  i번째 object query 예측값의 bounding box $$(x,y,w,h)$$.

#### Bounding box loss $$\mathcal{L}_{box}$$.
![image](https://user-images.githubusercontent.com/39285147/197422932-1866e001-8086-4f89-a231-4582d8e304d2.png)
- $$L1$$: l1 normalization.
- $$\lambda$$: 하이퍼 파라미터

> **IoU**
>> ![image](https://user-images.githubusercontent.com/39285147/197422984-634e754a-c7db-47fd-9eaa-2523296a2057.png)

****
# DETR Architecture
![image](https://user-images.githubusercontent.com/39285147/197422990-0d50e9ab-0866-40d2-9940-ff3ffb91fdde.png)

## 1) Backbone
- Feature extraction
    - 각 객체에 대한 특징이 아닌 이미지 전체에 대한 특징 추출.

## 2) Transformer Encoder
Encoder의 역할이 이미지의 전체 pixel들 중 같은 객체의 pixel들 사이에 높은 attention score를 부여하여 객체의 존재 여부와 형태를 파악한다.

- Encoder에서는 이미지 특징들 간의 **상호 연관성**과 **위치 정보에 대한 문맥 정보** 이해하여 객체를 구분한다.
    - 상호 연관성:
        - 강아지 사진에서 하나의 특징 벡터는 예를 들어 강아지의 눈에 해당할 수 있습니다. 이때, Encoder는 다른 특징 벡터와 함께 학습되면서, 강아지의 눈과 다른 특징들 간의 상호 연관성을 파악합니다. 예를 들어, 강아지의 눈, 코, 귀는 모두 강아지라는 클래스 객체를 예측한다는 점에서 모두 연관되어 있음을 학습할 수 있습니다.
    - 위치 정보:
        - Encoder는 이미지 내의 특징들이 위치 정보를 가지고 있다는 것을 인지합니다. 강아지 사진에서, 강아지의 눈이 강아지의 머리 부분에 위치하고, 강아지의 코가 강아지의 얼굴 중앙에 위치하는 것을 학습할 수 있습니다.
- CNN 출력을 flatten하여 1차원의 Transformer 인코더 인풋 형식으로 맞춰준다.
- 기존 transformer encoder에 **positional encoding** 추가
    - 덕분에 autoregressive와 다르게 인풋 순서 상관 안 써도됨


## 3) Transformer Decoder
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/14330c56-d5d7-424e-9567-670d05cec198)

Decoder의 기본적인 역할은 그 객체가 어떤 객체인지를 파악하는 데에 있습니다.

**기존 Transformer Decoder**
- Masked multi-head attention: autoregressive.
    - masking을 통해 다음 token을 예측하는 autoregressive 방법

**새로운 Transformer Decoder**
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/daee7e49-fb68-42ae-a97b-95ad0e42d866)

- Multi-head attention: non-autoregressive.
    - 입력된 이미지에 동시에 모든 객체의 위치를 예측하기 때문에 별도의 masking 과정을 필요로 하지 않습니다.
- **한번에** $$N$$개의 서로 다른 object query를 통해 각 obejct를 병렬 예측.
    - Input embedding
        - *object query(= learned positional encoding)*.
            - 각각의 object query는 하나의 객체를 예측하는 region proposal에 대응.
    - $$N$$개의 object query는 디코더에 의해 $$N$$개의 output embeddings으로 변환되어 이후 FFN의 인풋으로 들어감.
- MSHA & cross attention을 통해 각 object query 간의 global 관계 학습한다.
    - **Multi-head self-attention**: 각 object query 간의 상호 연관성을 학습하여 해당 객체 이해.
        - object 끼리 비교하는 과정을 거쳐야 서로간의 공통점, 차이점을 비교하면서 해당 object에 대한 이해도가 높아질 수 밖에 없다.
            - 강아지 클래스 객체 쿼리가 msha에서 강아지 다리가 4개이고, 닭 다리가 2개라는 상호 연관성을 학습하여 강아지라는 객체는 다리가 4개라는 점을 강조하며 강아지 객체를 보다 잘 이해하게 됨.
        - Permutation-invariant
    - **Multi-head Encoder-Decoder Attention**: 각 object query가 인코더 출력(image feature maps)과 무슨 관련 있는지 조사.
        - 각각의 object query가 인코더에서 추출된 전역적인 object 정보(이미지 특성 맵)를 활용하여 상호 작용.
            - 강아지 객체 쿼리는 cross attention을 통해 인코더 출력(image feature maps)과 상호작용하여 다리가 4개라는 특징을 이해하고, 닭 객체 쿼리는 다리가 2개라는 특징을 학습한다. 
        - **Query**: Decoder의 object query.
        - **Key, Value**: Encoder 출력.

## 4) Prediction FFN
- FFN = linear layer1(박스위치회귀) $$\rightarrow$$ 활성화 함수 $$\rightarrow$$ linear layer2(클래스 예측).
- 최종 detection 예측; 바운딩 박스와 클래스 예측을 동시에 수행
- Procedure
    - 1) **box 위치 예측**: FFN $$\rightarrow$$ 상대적인 중앙값 예측
    - 2) **클래스 예측**: linear layer $$\rightarrow$$ softmax로 class 예측
        - 실제 object 외 class = $$\emptyset$$.

> FFN(Feed-Forward Network)은 일반적으로 신경망 구조에서 사용되는 개념으로, **두 개의 선형 변환(Linear Transformation) 레이어와 활성화 함수(Activation Function)로 구성**됩니다.

> Linear Layer는 FFN의 한 종류로, 선형 변환(Linear Transformation)만 수행하는 레이어입니다. Linear Layer는 입력 벡터와 가중치 행렬 간의 행렬 곱셈 연산을 수행한 후, 편향(bias)을 더하고 활성화 함수를 적용하지 않고 그대로 출력합니다.

## Auxiliary decoding losses
- 알맞은 개수의 object 예측 가능
- 각 decoder layer에 대해 prediction FFN & Hungarian loss 추가
    - 모든 prediction FFN은 같은 파라미터를 사용
- 다른 decoder layer의 input 정규화를 위해 layer norm 추가

****
# Experiments ✏
## 성능 비교: Faster R-CNN and RetinaNet
![image](https://user-images.githubusercontent.com/39285147/197423492-347a9b5f-f3d1-4555-b6b4-d3bb0679dc22.png)

$$AP: Average Precision$$

$$AP_50: IoU > 50(correct)$$

DETR은 학습 속도가 느린대신 정확도가 높은 Two-stage 방법인 Faster RCNN과 견주어도 비슷한 성능을 낸다.

****
# 요약 ✔
## DETR 장점
- DETR 정확도 SOTA R-CNN 모델 능가
- 유연한 Transformer 아키텍쳐 $$\rightarrow$$ Panoptic segmentation 성능 ↑
- 구현이 쉽다 (inference 코드 50줄 이내)
- 큰 object 탐지 성능 ↑
    - global information performed by the self-attention
        - MSHA: DETR은 멀티 헤드 어텐션을 사용하여 객체 쿼리들 간의 상호 작용을 가능하게 합니다. 큰 객체는 여러 개의 객체 쿼리들과 상호 작용하며, 이를 통해 객체의 다양한 특징과 위치 정보를 잘 반영할 수 있습니다.
    - enabled by the non-local computations of the transformer
        - Cross attention: DETR은 크로스 어텐션을 사용하여 객체 쿼리들(Object Queries)과 이미지 특징 맵(Image Feature Maps) 사이의 관련성을 계산합니다. 큰 객체는 이미지 특징 맵에서 더 넓은 영역을 차지하므로, 크로스 어텐션을 통해 객체 쿼리와 관련된 특징들이 효과적으로 추출되어 객체를 잘 표현할 수 있습니다.

- Customized layers 필요 없음 $$\rightarrow$$ better reproduction
    - 때문에 DETR은 ResNet, Transformer 프레임 워크에서도 재사용 가능 

## DETR 한계
- Training 너무 오래 걸림
    - 모델 파라미터 多
    - 학습 데이터 多
- 작은 object 탐지 성능 ↓
    - CNN 모델이 작은 객체 탐지 성능이 좋은 이유는 작은 객체에 대한 지역적인 특징을 잘 추출하는 데에 특화되어 있기 때문입니다. CNN은 작은 패턴과 지역적인 구조를 인식하는데 강점을 가지며, 이로 인해 작은 객체에 대한 정확한 탐지를 수행하는데 효과적입니다. DETR은 작은 객체 탐지 성능이 낮은 이유는 크로스 어텐션과 멀티 헤드 어텐션 메커니즘이 작은 객체의 정보를 유지하기 어렵기 때문입니다. 크로스 어텐션은 객체 쿼리와 이미지 특징 맵의 관련성을 계산하는데, 작은 객체의 특징은 더 넓은 영역으로 희석되거나 무시될 수 있으며, 멀티 헤드 어텐션은 작은 객체의 지역적인 정보를 적절히 고려하지 못할 수 있기 때문입니다. 이로 인해 작은 객체의 정확한 탐지가 어려워지는 경향이 있습니다.

****
# Reference 👓
[**GitHub Repository**](https://github.com/hchoi256/carla-research-project)

[*UW Madison CARLA Research Team*](https://cavh.cee.wisc.edu/carla-simulation-project/)

[*CARLA Simulator*](https://carla.readthedocs.io/en/latest/)

[End to End Object Detection with Transformers](https://velog.io/@long8v/End-to-End-Object-Detection-with-Transformers)

[*Hungarian Algorithm*](https://gazelle-and-cs.tistory.com/29)

[보람이 - DETR 논문 리뷰](https://powerofsummary.tistory.com/205)