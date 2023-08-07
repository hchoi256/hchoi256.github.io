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
**Object detection**은 Classification과 Localization을 합친 것으로, 사진속 객체에 대해 클래스 분류와 bounding box 위치를 표현하는 것을 말합니다.

이전 객체탐지 방법들은 사진속 객체가 존재할 것 같은 후보영역을 차출하기 위해 *Surrogate Task(간접적인 방법)*으로 객체탐지를 구현했습니다:
- Two-stage (속도 느림, 정확성 좋음):
    - **RCNN, Fast RCNN**: Selective Search
    - **Faster RCNN**: Sliding window, Anchor boxes, RPN
- One-stage (속도 빠름, 정확성 나쁨):
    - **YOLO**: Grid Division.

****
# Preliminaries 👀
## Non-Maximum Suppression (NMS)
![image](https://user-images.githubusercontent.com/39285147/197414699-970639a6-076d-4b2b-b1de-763931c9082e.png)

- 객체별 가장 높은 예측 확률을 보인 bounding box를 기준으로 중복되는 다른 boxes를 제거하는 후처리 기법. 

후처리 기법인 **NMS**이 없다면, 모델이 하나의 객체에 대한 여러 개의 bounding box를 처리할 수 없습니다.

이는 모델이 End-to-End 학습이 아닌 간접적인 pipeline를 갖고 있음을 의미합니다.

하여 정확한 결과를 낼 수 없어 NMS같은 추가적인 후처리의 힘을 빌리는 surrogate tasks를 대신하고자 등장한 것이 **direct set prediction 방법**입니다.

> **Surrogate**: output이 정확히 측정될 수 없을 때, 대신 output 기준을 제공합니다.

## Direct Set Prediction
우리가 흔히 아는 집합(set)은 하나의 집합 안에 중복되는 요소가 없고 순서의 제약이 없는 것이 특징입니다.

**Direct set prediction**은 이러한 집합의 특성을 이용하여, 하나의 객체에 대해 단 하나의 bounding box만 매칭되는 것을 도와 중복되는 box의 생성을 회피하고 NMS 같은 후처리의 의존성에서 자유롭습니다.
- **순서 제약 X**: 각 객체별 병렬성 보장.
- **중복 X**: 객체별 독립적인 하나의 bounding box 가짐.

## Bipartite matching(이분매칭)
![image](https://user-images.githubusercontent.com/39285147/197421855-3281f5d4-8b83-4983-b407-6f84649dbccc.png)

상기 Bipartite Graph에서 A 그룹의 정점에서 B 그룹의 정점으로 간선을 연결 할 때, **A 그래프 하나의 정점이 B 그래프 하나의 정점만** 가지도록 구성된 것이 **이분 매칭**입니다.

상기 이분 그래프에서, 좌측은 이분 그래프이지만 이분 매칭은 아니고 우측은 둘다 맞습니다.

이분 그래프에서의 이분 매칭은 다음과 같은 Direct set prediction의 특징을 갖습니다:
- **중복 X**
- **순서 제약 X**

> 참고로 기존 RCNN 시리즈 모델은 **autoregressive** decoding 기반이라 **object 순서가 존재**해서 객체별 parallellism이 보장되지 않습니다.

## Set loss function
Set loss function은 이분 매칭을 통해 매칭된 결과를 바탕으로 예측값과 Ground-truth 간의 차이를 측정하는 손실 함수입니다.
- 예측값과 GT는 모두 $$(class,\ box\ location)$$ 형식을 갖습니다.
    - box 위치는 $$(x,y,w,h)$$ 형식으로 제공된다; $$x$$와 $$y$$는 box의 중앙이고, $$w$$와 $$h$$는 해당 box의 각각 너비와 높이입니다.
    
> **Gound-truth**
>
>> ![image](https://user-images.githubusercontent.com/39285147/197421710-7405f615-8cfb-40b7-bd86-3ee8f7346a96.png)

이분 매칭으로 얻은 매칭 결과를 바탕으로 **객체의 클래스 예측과 bounding box 예측에 대한 오차**를 계산하며, 이를 종합하여 최종적인 Set loss를 계산합니다.
- 이는 DETR 모델의 목적함수([Hungarian algorithm](https://gazelle-and-cs.tistory.com/29))로 사용됩니다.
    - Hungarian algorithm: 가중치가 있는 이분 그래프(weighted bitarted graph)에서 maximum weight matching을 찾기 위한 알고리즘입니다.

Set loss를 최소화하는 것은 모델의 객체 감지 성능을 향상시키는데 도움이 되며, 이를 위해 이분 매칭이 중요한 역할을 합니다.

****
# Methodology  ✒
## DETR Model
![image](https://user-images.githubusercontent.com/39285147/197411640-e6c3de0f-b4f3-4665-ae05-6a0b45c90bf3.png)
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/b53e3024-b4a6-4897-9336-eddb08a5cca7)

- Bipartite matching loss + transformers with (non-autoregressive) parallel decoding
- 하나의 CNN를 Transformer 아키텍쳐와 병합
- 50줄 이내의 간단한 코드 구현

> **자기회귀(AR; Autoregressive)**: 과거의 움직임에 기반 미래 예측.

## Set Prediction
**Direct Set Prediction**
- postprocessing-free
    - NMS 같은 후처리 기법이 필요 없습니다.
- Non-autoregressive sequence models.
    - bipartite matching loss            
    - permutation-invariance

## Object Detection Set Prediction Loss
### Matching Cost $$\mathcal{L}_{match}$$
![image](https://user-images.githubusercontent.com/39285147/197422840-8c8770b5-895b-4c82-b967-da083a62c4df.png)

<span style="color:yellow"> $$\mathcal{L}_{match}=-\hat{p}_{\hat{\sigma}(i)}(c_i)+\mathbb{1}_{c_i \neq \emptyset} \mathcal{L}_{box}(b_i,\hat{b}_{\hat{\sigma}(i)})$$ </span>

$$\mathcal{L}_{match}$$는 Groud-Truth와 모델 예측 결과의 순열 중에서 최적의 조합쌍을 탐색합니다.

이 Loss는 역전파 시 기울기가 흐르지 않으며, 단순히 최적의 조합쌍을 구하는 것에 목적이 있습니다.

### Hungarian Algorithm $$\mathcal{L}_{hungarian}$$
![image](https://user-images.githubusercontent.com/39285147/197422872-acf77efd-3103-4008-921c-f62aa22a13fc.png)
- $$\mathbb{1}_{\{c_i \neq \emptyset \}}$$: 클래스 $$c_i$$가 존재하면 $$1$$, 아니면 $$0$$.
- $$\hat{p}_{\hat{\sigma}(i)}(c_i)$$: 클래스 $$c_i$$을 예측할 확률.
    - $$log$$를 씌워주는 이유는 $$\mathcal{L}_{box}$$ 값과 크게 차이가 나지 않도록 균형을 맞추기 위함 입니다.
- $$\mathcal{L}_{box}(b_i,\hat{b}_{\hat{\sigma}(i)})$$: bounding box 손실값.
    - $$b_i$$: i 번째 GT 정답값의 bounding box $$(x,y,w,h)$$.
    - $$\hat{b}_{\hat{\sigma}(i)}$$:  i번째 object query 예측값의 bounding box $$(x,y,w,h)$$.

$$\mathcal{L}_{match}$$를 통해 구한 최적의 조합쌍 $$\hat{\sigma}$$과 GT 간의 Hungarian Loss를 계산합니다.

$$\mathcal{L}_{match}$$과의 차이는 Negative Log Likelihood (NLL)을 클래스 예측 손실함수에 추가했다는 점입니다.

이는 Loss는 역전파 시 기울기가 흐르기 때문이며, NLL로 정의하여 기울기 값을 구할 수 있도록 만듭니다.

### Bounding box loss $$\mathcal{L}_{box}$$.
![image](https://user-images.githubusercontent.com/39285147/197422932-1866e001-8086-4f89-a231-4582d8e304d2.png)
- $$L1$$: l1 normalization.
- $$\lambda$$: 하이퍼 파라미터

#### IoU
![image](https://user-images.githubusercontent.com/39285147/197422984-634e754a-c7db-47fd-9eaa-2523296a2057.png)

서로 다른 영역 사이의 교집합 부분을 합집합 영역으로 나누어서 IoU 값을 구합니다.

$$\mathcal{L}_1$$ 혹은 $$\mathcal{L}_2$$은 IoU값이 달라도 그 값이 같을 수 있기 때문에 객체 탐지 분야에서는 보통 IoU를 손실 함수에 활용합니다.

#### GIoU
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/a28fda19-e7d3-496a-a786-914b21223612)

기존 IoU는 서로 다른 두 영역의 교집합이 없을 경우(mutually exclusive), 그 값이 0이라서 학습에 영향이 없습니다.

하여 이러한 문제점을 해결하고자 **GIoU**라는 방법을 사용하는데, 이것은 서로 다른 두 영역의 교집합이 없더라도 loss를 정의할 수 있습니다.

> 그 외에도 GIoU가 가진 단점을 보완한 DIoU 등 여러 Metric이 존재합니다.

****
# DETR Architecture
![image](https://user-images.githubusercontent.com/39285147/197422990-0d50e9ab-0866-40d2-9940-ff3ffb91fdde.png)

## 1) Backbone
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/6093b949-ba1f-4060-94bc-6ae9d9438a18)

1) **인풋**: 입력 전체 이미지 $$\mathcal{X} \in \mathbb{R}^{H_0 \times W_0 \times C}$$.

2) **CNN**: $$\mathcal{X}$$을 CNN에 통과시켜 특징맵 추출.
- ResNet50의 경우 $$32$$로 $$\mathcal{X}$$의 높이$$(H)$$와 너비$$(W)$$를 downsampling하고 2048의 채널을 리턴하게 됩니다.

3) **1d conv**: Transformer 출력으로 활용할 수 있도록 1d conv에 통과시켜 $$(H \times W, d)$$ 차원으로 만듭니다.
- $$d$$: embedding dimension.

4) **Flatten**: $$d$$개의 채널에 대해 channel-wise하게 픽셀단위로 flatten시켜서 Transformer 입력 시퀀스를 만듭니다.

## 2) Transformer Encoder
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/14330c56-d5d7-424e-9567-670d05cec198)

Encoder는 입력으로 들어온 시퀀스들에 positional encoding을 적용하여 최종 input embeddings를 만듭니다.

이후, Encoder의 self-attention에서는 입력 시퀀스 간의 **상호 연관성**과 **위치 정보에 대한 문맥 정보** 이해하여 객체를 구분합니다.
- **상호 연관성**:
    - 강아지 사진에서 고차원 층에서의 하나의 특징 벡터는 예를 들어 강아지의 눈에 해당할 수 있습니다 (저차원 층에서는 선과 같은 간단한 특징). 이때, Encoder는 다른 특징 벡터와 함께 학습되면서, 강아지의 눈과 다른 특징들 간의 상호 연관성을 파악합니다. 예를 들어, 강아지의 눈, 코, 귀는 모두 강아지라는 클래스 객체를 예측한다는 점에서 모두 연관되어 있음을 학습할 수 있습니다.
- **위치 정보**:
    - Encoder는 이미지 내의 특징들이 위치 정보를 가지고 있다는 것을 인지합니다. 강아지 사진에서, 강아지의 눈이 강아지의 머리 부분에 위치하고, 강아지의 코가 강아지의 얼굴 중앙에 위치하는 것을 학습할 수 있습니다.

다르게 말하면, Encoder의 multi-head self-attention은 이미지의 전체 pixel들 중 같은 객체의 pixel들 사이에 높은 attention score를 부여하여 객체의 존재 여부와 형태를 파악합니다.

## 3) Transformer Decoder
### Object Query
Object query는 **학습 가능한 Positional Encoding**으로써 전체 이미지에 존재하는 각 객체에 대한 위치 정보를 표현하게 됩니다.

    class DETR(nn.Module):
        def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        def forward(self, samples: NestedTensor):
            self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

DETR 모델에서는 안전하게 COCO 데이터셋에서 한 이미지 당 존재하는 가장 많은 객체의 개수 보다 큰 값인 100개의 object query를 디코더에서 사용합니다 (`num_queries` $$=100$$).

상기 코드에서 nn.Embedding으로 랜덤으로 초기화된 `query_embed`가 객체쿼리에 대한 학습 가능한 Embedding을 의미하게 되며, forward 함수에서 self.transformer를 invoke할 때, 실제 위치 인코딩 값을 담고 있는 `query_embed.weight`를 넘겨주게 됩니다.

    class Transformer(nn.Module):
        def forward(self, src, mask, query_embed, pos_embed):
            tgt = torch.zeros_like(query_embed)

    class TransformerDecoder(nn.Module):
    def forward(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
            for layer in self.layers:
                output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos)

상기 Transformer 클래스의 forward 함수에서 쿼리 `tgt`를 $$0$$으로 초기화합니다.

인코더를 거친 후, 디코더의 각 레이어를 iterate하게 되는데, 이 때 `query_pos` 값은 `query_embed.weight`라는 해딩 객체 쿼리의 위치 인코딩을 의미하게 됩니다.

    class TransformerDecoderLayer(nn.Module):
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward_post(self, tgt, memory,
                        tgt_mask: Optional[Tensor] = None,
                        memory_mask: Optional[Tensor] = None,
                        tgt_key_padding_mask: Optional[Tensor] = None,
                        memory_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None,
                        query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
            
상기 디코더 레이어 클래스를 보면, forward 함수에서 self.with_pos_embed 함수에서 `tgt` 쿼리에 객체 쿼리의 위치 정보인 `query_pos`를 더하여 임베딩해주는 모습입니다.

### Decoder 구조
![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/14330c56-d5d7-424e-9567-670d05cec198)

Decoder의 기본적인 역할은 (1)query slot의 각 object query가 고유한 객체를 탐지하도록 돕고, (2)인코더의 출력으로 부터 그 객체가 어떤 객체인지를 파악하는 데에 있습니다.
- (1) Decoder Self-Attention.
- (2) Encoder-Decoder Attention.

여기서 **Multi-head self attention**는 모든 objecy query 간의 상호 연관성을 이해하여 각 query가 각자 고유한 객체를 탐지하도록 줍니다.
- object 끼리 비교하는 과정을 거쳐야 서로간의 공통점, 차이점을 비교하면서 해당 object에 대한 이해도를 높일 수 있습니다. 가령, 강아지 클래스 객체 쿼리가 msha에서 강아지 다리가 4개이고, 닭 다리가 2개라는 상호 연관성을 학습하여 강아지라는 객체는 다리가 4개라는 점을 강조하며 강아지 객체를 보다 잘 이해하게 됩니다.        
- 기존 Transformer의 디코더는 Multi-head masked self-attention을 사용하지만, DETR 모델은 입력된 이미지에 동시에 모든 객체의 위치를 예측하기 때문에 **별도의 masking 과정을 필요로 하지 않으며**, **순서 제약이 없고(permutation invariant)**, object query들이 독립적으로 한번에 학습되기 때문에 **non-autoregressive 구조**를 띄고 있습니다.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/daee7e49-fb68-42ae-a97b-95ad0e42d866)

- **Cross attention (Multi-head Encoder-Decoder Attention)**: 인코더로 부터 받은 정보를 바탕으로 객체 간의 global 관계를 학습하여 각 object query가 어떤 객체 정보에 대응되는지 학습합니다. 
    - **Query**: Decoder의 object query.
    - **Key, Value**: Encoder 출력.
- **Multi-head self-attention**: 각 object query 간의 상호 연관성을 학습하여 각자 고유한 객체를 이해합니다.
    - **Query, Key, Value**: Decoder의 object query.

![image](https://github.com/hchoi256/hchoi256.github.io/assets/39285147/049ba931-f2fe-44a2-bbb1-b695901e337a)

가령, 상기 이미지에서 보물지도가 전체 이미지이고, 보물 사냥꾼이 object query라고 해보겠습니다.

1. **Encoder**: 보물지도에서 대략적인 보물의 위치를 Encoder를 통해 도출.
2. **Encoder-Decoder Attention**: 보물 사냥꾼들은 Encoder로 부터 받은 정보를 바탕으로 자신들의 위치에서 가장 가까운 보물에 접근하는 방식으로 학습하게 됩니다.
3. **Decoder Self Attention**: 한 보물 사냥꾼이 한 보물을 취하였을 때, 다른 사냥꾼들은 해당 보물을 더 이상 탐하지 않습니다.
4. 이런 식으로 모든 보물이 발견되었을 때, 아무 보물도 발견하지 못한 사냥꾼들은 **no object** 레이블을 할당받게 됩니다.

## 4) Prediction FFN
FFN은 Transformer의 최종 출력인 object query들이 예측한 클래스와 박스의 위치를 생성합니다.

이때, FFN은 2개의 선형 레이어가 존재하고, 각각 클래스 예측과 박스 회귀를 위한 레이어로 활용됩니다.

1. FFN 시작
2. Linear Layer1(박스위치회귀) 
3. 활성화 함수
4. Linear Layer2(클래스 예측)
5. 최종 바운딩 박스와 클래스 예측 동시 수행

> FFN(Feed-Forward Network)은 일반적으로 신경망 구조에서 사용되는 개념으로, **두 개의 선형 변환(Linear Transformation) 레이어와 활성화 함수(Activation Function)로 구성**됩니다.

> Linear Layer는 FFN의 한 종류로, 선형 변환(Linear Transformation)만 수행하는 레이어입니다. Linear Layer는 입력 벡터와 가중치 행렬 간의 행렬 곱셈 연산을 수행한 후, 편향(bias)을 더하고 활성화 함수를 적용하지 않고 그대로 출력합니다.

## Auxiliary decoding losses
기존 DETR은 충분하다고 생각되는 100개의 임의의 object query를 활용하여 디코더의 입력으로 활용하게 됩니다.

이렇게 하면 일부 query는 실제로 필요하지 않을 수 있으며, 불필요한 학습 시간을 소모하게 됩니다. 이러한 불필요한 query들이 학습에 영향을 주어 학습 속도를 느리게 만들 수 있습니다

논문에서는 이러한 문제를 해결하기 위해 다음과 같은 방법들을 제안합니다:
- **새로운 prediction FFN 레이어 추가**: 각 decoder layer에 새로운 prediction Feed-Forward Network (FFN) 레이어를 추가합니다. 이 레이어는 해당 decoder layer에 있는 object query의 개수를 예측하는데 사용됩니다. 이렇게 예측된 개수만큼만 object query가 해당 decoder layer로 전달됩니다. 즉, 실제로 필요한 query만 선택하여 불필요한 query들을 제거하고 학습 시간을 줄일 수 있습니다.
- **Hungarian loss 사용**: 새로운 prediction FFN 레이어를 통해 예측된 query 개수와 실제 필요한 query 개수 사이의 차이에 대한 손실 함수로 Hungarian loss를 사용합니다. Hungarian loss는 예측과 실제 간의 최소 할당 비용을 찾아내는 방식으로, 최적화된 할당 결과를 얻을 수 있도록 도와줍니다.
- **Layer norm 추가**: 서로 다른 decoder layer들의 입력을 정규화하기 위해 Layer normalization을 추가합니다. 이를 통해 학습이 안정화되고 모델의 성능이 개선됩니다.

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

[Object Query](https://herbwood.tistory.com/26)