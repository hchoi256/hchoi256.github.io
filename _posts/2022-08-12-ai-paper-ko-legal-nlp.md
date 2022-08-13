---
layout: single
title: "[논문 분석] A Multi-Task Benchmark for Korean Legal Language Understanding and Judgement Prediction (arXiv 2022)"
categories: AI
tag: [AI, dissertation, Research, 논문, kaist]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/ai-thumbnail.jpg
sidebar:
    nav: "docs"
---

![image](https://user-images.githubusercontent.com/39285147/184294516-0dad2074-9c87-44f1-af52-6ea4c0b5f174.png)

[**논문**](https://arxiv.org/abs/2206.05224)

# 들어가면서
법률 계약서는 일반인이 독해하기 어려운 단어들 뿐만 아니라, 한 문장이 한 페이지를 차지할 정도로 긴 문장들을 포함한다.

이전에 참가한 한 세미나에서는 이러한 법률 계약서에 존재하는 오류를 검사하는 AI 모델을 실제 변호사와 대결시킨 사례를 소개했다.

AI 모델은 **26초 만에 94%의 정확도**로 오류를 검증해내었고, 반면 사람 변호사는 **96분 동안 86%의 정확도**로 오류를 잡아냈다.

이러한 압도적인 AI의 성능은 매번 적응되지 않을 정도로 정말이지 어마무시하다.

하지만, 상기 법률 계약서의 문서 내용은 영어로 적혀있었다. 다시 말해, 자연어 처리 분야에서 기계독해가 까다로운 **한국어로 적힌 법률 계약서**는 아직까진 그만한 성능을 내는 AI 모델이 전무하다.

상기 목표를 달성하기 위해서는 다양한 조건이 수반되어야 할터인데, 가령 한국어 기반 모델 검증을 위한 **법률 평가 데이터셋**이 필요할 것이고, 까다로운 한국어 문법을 정확히 이해하는 성능좋은 언어모델이 필요하다.

여기까지는 내 잡담이었으니 가볍게 무시해도 상관없다.

하지만, 앞서 언급된 사례는 다양한 전문 분야에서 **한국어 기반 NLP 연구**의 필요성을 부각한다는 점은 인지하자.

# INTRODUCTION
[*KAIST AI Grad School, located in Seoul*]

![image](https://user-images.githubusercontent.com/39285147/184298843-e3edab46-0d27-4a74-a1f1-62762d1a794a.png)

정부에서 출시한 규칙 및 도메인 지식에 근거한 기존 법률 시스템은 유의미한 성과를 내기도 했으나, **범용성이 부족**하다는 한계점이 존재했다.

이러한 실패에는 해당 전문 분야에 대한 **한국어 NLP 데이터셋 부재**와 NLP 학습에 **까다로운 한국어 특성**이 내재되어 있을 것이다.

하지만, Deep Learning에 발전에 따라 자연어 처리 또한 많은 변화의 시기를 거쳐서 '판결 예측'과 같은 여러 법률 분야에 새로운 기술적 패러다임을 제시한다.

이러한 격동의 시기에 발맞춰 전세계에서는 AI 모델 학습에 필수적인 법률 데이터셋을 만들기 시작했고, 서교수 연구팀 또한 기존에 정부에서 내놓은 활용 가치가 떨어졌던 한국어 법률 데이터셋로부터 더 확장하여 새로운 한국어 기반 법률 데이터셋 연구 개발에 임하게 되었다.

서교수를 주축으로한 연구팀은 '최초로' **한국어 기반 대용량 법률 AI 데이터셋**과 **'LBOX OPEN'**이라는 법률 평가 데이터셋, 그리고 **LCUBE라는 한국어 법률 언어 모델**을 만들게 되었다 (*LBOX OPEN과 LCUBE는 해당 논문 페이지를 통해 다운받아 이용 가능하다*).

LBOX OPEN은 *1개의 법률 corpus*, *두 개의 분류 문제*, *두 개의 법률 판단 예측 문제*, 그리고 *한 개의 요약 문제*를 위한 평가 데이터셋이다.

> 범죄와 같이 어떠한 종류의 사례들을 기반으로 구성된 데이터셋인지, 언제/어떻게 데이터를 수집했는지 등 보다 자세한 내용은 해당 논문에서 확인하길 바란다.

# *LBOX OPEN* - Large-scale Korean legal AI benchmark.

LBOX OPEN의 구성은 다음과 같다:
- (1) **a large-scale legal precedent corpus** (PRECEDENT CORPUS)
- (2) **two classification tasks** (CASE NAME, STATUTE)
- (3) **two legal judgement prediction tasks** (LJP-CRIMINAL, LJPCIVIL)
- (4) **one summarization task** (SUMMARIZATION).

> 각 구성 요소에 대한 활용 방벙과 같은 보다 자세한 설명은 해당 논문에 나와있다.

## Data preprocessing
서교수 연구팀은 기존에 정부에서 내놓은 Korean precedents라는 raw data에 포함된 non-trivial 정보들을 자동 parsing하고자 **custom data engineering pipeline**을 만들었다.

해당 Pipeline에서 여러 분류 작업을 거친다.
- *RestNet*을 기반 *Layout classifier*를 활용하여 각 페이지를 *'text only'* 혹은 *'text w/ tbl or pictures'*로 분류하였다.
- *Mask-R-CNN*의 *Layout parser*를 사용하여 non-textual 요소들을 페이지로부터 분리한다.
- Save the information to the database:
  - If pdf, extract text w/ custom rule-based parser
  - Otherwise if images, extract text segments and their coordinates w/ proprietary OCR engine and use language model to corret OCR errors. 

상기 과정을 거친 후 **confidence score**를 계산해 기준치보다 낮을 경우, 해당 페이지들은 연구팀이 직접 수동으로 처리한다.

마지막으로, 하기 정보를 JSON 형식으로 저장하고 최종 output을 도출한다:
- (1) meta information such as case name, sentencing dates, and names of attendees
- (2) ruling
- (3) gist of claim
- (4) appeal
- (5) reasoning body that consists of facts, claims, reasoning, and decision of judges.

# *LCUBE* - Language model based on LBOX OPEN
**Classification tasks**
- GPT-2 활용 decoder-only 모델 --> comparable performance with MT5 (a competitive encoder-decoder language model with larger size)

**Summarization tasks**
- 타모델에 비해 상대적으로 안좋은 성능을 보인다.

# EXPERIMENTS
CASE NAME, STATUTE, LJP-CRIMINAL, LJP-CIVIL, 그리고 SUMMARIZATION tasks에 대하여 실험 평가를 진행한다.

## Model training
서교수님의 LK 연구소는 한정된 예산으로 GPU와 같은 하드웨어를 따로 소유하는 것이 아니라 클라우드를 대여하는 연구 방식으로 유명하다.

이번 프로젝트에서 역시 클라우드에서 Nvidia RTX3090/RTX6000를 대여하여 모델 학습을 진행했다.

다음은 모델 학습에 사용된 parameter settings이다:
- **learning rate**: 0.00003-0.0001
- **batch**: 8-16
- **optimizer**: AdamW
- **fine-tuning**: MT5-small (checkpoint: *'google/mt5-small'*)
- **accracy**: *F*1
- etc.

> 보다 상세한 spec은 해당 논문을 참조하길 바란다.

## Metric
- (1) the case is counted as a true positive if their values are equal
- (2) false positive f their values are not equal
- (3) When the target field exists only in GT but not in the prediction, the case is counted as a false negative.
- (4) If the target field is empty in the GT but exists in the prediction it is counted as a false positive. 
- (5) If the field is empty in both GT and the prediction, the case is considered as a truenegative.

> The zero labels in LJP-CRIMINAL are counted as nulls.

## Results
![image](https://user-images.githubusercontent.com/39285147/184300687-c0aa8c4c-5d9c-4156-b674-737bac61bbd8.png)

**LCUBE-base** vs. **KoGPT-2**
- In all tasks except SUMMARIZATION, LCUBE-base shows a higher performance

# LIMITATIONS
해당 논문 모든 내용은 한국 법체계의 first level court에 대한 precedents만을 고려한 연구 결과이다. 

또한, 원고, 피고와 같은 요인들을 인풋으로 고려하지 않은 판결 예측 결과라고 가정하며, LBOX OPEN은 legal information retrieval task 등 여러 문제에 대한 사례 정보는 반영하지 않는다.

상기 언급된 한계점들은 시스템적으로 구현이 매우 까다로우며, 도메인 지식이 풍부한 여러 전문가와의 협업이 필요시 될것이며, 이것이 곧 LBOX OPEN의 미래 과제일 것이다.
