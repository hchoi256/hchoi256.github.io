---
layout: single
title: "NLP - Part 6: Transformer"
categories: NLP
tag: [NLP, python]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/nlp-thumbnail.jpg
sidebar:
    nav: "docs"
---

# Transformer w/ Tensorflow
**번역, 질문 답변, 텍스트 요약** 등과 같은 작업을 위해 **순차적으로** 데이터를 처리하도록 설계된 2017년에 구글이 소개한 딥러닝 아키텍처이다. It is a deep learning architecture introduced by Google in 2017 designed to process data sequentially for tasks such as **translation, question answering, text summarization**, and more.

텍스트 생성기로 *GPT-2, GPT-3* 등이 있는데, *GPT-2*의 텍스트 생성 기능은 대부분이 사용할 수 있는 가장 인기 있는 transformer 아키텍처 중에 하나이다. As text generators, there are *GPT-2* and *GPT-3*, and the text generation function of *GPT-2* is one of the most popular transformer architectures that most of them can use.

> 단어들은 불연속적 단위이기 때문에 각각에 대힌 확률값을 얻을 수 있다. 여기서, '언어 모델'은 단어 시퀀스에 확률을 할당하는 모델입니다. Since words are discrete units, we can obtain a probability value for each. Here, a 'language model' is a model that assigns probabilities to word sequences.

# GPT-2 (Generative Pre-training Transformer 2)
WebText라 불리는 40GB 크기의 거대한 코퍼스에다가 인터넷에서 크롤링한 데이터를 합쳐서 훈련시킨 자귀 회귀 언어 모델이며, 이 모델은 디코더 스택만 사용하는 Attention를 활용한다 (BERT는 인코더 스택만 사용한다). It is a **auto-regressive* language model trained by combining data crawled from the Internet in a huge 40GB corpus called WebText, and this language model utilizes Attention using only the decoder stack (BERT uses only the encoder stack).
- **Generative**: 한 단어(토큰)가 들어오면 다음에 올 적절한 토큰을 생성하는 언어 모델이다. It is a language model that, when a word (token) comes in, generates the appropriate token to follow.
    - 예를들어, "오늘" 이라는 단어가 GPT 모델에 Input으로 들어가면, GPT는 "날씨가"  같은 뒤에 올 적절한 단어를 Output으로 내보낸다. For example, if the word "today" is input to the GPT model, GPT outputs the appropriate word followed by something like "weather" as output.
- **Pre-trained**: 말뭉치 (Corpus) 만을 가지고 *사전 학습*한다. *pre-learning* with only the corpus.
    - Encoding 과 Decoding 의 과정이 필요하지 않는다. Encoding and decoding processes are not required.

> **자기회귀 모델(auto-regressive model)**: 이전의 출력이 다음의 입력이 되는 모델을 의미한다.

> [GPT-2](https://hyyoka-ling-nlp.tistory.com/9)

> [BERT vs. GPT-2](https://hyyoka-ling-nlp.tistory.com/8)

```python
!pip install tensorflow
!pip install transformers # 특정 모델을 쉽게 다운로드해서 사용할 수 있다! can easily download and use specific models!
```

## Loading the libraries

```python
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large") # 문장 생성 creating sentences
```

```python

```

```python

```

```python

```

```python

```

```python

```

# BERT w/ PyTorch

```python

```

```python

```

```python

```
