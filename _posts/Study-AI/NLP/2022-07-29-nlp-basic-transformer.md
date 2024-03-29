---
layout: single
title: "[개발] PART 6: Transformer - GPT-2"
categories: NLP
tag: [NLP, GPT, Transformer]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/nlp-thumbnail.jpg
sidebar:
    nav: "docs"
---

# Transformer w/ Tensorflow
**번역, 질문 답변, 텍스트 요약** 등과 같은 작업을 위해 **순차적으로** 데이터를 처리하도록 설계된 2017년에 구글이 소개한 딥러닝 아키텍처이다. <span style="color: yellow"> It is a deep learning architecture introduced by Google in 2017 designed to process data sequentially for tasks such as **translation, question answering, text summarization**, and more.</span>

텍스트 생성기로 *GPT-2, GPT-3* 등이 있는데, *GPT-2*의 텍스트 생성 기능은 대부분이 사용할 수 있는 가장 인기 있는 transformer 아키텍처 중에 하나이다. <span style="color: yellow"> As text generators, there are *GPT-2* and *GPT-3*, and the text generation function of *GPT-2* is one of the most popular transformer architectures that most of them can use. </span>

> 단어들은 불연속적 단위이기 때문에 각각에 대힌 확률값을 얻을 수 있다. 여기서, '언어 모델'은 단어 시퀀스에 확률을 할당하는 모델입니다. <span style="color: yellow"> Since words are discrete units, we can obtain a probability value for each. Here, a 'language model' is a model that assigns probabilities to word sequences.</span>

# GPT-2 (Generative Pre-training Transformer 2)
WebText라 불리는 40GB 크기의 거대한 코퍼스에다가 인터넷에서 크롤링한 데이터를 합쳐서 훈련시킨 자귀 회귀 언어 모델이며, 이 모델은 디코더 스택만 사용하는 Attention를 활용한다 (BERT는 인코더 스택만 사용한다). <span style="color: yellow"> It is a **auto-regressive* language model trained by combining data crawled from the Internet in a huge 40GB corpus called WebText, and this language model utilizes Attention using only the decoder stack (BERT uses only the encoder stack).</span>
- **Generative**: 한 단어(토큰)가 들어오면 다음에 올 적절한 토큰을 생성하는 언어 모델이다. <span style="color: yellow"> It is a language model that, when a word (token) comes in, generates the appropriate token to follow.</span>
    - 예를들어, "오늘" 이라는 단어가 GPT 모델에 Input으로 들어가면, GPT는 "날씨가"  같은 뒤에 올 적절한 단어를 Output으로 내보낸다. <span style="color: yellow"> For example, if the word "today" is input to the GPT model, GPT outputs the appropriate word followed by something like "weather" as output.</span>
- **Pre-trained**: 말뭉치 (Corpus) 만을 가지고 *사전 학습*한다. <span style="color: yellow"> *pre-learning* with only the corpus.</span>
    - Encoding 과 Decoding 의 과정이 필요하지 않는다. <span style="color: yellow"> Encoding and decoding processes are not required.</span>

> **자기회귀 모델(auto-regressive model)**: 이전의 출력이 다음의 입력이 되는 모델을 의미한다. <span style="color: yellow">  A model in which the previous output becomes the next input.</span>

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
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large") # 문장 생성 sentence generation
```

```python
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id = tokenizer.eos_token_id) # EOS토큰을 PAD토큰으로 지정하여 warning이 나오지 않게 한다 Designating EOS tokens as PAD tokens to avoid warnings
```

상기 코드에서 보는 것처럼 GPT는 사전 훈련 기반 모델이며, fine-tuning을 거치지 않는다. <span style="color: yellow"> As shown in the code above, GPT is a pre-training-based model and does not undergo fine-tuning.</span>

> **eos_token_id**: The end of sequence token.

```python
SEED = 34
import tensorflow as tf
tf.random.set_seed(SEED)
```

## Encoding
input_sequence --- (encode) ---> tensor --- (decode) ---> greedy_output

```python
MAX_LEN = 70 
input_sequence = "There are times when we are really tired of people but we feel lonely too" # input sample
input_ids = tokenizer.encode(input_sequence, return_tensors="tf") # encoding with tensor as output
input_ids
```


        <tf.Tensor: shape=(1, 15), dtype=int32, numpy=
        array([[ 1858,   389,  1661,   618,   356,   389,  1107, 10032,   286,
                661,   475,   356,  1254, 21757,  1165]])>


## Decoding

```python
sample_outputs = GPT2.generate(input_ids, max_length = MAX_LEN, 
do_sample = True, top_k = 50, top_p = 0.85, num_return_sequences = 5)
```

**generate()**
- *max_length*: 출력 문자열이 가질 수 있는 단어의 최대 갯수 <span style="color: yellow"> Max number of words the output string can have</span>
- *do_sample*: activate sampling
- *top_k*: sampling only from the most likely k words
- *top_p*: Top-p sampling chooses from the smallest possible set of words whose cumulative probability exceeds the probability p
- *num_return_sequences*: 출력 개수 <span style="color: yellow"> # outputs</span>


```python
print("Output:\n" + 100 * "-")

for i, beam_output in enumerate(sample_outputs):
    print("{}: {}.".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
```


        Output:
        ----------------------------------------------------------------------------------------------------
        0: There are times when we are really tired of people but we feel lonely too. There are times when we're not sure what to do and want to leave but there is a lot of pressure. There are times when we're frustrated and don't know what to do. There are times when we don't know if we're right or wrong. There.
        1: There are times when we are really tired of people but we feel lonely too," he said.

        He has no doubt that the Chinese government will try to silence the opposition in any way possible.

        "The Chinese government is going to try to silence us with the use of the law, that's my feeling," he said.

        .
        2: There are times when we are really tired of people but we feel lonely too, which makes me think that our loneliness is a normal part of life," he said. "I think it's the fact that we live in a country where everyone can easily get into the spotlight or be in the news."

        Szewczyk said he found.
        3: There are times when we are really tired of people but we feel lonely too. In those times we can get along well together. If we are going to work, we can work together, if we are going out to do things, we can do things together. We can work together to find some peace, some happiness in our lives, and it.
        4: There are times when we are really tired of people but we feel lonely too.

        The first thing that happens is when you're in your 20s, I would think to myself, "I just want to go to the pub".

        I'll walk to the train station and I'll just think, "I want to go to the.


상기 결과에서 볼 수 있듯이, 하나의 문장 인풋으로 GPT-2가 유사도가 높은 여러 개의 문장을 생성해냈다. <span style="color: yellow"> As can be seen from the above results, with one sentence input, GPT-2 generated several sentences with high similarity.</span>

생성된 각 문장은 우리가 지정한 아키텍쳐과 맞아 떨어진다; 문장 길이 최대 70자 이내. <span style="color: yellow"> Each generated statement matches the architecture we specified; Sentence length up to 70 characters.</span>

우리 아키텍쳐가 지침하는대로, 이러한 문장들은 인풋 텍스트와 유사도가 가장 높은 문장으로 선별한 50개 중 최대 5개를 추출하여 보여준다. <span style="color: yellow"> As our architecture guides, these sentences extract and display up to 5 out of 50 sentences with the highest similarity to the input text.</span>

> **Beam Search**: 매번 선택하는 단어의 갯수로, 선택은 확률 값이 높은 순서대로 한다. num_beams가 2인 경우, 다음 2가지 확률값이 높은 단어에 대해서 탐색한다. 단어 생성에서 가능성이 더 높은 다음 예측 단어를 놓치는 Greedy 방식의 단점을 보완하고자 고안되었다. <span style="color: yellow"> With the number of words selected each time, the selection is made in order of highest probability value. When num_beams is 2, the following two high probability words are searched for. It was designed to compensate for the disadvantage of the Greedy method, which misses the next more likely predictive word in word generation.</span>

# [BERT](https://hchoi256.github.io/nlp/bert-1/)
