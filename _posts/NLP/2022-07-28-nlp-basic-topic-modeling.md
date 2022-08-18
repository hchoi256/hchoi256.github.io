---
layout: single
title: "NLP - Part 5: Topic Modeling"
categories: NLP
tag: [NLP, Topic Modeling]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/nlp-thumbnail.jpg
sidebar:
    nav: "docs"
---

# What is Topic Modeling?
주제 모델링(Topic Modeling)이란 주제에 해당하는 텍스트 데이터의 패턴을 식별하는 과정이다. <span style="color: blue"> Topic Modeling is the process of identifying a pattern of text data corresponding to a topic. </span>

텍스트에 여러 주제가 포함된 경우 이 기술을 사용하여 입력 텍스트 내에서 해당 주제를 식별하고 분리할 수 있다. <span style="color: blue"> If the text contains multiple subjects, this technique can be used to identify and isolate those subjects within the input text. </span>

이 기술은 주어진 문서 셋에서 숨겨진 주제를 찾는데 사용할 수도 있다. <span style="color: blue"> This technique can also be used to find hidden topics in a given set of documents. </span>

## Chracteristics of Topic Modeling
레이블이 지정된 데이터가 필요하지 않다 (비지도학습) <span style="color: blue"> No labeling (Unsupervised Learning) </span>

인터넷에서 생성되는 방대한 양의 텍스트 데이터를 감안할 때 토픽 모델링은 방대한 양을 데이터를 요약할 수 있기 때문에 중요하다. <span style="color: blue"> Topic Modeling can summarize tons of text data springing up online.  </span>


# Loading the libraries
```python
import nltk
from nltk.tokenize import RegexpTokenizer # Vectorization
from nltk.corpus import stopwords # 불용어 stopwords
from nltk.stem.snowball import SnowballStemmer # 어간 추출 Stemmer
from gensim import models, corpora
```

# Loading the dataset
```python
def load_data(input_file):
    data = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            data.append(line[:-1])
    return data
```

```python
load_data("7.1 data.txt")
```

                ['The Roman empire expanded very rapidly and it was the biggest empire in the world for a long time.',
                'An algebraic structure is a set with one or more finitary operations defined on it that satisfies a list of axioms.',
                'Renaissance started as a cultural movement in Italy in the Late Medieval period and later spread to the rest of Europe.',
                'The line of demarcation between prehistoric and historical times is crossed when people cease to live only in the present.',
                'Mathematicians seek out patterns and use them to formulate new conjectures.  ',
                'A notational symbol that represents a number is called a numeral in mathematics. ',
                'The process of extracting the underlying essence of a mathematical concept is called abstraction.',
                'Historically, people have frequently waged wars against each other in order to expand their empires.',
                'Ancient history indicates that various outside influences have helped formulate the culture and traditions of Eastern Europe.',
                'Mappings between sets which preserve structures are of special interest in many fields of mathematics.']


# Tokenization

```python
def process(input_text):
    tokenizer = RegexpTokenizer(r"\w+")
    stemmer = SnowballStemmer("english")
    stop_words = stopwords.words("english")

    tokens = tokenizer.tokenize(input_text.lower())
    tokens = [ x for x in tokens if not x in stop_words ]
    tokens_stemmed = [ stemmer.stem(x) for x in tokens ]

    return tokens_stemmed
```

```python
tokens = [ process(x) for x in data ]
print(tokens)
```


                [['roman', 'empir', 'expand', 'rapid', 'biggest', 'empir', 'world', 'long', 'time'], ['algebra', 'structur', 'set', 'one', 'finitari', 'oper', 'defin', 'satisfi', 'list', 'axiom'], ['renaiss', 'start', 'cultur', 'movement', 'itali', 'late', 'mediev', 'period', 'later', 'spread', 'rest', 'europ'], ['line', 'demarc', 'prehistor', 'histor', 'time', 'cross', 'peopl', 'ceas', 'live', 'present'], ['mathematician', 'seek', 'pattern', 'use', 'formul', 'new', 'conjectur'], ['notat', 'symbol', 'repres', 'number', 'call', 'numer', 'mathemat'], ['process', 'extract', 'under', 'essenc', 'mathemat', 'concept', 'call', 'abstract'], ['histor', 'peopl', 'frequent', 'wage', 'war', 'order', 'expand', 'empir'], ['ancient', 'histori', 'indic', 'various', 'outsid', 'influenc', 'help', 'formul', 'cultur', 'tradit', 'eastern', 'europ'], ['map', 'set', 'preserv', 'structur', 'special', 'interest', 'mani', 'field', 'mathemat']]


상기 데이터가 2차원 배열로써 존재하기 때문에, 딕셔너리를 활용하여 이용하기 용이하게 변환해준다. <span style="color: blue"> Since the data exists as a two-dimensional array, we are using a dictionary to convert the data for easy use. </span>

```python
dict_tokens = corpora.Dictionary(tokens)

for token in tokens:
    print(token)
```


                ['roman', 'empir', 'expand', 'rapid', 'biggest', 'empir', 'world', 'long', 'time']
                ['algebra', 'structur', 'set', 'one', 'finitari', 'oper', 'defin', 'satisfi', 'list', 'axiom']
                ['renaiss', 'start', 'cultur', 'movement', 'itali', 'late', 'mediev', 'period', 'later', 'spread', 'rest', 'europ']
                ['line', 'demarc', 'prehistor', 'histor', 'time', 'cross', 'peopl', 'ceas', 'live', 'present']
                ['mathematician', 'seek', 'pattern', 'use', 'formul', 'new', 'conjectur']
                ['notat', 'symbol', 'repres', 'number', 'call', 'numer', 'mathemat']
                ['process', 'extract', 'under', 'essenc', 'mathemat', 'concept', 'call', 'abstract']
                ['histor', 'peopl', 'frequent', 'wage', 'war', 'order', 'expand', 'empir']
                ['ancient', 'histori', 'indic', 'various', 'outsid', 'influenc', 'help', 'formul', 'cultur', 'tradit', 'eastern', 'europ']
                ['map', 'set', 'preserv', 'structur', 'special', 'interest', 'mani', 'field', 'mathemat']

# Vectorization: Bag of Words

```python
doc_term_mat = [ dict_tokens.doc2bow(token) for token in tokens ]
print(doc_term_mat)
```


                [[(0, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1)], [(18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1)], [(6, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1), (35, 1), (36, 1), (37, 1), (38, 1)], [(39, 1), (40, 1), (41, 1), (42, 1), (43, 1), (44, 1), (45, 1)], [(46, 1), (47, 1), (48, 1), (49, 1), (50, 1), (51, 1), (52, 1)], [(46, 1), (47, 1), (53, 1), (54, 1), (55, 1), (56, 1), (57, 1), (58, 1)], [(1, 1), (2, 1), (33, 1), (36, 1), (59, 1), (60, 1), (61, 1), (62, 1)], [(18, 1), (19, 1), (40, 1), (63, 1), (64, 1), (65, 1), (66, 1), (67, 1), (68, 1), (69, 1), (70, 1), (71, 1)], [(16, 1), (17, 1), (47, 1), (72, 1), (73, 1), (74, 1), (75, 1), (76, 1), (77, 1)]]


# Latent Dirichlet Allocation
```python
num_topics = 2

# Latent Dirichlet Allocation
ldamodel = models.ldamodel.LdaModel(doc_term_mat, num_topics=num_topics, id2word=dict_tokens, passes=25)
```

> **잠재 디리클레 할당(Latent Dirichlet allocation, LDA)**: 주어진 문서에 대하여 각 문서에 어떤 주제들이 존재하는지를 서술하는 대한 확률적 토픽 모델 기법 중 하나이다. <span style="color: blue"> It is one of the probabilistic topic model techniques for describing which topics exist in each document for a given document. </span>


```python
num_words = 5 # 각 주제에 대한 기여하는 상위 5개 단어 출력 the top 5 contributing words to each topic

print("Top " + str(num_words) + " contributing words to each topic.")
for item in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
    print("Topic : ", item[0])
    
    list_of_strings = item[1].split(" + ")
    for text in list_of_strings:
        weight = text.split("*")[0]
        word = text.split("*")[1]
        print(word, "==>", str(round(float(weight) * 100, 2)) + '%')
    print()
```

                Top 5 contributing words to each topic.
                Topic :  0
                "empir" ==> 3.4%
                "cultur" ==> 2.4%
                "europ" ==> 2.4%
                "time" ==> 2.4%
                "histor" ==> 2.4%

                Topic :  1
                "mathemat" ==> 4.9%
                "structur" ==> 3.7%
                "set" ==> 3.7%
                "call" ==> 3.1%
                "one" ==> 2.2%


> [LDA](https://wikidocs.net/30708)