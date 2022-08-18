---
layout: single
title: "NLP: Data Preprocessing"
categories: NLP
tag: [NLP, Data Preprocessing]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
# author_profile: false
header:
    teaser: /assets/images/posts/nlp-thumbnail.jpg
sidebar:
    nav: "docs"
---

# Loading the libraries and dataset

```python
import pandas as pd
import numpy as np
```

```python
raw_data = pd.read_csv("Corona_NLP_train.csv", encoding = "latin-1") # load the dataset

raw_data.shape # check the shape

raw_data.head() # view the dataset briefly
```

```python
raw_data.drop(["UserName", "ScreenName", "Location", "TweetAt"], axis = 1) # remove unnecessary columns

raw_data = raw_data [ ["OriginalTweet", "Sentiment"] ] # load the necessary columns
```

# How to preprocess data for NLP

```python
import nltk
from nltk.tokenize import RegexpTokenizer # tokenizer 
from nltk.corpus import stopwords # 불용어
from nltk.stem.porter import PorterStemmer # 어간추출
```

```python
tokenizer = RegexpTokenizer(r"\w+") # tokenize the text by word

nltk.download("stopwords")
stop_words = stopwords.words("english") # 불용어 데이터 저장

stemmer = PorterStemmer() # 어간추출
```

```python
# create the preprocessing function
def text_preprocess(text):
    text = tokenizer.tokenize(text)
    text = [i.lower() for i in text if i not in stop_words]
    text = [stemmer.stem(i) for i in text]
    return text
```

```python
txt = "@tea_lover: I love tea!!! I drink at least 2 cups of tea everyday #tea"
text_preprocess(txt)
```

        ['tea_lov',
        'i',
        'love',
        'tea',
        'i',
        'drink',
        'least',
        '2',
        'cup',
        'tea',
        'everyday',
        'tea']


# Exercise with real dataset

```python
raw_data
```

![image](https://user-images.githubusercontent.com/39285147/183246219-33876610-98a7-4103-a294-91d69359aeb5.png)


```python
raw_data["OriginalTweet"] = raw_data["OriginalTweet"].apply(lambda x: text_preprocess(x))
```

        0        [menyrbi, phil_gahan, chrisitv, http, co, ifz9...
        1        [advic, talk, neighbour, famili, exchang, phon...
        2        [coronaviru, australia, woolworth, give, elder...
        3        [my, food, stock, one, empti, pleas, panic, th...
        4        [me, readi, go, supermarket, covid19, outbreak...
                                    ...                        
        41152    [airlin, pilot, offer, stock, supermarket, she...
        41153    [respons, complaint, provid, cite, covid, 19, ...
        41154    [you, know, itâ, get, tough, kameronwild, rati...
        41155    [is, wrong, smell, hand, sanit, start, turn, c...
        41156    [tartiicat, well, new, use, rift, s, go, 700, ...
        Name: OriginalTweet, Length: 41157, dtype: object