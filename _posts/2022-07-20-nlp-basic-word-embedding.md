---
layout: single
title: "NLP - Part 2: Word Embedding"
categories: NLP
tag: [NLP, python]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
sidebar:
    nav: "docs"
---

# Part 1: Vectorization

## Vectorization 필요성
Machine (기계)는 문자와 단어를 이해할 수 없다.

0과 1로 이루어진 이진 형태의 데이터를 기계는 이해할 수 있다.

**Computer Vision**의 가장 기본은 이미지는 픽셀 (pixel)로 이루어져 있고, 픽셀에 대한 정보는 x, y와 같은 픽셀의 위치 그리고 해당 픽셀의 색상 정보 (보통 RGB)를 가지고 있다. 이런 정보들은 숫자로 쉽게 만들 수가 있다!

**NLP**의 텍스트 데이터 역시 기계가 이해할 수 있도록 숫자로 표현해야 한다.

*CountVectorizer* 텍스트를 숫자 데이터로 변환하는 방법으로, 텍스트를 수치 데이터로 변화하는데 사용하는 method! sklearn을 통해 사용 가능하다.

## Vectorization 구현

```python
%pip install scikit-learn
```

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

text1 ='''Tracy loves writing about data science.'''
text2 = '''Tracy loves writing for medium publications.'''
text3 = '''Tracy loves studying Python programming.'''
text4 = '''Tracy loves entering Kaggle competitions.'''
text5 = '''Tracy loves making YouTube videos.'''
text6 = '''Tracy loves getting new subscribers.'''
corpus = [text1, text2, text3, text4, text5, text6]
count_vec = CountVectorizer()

word_count_vec = count_vec.fit_transform(corpus)
word_count_vec.shape
```


    (6, 21)




```python
print(count_vec.get_feature_names_out())
```

    ['about' 'competitions' 'data' 'entering' 'for' 'getting' 'kaggle' 'loves'
     'making' 'medium' 'new' 'programming' 'publications' 'python' 'science'
     'studying' 'subscribers' 'tracy' 'videos' 'writing' 'youtube']
    


```python
print("Vocabulary: ", count_vec.vocabulary_)
```

    Vocabulary:  {'tracy': 17, 'loves': 7, 'writing': 19, 'about': 0, 'data': 2, 'science': 14, 'for': 4, 'medium': 9, 'publications': 12, 'studying': 15, 'python': 13, 'programming': 11, 'entering': 3, 'kaggle': 6, 'competitions': 1, 'making': 8, 'youtube': 20, 'videos': 18, 'getting': 5, 'new': 10, 'subscribers': 16}
    

```python
print("Encoded Document is: ")
print(word_count_vec.toarray())
```

    Encoded Document is: 
    [[1 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 1 0]
     [0 0 0 0 1 0 0 1 0 1 0 0 1 0 0 0 0 1 0 1 0]
     [0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 1 0 0 0]
     [0 1 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0]
     [0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1]
     [0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 1 0 0 0]]
    


```python
matrix = count_vec.fit_transform(corpus)
pd.set_option("display.max_columns", None) # 테이블 전체를 한 눈에 보여주기
counts = pd.DataFrame(matrix.toarray(), index = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"], columns = count_vec.get_feature_names())
counts.loc["Total", :] = counts.sum()
counts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>about</th>
      <th>competitions</th>
      <th>data</th>
      <th>entering</th>
      <th>for</th>
      <th>getting</th>
      <th>kaggle</th>
      <th>loves</th>
      <th>making</th>
      <th>medium</th>
      <th>new</th>
      <th>programming</th>
      <th>publications</th>
      <th>python</th>
      <th>science</th>
      <th>studying</th>
      <th>subscribers</th>
      <th>tracy</th>
      <th>videos</th>
      <th>writing</th>
      <th>youtube</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>doc1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>doc2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>doc3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>doc4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>doc5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>doc6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>