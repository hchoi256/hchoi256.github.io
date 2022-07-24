---
layout: single
title: "ML Project 5: 자연어 처리"
categories: ML
tag: [machine learning, python]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/ml-thumbnail.jpg
sidebar:
    nav: "docs"
---

# 코드
**[Notice]** [download here](https://github.com/hchoi256/machine-learning-development)
{: .notice--danger}

# PART 1: 이메일 스팸 필터
이 프로젝트는 [Naive Beyas Classifier](https://github.com/hchoi256/ai-terms/blob/main/README.md)를 활용하여 SMS 스팸 분류 방법을 제시한다.

## Learning Goals
1. **나이브 베이즈 정리**에 대한 이해
- *베이즈 정리*에 기반한 분류 기술

2. 자연어 처리의 기초
- 토큰화 with *NLTK*
- 특징 추출 with *Count Vectorizer*

3. 우도, 사전 확률, 주변 우도의 차이점

4. 불균형 데이터 처리 방법

## 데이터 불러오기

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```

```python
spam_df = pd.read_csv("emails.csv") # 댓글로 요청시 공유해드립니다~
spam_df.head(10)
```

![image](https://user-images.githubusercontent.com/39285147/180648071-f7e5cc4d-ef12-457b-8777-bc7419665b16.png)

## 데이터 시각화

```python
# spam 0 or 1 구분
ham = spam_df[spam_df['spam']==0]
spam = spam_df[spam_df['spam']==1]

# 분포 시각화
spam['length'].plot(bins=60, kind='hist') 
print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%") # 스팸 비율

ham['length'].plot(bins=60, kind='hist') 
print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%") # 햄 비율
```

[*spam*]

![image](https://user-images.githubusercontent.com/39285147/180648230-596590f7-e756-42a4-83a0-92f6bc7cce72.png)


        Spam percentage = 23.88268156424581 %


[*ham*]

![image](https://user-images.githubusercontent.com/39285147/180648232-5b6010a1-797f-4da6-b2c0-d0487d4c1f9c.png)


        Ham percentage = 76.11731843575419 %


```python
sns.countplot(spam_df['spam'], label = "Count") # 각 카테고리 값별로 데이터가 얼마나 있는지 표시
```

![image](https://user-images.githubusercontent.com/39285147/180648340-8fab09eb-0467-4352-9a2e-45bc857caf33.png)


## 자연어 처리

### 데이터 전처리
#### 불용어 제거 with 'Stopwords' + 소문자 통일하기

```python
from nltk.corpus import stopwords
stopwords.words('english') # 불용어 리스트 불러오기

sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']
```


        ['i',
        'me',
        'my',
        'myself',
        'we',
        'our',
        'ours',
        'ourselves',
        'you',
        "you're",
        "you've",
        "you'll",
        "you'd",
        'your',
        'yours',
        'yourself',
        'yourselves',
        'he',
        'him',
        'his',
        'himself',
        'she',
        "she's",
        'her',
        'hers',
        ...
        "weren't",
        'won',
        "won't",
        'wouldn',
        "wouldn't"]


```python
sample_data = [word.lower() for word in sample_data.split() if word.lower() not in stopwords.words('english')] # 불용어에 속하는 단어 삭제하고 소문자로 리턴한다
```

#### 구두점 삭제
```python
import string
string.punctuation
```


        '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'



```python
sample_data = [char for char in sample_data if char not in string.punctuation]
```


### Count Vectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer() # 텍스트를 숫자로 변환
X = vectorizer.fit_transform(sample_data)

print(vectorizer.get_feature_names())

```

        ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']


```python
print(X.toarray())  
```


        [[0 1 1 1 0 0 1 0 1]
        [0 2 0 1 0 1 1 0 1]
        [1 0 0 1 1 0 1 1 1]
        [0 1 1 1 0 0 1 0 1]]


보시는 것처럼 샘플에 존재하는 각 단어들이 인덱스로써 할당하는 *인코딩* 과정이 수행되었습니다.

무슨 말이냐면, '[0 1 1 1 0 0 1 0 1]'이라는 숫자화된 값은 하기처럼 매칭됩니다:
- *this: 0, is: 1, the: 1, first: 1, second: 0, third: 0, document: 1, and: 0, one: 0*


```python
# Vectorizer를 Message List에 적용하기
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = message_cleaning)
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])

print(spamham_countvectorizer.toarray()) 
```


        [[0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        ...
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]]
        

```python
spamham_countvectorizer.shape
```


        (5728, 37229)



## Training/Test 데이터 생성
```python
from sklearn.model_selection import train_test_split
label = spam_df['spam'].values # 스팸 index (0 or 1) 가져오기
X_train, X_test, y_train, y_test = train_test_split(spamham_countvectorizer, label, test_size=0.2)
```


```python
from sklearn.naive_bayes import MultinomialNB # 다항 나이브 베이즈 분류기

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)# 분류기 학습하기
```


```python
# 테스트 샘플 생성
test_countvectorizer = vectorizer.transform(X_test) # 샘플 벡터화 작업 수행
test_predict = NB_classifier.predict(test_countvectorizer) # 샘플에 분류기 적용
```

만약, 메시지 내용의 spam 인자값이 1이라면, NBClassifier에 의해 스팸 메시지로 분류된 것이다.


## 모델 평가: 성능지표/혼동행렬

### 혼동 행렬 (Confusion Matrix)
```python
from sklearn.metrics import classification_report, confusion_matrix

```
```python
# Confusion Matrix
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)
```


![image](https://user-images.githubusercontent.com/39285147/180649439-46c5a9af-0ef0-4e10-80ca-41d6e978da11.png)


14(FN + FP)개의 오분류만 존재하고, 나머지는 전부 옳게 분류한 것을 볼 수 있다.

하지만, 이 결과값은 어디까지나 학습 데이터 훈련 성능이므로 테스트셋에서는 역전될 수 있으니 절대적 신뢰는 금물이다.

```python
# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```


![image](https://user-images.githubusercontent.com/39285147/180649505-8fd871fa-cd4b-4ae2-be89-b85f67a7a0b7.png)


다행히 테스트 데이터에서도 비슷한 성능 평가를 보여준다. 12개의 오분류만 존재하는 것을 확인해볼 수 있다.

하지만, 테스트셋과 training 데이터의 비율 분포에 주목하고, 절대적인 수치에 의존하여 성능 평가를 하면 안된다.

### 성능 지표 (Classification Report)

```python
print(classification_report(y_test, y_predict_test))
```


                    precision    recall  f1-score   support

                0       1.00      0.99      0.99       881
                1       0.96      0.99      0.98       265

        avg / total       0.99      0.99      0.99      1146



# PART 2: YELP 후기





