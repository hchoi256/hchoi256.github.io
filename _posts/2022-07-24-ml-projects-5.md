---
layout: single
title: "ML Project 5: Natural Language Processing"
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

# Code
**[Notice]** [download here](https://github.com/hchoi256/machine-learning-development)
{: .notice--danger}

# Learning Goals
1. **Naive Beyas Theorem**
- *베이즈 정리*에 기반한 분류 기술 Classification techniques with beyas theorem

2. 자연어 처리의 기초 Understanding NLP
- tokenization with *NLTK*
- Extracting features with *Count Vectorizer* 

3. 우도, 사전 확률, 주변 우도의 차이점 Likelihood, prior, marginal likelihood

4. 불균형 데이터 처리 방법 How to handle unbalanced data

# PART 1: Email Spam Filtering
이 프로젝트는 [Naive Beyas Classifier](https://github.com/hchoi256/ai-terms/blob/main/README.md)를 활용하여 SMS 스팸 분류 방법을 제시한다. Presenting the method of SMS Spam Classification using Naive Beyas Classifier

## Loading the dataset

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```

```python
spam_df = pd.read_csv("emails.csv")
spam_df.head(10)
```

![image](https://user-images.githubusercontent.com/39285147/180648071-f7e5cc4d-ef12-457b-8777-bc7419665b16.png)

## Data visualization

```python
# spam 0 or 1
ham = spam_df[spam_df['spam']==0]
spam = spam_df[spam_df['spam']==1]

# Visualizing the distribution 
spam['length'].plot(bins=60, kind='hist') 
print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")

ham['length'].plot(bins=60, kind='hist') 
print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")
```

[*spam*]

![image](https://user-images.githubusercontent.com/39285147/180648230-596590f7-e756-42a4-83a0-92f6bc7cce72.png)


        Spam percentage = 23.88268156424581 %


[*ham*]

![image](https://user-images.githubusercontent.com/39285147/180648232-5b6010a1-797f-4da6-b2c0-d0487d4c1f9c.png)


        Ham percentage = 76.11731843575419 %


```python
sns.countplot(spam_df['spam'], label = "Count") # 각 카테고리 값별로 데이터가 얼마나 있는지 표시 Data distribution by category
```

![image](https://user-images.githubusercontent.com/39285147/180648340-8fab09eb-0467-4352-9a2e-45bc857caf33.png)


## NLP

### Data Preprocessing
#### 불용어 제거 and 소문자 통일하기 Stopwords and lowercase

```python
from nltk.corpus import stopwords
stopwords.words('english') # 불용어 리스트 불러오기 Loading the list of stopwords

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
sample_data = [word.lower() for word in sample_data.split() if word.lower() not in stopwords.words('english')]
```

#### Removing punctuation
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

vectorizer = CountVectorizer() # Text to number
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


샘플에 존재하는 각 단어들이 인덱스로써 할당되는 *인코딩* 과정이 수행되었다. Encoding process is complete.

무슨 말이냐면, '[0 1 1 1 0 0 1 0 1]'이라는 숫자화된 값은 하기처럼 매칭됩니다:
- *this: 0, is: 1, the: 1, first: 1, second: 0, third: 0, document: 1, and: 0, one: 0*


```python
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



## Creating the Training/Test dataset 
```python
from sklearn.model_selection import train_test_split
label = spam_df['spam'].values # Loading the spam index (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(spamham_countvectorizer, label, test_size=0.2)
```


```python
from sklearn.naive_bayes import MultinomialNB # 다항 나이브 베이즈 분류기

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
```


```python
test_countvectorizer = vectorizer.transform(X_test) # 샘플 벡터화 작업 수행
test_predict = NB_classifier.predict(test_countvectorizer) # 샘플에 분류기 적용
```

만약, 메시지 내용의 spam 인자값이 1이라면, NBClassifier에 의해 스팸 메시지로 분류된 것이다.


## Evaluating the model: Confusion Matrix/Classification Report

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


14(FN + FP)개의 오분류만 존재하고, 나머지는 전부 옳게 분류한 것을 볼 수 있다. Training dataset --> # errors = 14(FN + FP)

하지만, 이 결과값은 어디까지나 학습 데이터 훈련 성능이므로 테스트셋에서는 역전될 수 있으니 절대적 신뢰는 금물이다. Cannot trust this result 100% because it is just based on the training dataset.

```python
# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```


![image](https://user-images.githubusercontent.com/39285147/180649505-8fd871fa-cd4b-4ae2-be89-b85f67a7a0b7.png)


다행히 테스트 데이터에서도 비슷한 성능 평가를 보여준다. 12개의 오분류만 존재하는 것을 확인해볼 수 있다. Test dataset --> # errors = 12(FN + FP)

하지만, 테스트셋과 training 데이터의 비율 분포에 주목하고, 절대적인 수치에 의존하여 성능 평가를 하면 안된다. Paying attention to the distribution of the training and test dataset 

### 성능 지표 (Classification Report)

```python
print(classification_report(y_test, y_predict_test))
```


                    precision    recall  f1-score   support

                0       1.00      0.99      0.99       881
                1       0.96      0.99      0.98       265

        avg / total       0.99      0.99      0.99      1146



# PART 2: YELP Reviews
'엘프'는 기업체와 서비스에 대한 대중의 리뷰 포럼을 제공하는 앱으로 자연어 처리 기술을 사용해 엘프 리뷰를 분석해본다. A public review forum for businesses and services, using natural language processing technology to analyze YELP reviews.

자연어 처리를 활용하여 리뷰에 있는 순수 텍스트에 기반해 고객 기분이 좋은지 나쁜지 감성 분석을 시행한 후, 모델이 자동적으로 해당 고객이 상품에 별점을 얼마나 부과할지 예측한다. After sentiment analysis to raw test data using NLP, the model predicts appropriate star rate


## Loading the dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
yelp_df = pd.read_csv("yelp.csv")
```

![image](https://user-images.githubusercontent.com/39285147/180663433-611d058b-ec9e-4276-a517-9b7020c66700.png)


## Data Visualization

```python
yelp_df['length'] = yelp_df['text'].apply(len)
yelp_df['length'].plot(bins=100, kind='hist') 
```

![image](https://user-images.githubusercontent.com/39285147/180663579-4a692ad7-7df5-4913-8a64-d4edd41607ec.png)


대략 400단어가 분포에서 가장 많은 것으로 관찰된다. Approximately, 400 words appear the most in the distribution


```python
yelp_df.length.describe()
```

        count    10000.000000
        mean       710.738700
        std        617.399827
        min          1.000000
        25%        294.000000
        50%        541.500000
        75%        930.000000
        max       4997.000000
        Name: length, dtype: float64


위 결과에서 가장 많은 단어 수는 4997개이고, 평균 단어 수는 710인 것으로 발견됐다. The max number of words = 4997, avg. = 710

그렇다면, 각 별점 별 리뷰 개수 분포는 어떠할까? What about the star distribution?

```python
sns.countplot(y = 'stars', data=yelp_df) # count by column 'stars'
```


![image](https://user-images.githubusercontent.com/39285147/180664615-c1904d4b-a7f2-49b5-b6c1-4c5001687b63.png)


분포도에서 나온 바와 같이 별점 4점이 가장 많은 빈도수를 차지한다. 4 stars appear the most

여기서, 우리는 각 별점 별로 단어 길이 개수 분포를 확인해보고자 한다. What about the length of words by star rate

```python
g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=5) # 한 줄에 총 5개 grid를 별점 기준으로 나누기
g.map(plt.hist, 'length', bins = 20, color = 'r') # 히스토그램 w/ x-axis = length
```

![image](https://user-images.githubusercontent.com/39285147/180664713-5cda0980-4cb4-4f77-b58d-b11f059294d7.png)


분포도에서 확인해볼 수 있듯이, x축은 단어의 길이를 나타낸다. x-axis = length of words

따라서, 길이가 짧은 리뷰의 빈도수가 각 별점마다 높게 나타난 것을 확인해볼 수 있다. The short length of reviews appear the most by each star rate

이제, 별점 1과 5를 비교해보자. What about star rate 1 vs. 5?

```python
yelp_df_1 = yelp_df[yelp_df['stars']==1]
yelp_df_5 = yelp_df[yelp_df['stars']==5]
yelp_df_1_5 = pd.concat([yelp_df_1 , yelp_df_5]) # 한 행렬로 합치기
```


```python
print( '1-Stars percentage =', (len(yelp_df_1) / len(yelp_df_1_5) )*100,"%")
```

        1-Stars percentage = 18.330885952031327 %

별점 1점은 별점 5점에 대비하여 18.3%의 빈도수를 차지한다. star rate 1 appear 18.3% more than 5

따라서, 별점 5점은 81.67%를 차지할 것이다. hence, star rate 5 takes 81.67%

```python
sns.countplot(yelp_df_1_5['stars'], label = "Count") 
```

![image](https://user-images.githubusercontent.com/39285147/180664906-878df245-f501-4236-bde6-903fd5cb8cee.png)


## NLP

### 불용어/구두점 제거 Stopwords/Punctuation

```python
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean
```

<details>
<summary>Result(hide/show)</summary>
<div markdown="1">

```python
yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)
print(yelp_df_clean[0]) # show the cleaned up version
```


        ['wife', 'took', 'birthday', 'breakfast', 'excellent', 'weather', 'perfect', 'made', 'sitting', 'outside', 'overlooking', 'grounds', 'absolute', 'pleasure', 'waitress', 'excellent', 'food', 'arrived', 'quickly', 'semibusy', 'Saturday', 'morning', 'looked', 'like', 'place', 'fills', 'pretty', 'quickly', 'earlier', 'get', 'better', 'favor', 'get', 'Bloody', 'Mary', 'phenomenal', 'simply', 'best', 'Ive', 'ever', 'Im', 'pretty', 'sure', 'use', 'ingredients', 'garden', 'blend', 'fresh', 'order', 'amazing', 'EVERYTHING', 'menu', 'looks', 'excellent', 'white', 'truffle', 'scrambled', 'eggs', 'vegetable', 'skillet', 'tasty', 'delicious', 'came', '2', 'pieces', 'griddled', 'bread', 'amazing', 'absolutely', 'made', 'meal', 'complete', 'best', 'toast', 'Ive', 'ever', 'Anyway', 'cant', 'wait', 'go', 'back']


```python
print(yelp_df_1_5['text'][0]) # show the original version
```


        My wife took me here on my birthday for breakfast and it was excellent.  The weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.  Our waitress was excellent and our food arrived quickly on the semi-busy Saturday morning.  It looked like the place fills up pretty quickly so the earlier you get here the better.

        Do yourself a favor and get their Bloody Mary.  It was phenomenal and simply the best I've ever had.  I'm pretty sure they only use ingredients from their garden and blend them fresh when you order it.  It was amazing.

        While EVERYTHING on the menu looks excellent, I had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious.  It came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete.  It was the best "toast" I've ever had.

        Anyway, I can't wait to go back!


</div>
</details>

### [Count Vectorizer](#count-vectorizer)

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])

yelp_countvectorizer.shape
```

        (4086, 26435)


```python
print(yelp_countvectorizer.toarray())  
```


        [[0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        ...
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]]


## Creating the Training/Test dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(yelp_countvectorizer, yelp_df_1_5['stars'].values, test_size=0.2)
```


## Training the model (Naive Beyas)

```python
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
```

## Evaluating the model
### Confusion Matrix
```python
from sklearn.metrics import classification_report, confusion_matrix

y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)
```


![image](https://user-images.githubusercontent.com/39285147/180665314-052c9c4f-9ee8-4e45-83e8-40498b220715.png)


모델이 학습 데이터를 훈련한 결과를 보여주는 상기 혼동 행렬에서 오분류가 64개인 것으로 확인된다. Training dataset --> # errors = 64 in confusion matrix


```python
# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```

![image](https://user-images.githubusercontent.com/39285147/180665355-5991613d-7965-4802-982c-6913ef276720.png)


반대로 모델이 테스트 데이터를 예측한 결과를 보여주는 상기 혼동 행렬에서 오분류가 67개인 것으로 확인된다. Test dataset --> # errors = 67 in confusion matrix

### 성능 지표 (Classification Report)
```python
print(classification_report(y_test, y_predict_test))
```

                    precision    recall  f1-score   support

                1       0.86      0.68      0.76       156
                5       0.93      0.97      0.95       662

        avg / total       0.92      0.92      0.91       818


## TF-IDF (term frequency-inverse document frequency)
정보 검색과 텍스트 마이닝에서 이용하는 가중치로, 여러 문서로 이루어진 문서군이 있을 때 어떤 단어가 특정 문서 내에서 얼마나 중요한 것인지를 나타내는 통계적 수치이다. A statistical measure that evaluates how relevant a word is to a document in a collection of documents.

가령, 부동산 관련하여 '계약'이라는 단어가 지극히 중요할 것이다. For example, 'contract' is important in terms of 'real estates'

For more information, click [here](https://hchoi256.github.io/nlp/nlp-basic-word-embedding/) or [code](#code)를 참조하길 바란다.