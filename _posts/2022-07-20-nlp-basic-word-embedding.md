---
layout: single
title: "NLP - Part 2: Word Embedding"
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



# Tokenization 복습


```python
import pandas as pd
import numpy as np
```

```python
# Load data
data = pd.read_csv('IMDB-Dataset.csv')
```

```python
data.shape
```

    (50000, 2)


```python
frame = data[:1000]
frame.head()
```

![image](https://user-images.githubusercontent.com/39285147/181003013-73bfa56f-9a5c-4bb7-9a38-794b1828801a.png)


```python
feature = frame.review
label = frame.sentiment

frame.sentiment = frame.sentiment.replace( {"positive" : 1, "negative" : 0} ) # convert from word to number
```

```python
# load libraries
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger") #
```

```python
# data preprocessing
stop_words = list(set(stopwords.words("english")))
all_words = []

for p in frame.review:
    cleaned = re.sub(r"[^a-zA-Z\s]", "", p) # remove unnecessary words
    tokens = nltk.word_tokenize(cleaned)
    word_length = [ i for i in tokens if 3 <= len(i) <=7 ] # limit by length of tokens
    word_lower = [ i.lower() for i in word_length ] # lowercase tokens
    stopped = [w for w in word_lower if not w in stop_words] # remove stopwords
    all_words.append(stopped)
```

```python
words_fq = nltk.FreqDist(all_words)
words_fq
```

    FreqDist({'giant': 6, 'komodo': 6, 'one': 5, 'dont': 4, 'bullets': 3, 'dragons': 3, 'itbr': 3, 'right': 3, 'animal': 3, 'camera': 3, ...})



```python
# data visualization
import matplotlib.pyplot as plt

words_fq.plot(30, cumulative=False)
plt.show()
```


![image](https://user-images.githubusercontent.com/39285147/181004840-3db47747-7566-43a3-beeb-eb52e2f9f4f6.png)


상기 분포도에서 'giant' 단어가 가장 많이 사용된 것을 확인해볼 수 있다. The plot shows that 'giant' is used the most.

# Vectorization

## Why We Need Vectorization?
Machine (기계)는 문자와 단어를 이해할 수 없다. Machines cannot understand sentences or words. 

0과 1로 이루어진 이진 형태의 데이터를 기계는 이해할 수 있다. But they can understand binary data (i.e., 0101010)  

**Computer Vision**의 가장 기본은 이미지는 픽셀 (pixel)로 이루어져 있고, 픽셀에 대한 정보는 x, y와 같은 픽셀의 위치 그리고 해당 픽셀의 색상 정보 (보통 RGB)를 가지고 있다. 이런 정보들은 숫자로 쉽게 만들 수가 있다! **Computer Vision** is a collection of pixels requiring information about the location of x, y, and RGB. We can easily produce such information with numbers.

**NLP**의 텍스트 데이터 역시 기계가 이해할 수 있도록 숫자로 표현해야 한다. In NLP, text data must also be numbers.

**CountVectorizer** 텍스트를 숫자 데이터로 변환하는 방법으로, 텍스트를 수치 데이터로 변화하는데 사용하는 method! sklearn을 통해 사용 가능하다. **CountVectorizer** is the method to convert text to numbers, which can be achieved with *sklearn*.

## 1. CountVectorizer
텍스트 데이터에서 '횟수'를 기준으로 특징을 추출하는 방법이다. Extracting features by 'number' from text data.

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
pd.set_option("display.max_columns", None) # show table on one page
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

### Exercise

```python
import pandas as pd
import numpy as np

data = pd.read_excel("Electronics_data.xlsx")
data.head()
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
      <th>Unnamed: 0</th>
      <th>sentiment</th>
      <th>title</th>
      <th>Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>Great CD</td>
      <td>My lovely Pat has one of the GREAT voices of h...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>One of the best game music soundtracks - for a...</td>
      <td>Despite the fact that I have only played a sma...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>Batteries died within a year ...</td>
      <td>I bought this charger in Jul 2003 and it worke...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>works fine, but Maha Energy is better</td>
      <td>Check out Maha Energy's website. Their Powerex...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>Great for the non-audiophile</td>
      <td>Reviewed quite a bit of the combo players and ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (50000, 4)



```python
data["Full_text"] = data["title"].str.cat(data["Reviews"], sep = " ") # title과 Reviews 합치기
data.head()
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
      <th>Unnamed: 0</th>
      <th>sentiment</th>
      <th>title</th>
      <th>Reviews</th>
      <th>Full_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>Great CD</td>
      <td>My lovely Pat has one of the GREAT voices of h...</td>
      <td>Great CD My lovely Pat has one of the GREAT vo...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>One of the best game music soundtracks - for a...</td>
      <td>Despite the fact that I have only played a sma...</td>
      <td>One of the best game music soundtracks - for a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>Batteries died within a year ...</td>
      <td>I bought this charger in Jul 2003 and it worke...</td>
      <td>Batteries died within a year ... I bought this...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>works fine, but Maha Energy is better</td>
      <td>Check out Maha Energy's website. Their Powerex...</td>
      <td>works fine, but Maha Energy is better Check ou...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>Great for the non-audiophile</td>
      <td>Reviewed quite a bit of the combo players and ...</td>
      <td>Great for the non-audiophile Reviewed quite a ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
corpus = data['Full_text'].tolist() # Fulltext 리스트 형태로 corpus에 저장
len(corpus)
```




    500




```python
import nltk
import re

stop_words = nltk.corpus.stopwords.words("english")
ps = nltk.porter.PorterStemmer() # 어간 추출 (i.e., dies/dead/died --> die)

def normalize_document(doc):
    # normalization
    doc = re.sub(r'[^a-zA-Z\s]', "", doc, re.I|re.A) # 구두점 및 특수문자 제거 remove punctuation and speical characteristics
    doc = doc.lower()   # 소문자화 lowercase
    doc = doc.strip()   # 문자열의 앞 뒤에 있을 빈 칸 제거 strip string

    tokens = nltk.word_tokenize(doc) # tokenization

    filtered_tokens = [w for w in tokens if w not in stop_words] # remove stopwords

    # doc = " ".join(filtered_tokens)
    doc = " ".join([ ps.stem(w) for w in filtered_tokens ])

    return doc  
```

> re.I|re.A: re.sub의 flags ([click](https://docs.python.org/ko/3/library/re.html))


```python
normalize_corpus = np.vectorize(normalize_document) # vectorize the text
norm_corpus = normalize_corpus(corpus) # noremarize the corpus
norm_corpus
```




    array(['great cd love pat one great voic gener listen cd year still love im good mood make feel better bad mood evapor like sugar rain cd ooz life vocal jusat stuun lyric kill one life hidden gem desert isl cd book never made big beyond everytim play matter black white young old male femal everybodi say one thing sing',
           'one best game music soundtrack game didnt realli play despit fact play small portion game music heard plu connect chrono trigger great well led purchas soundtrack remain one favorit album incred mix fun epic emot song sad beauti track especi like there mani kind song video game soundtrack must admit one song lifea distant promis brought tear eye mani occasionsmi one complaint soundtrack use guitar fret effect mani song find distract even werent includ would still consid collect worth',
           'batteri die within year bought charger jul work ok design nice conveni howev year batteri would hold charg might well get alkalin dispos look elsewher charger come batteri better stay power',
          ...])




```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv_matrix = cv.fit_transform(norm_corpus) # perform vectorization
cv_matrix = cv_matrix.toarray() # pd.DataFrame 인자로 넣기위해 배열화 to be used for pd.DataFrame

```


```python
vocabulary = cv.get_feature_names()
len(vocabulary)
```




    4754




```python
one_hot = pd.DataFrame(cv_matrix, columns= vocabulary)
one_hot.shape
```




    (500, 4754)




```python
one_hot
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
      <th>aa</th>
      <th>abbrevi</th>
      <th>abduct</th>
      <th>abil</th>
      <th>abl</th>
      <th>aboard</th>
      <th>abor</th>
      <th>abound</th>
      <th>abridg</th>
      <th>abroad</th>
      <th>...</th>
      <th>yr</th>
      <th>yum</th>
      <th>yuppi</th>
      <th>zebra</th>
      <th>zep</th>
      <th>zero</th>
      <th>zhivago</th>
      <th>zillion</th>
      <th>zr</th>
      <th>zydeco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4754 columns</p>
</div>

## 2. TF-IDF
정보 검색과 텍스트 마이닝에서 이용하는 가중치로, 여러 문서로 이루어진 문서군이 있을 때 어떤 단어가 특정 문서 내에서 얼마나 중요한 것인지를 나타내는 통계적 수치이다. a statistical measure that evaluates how relevant a word is to a document in a collection of documents

*TF*: 특정 단어가 하나의 데이터 안에서 등장하는 횟수 how many times certain word appears in one data

*DF*: 문제 빈도 값으로, 특정 단어가 여러 데이터에 자주 등장하는지 알려주는 지표 how many times certain word appears in other data

*IDF(Inverse)*: DF의 역수를 취해서 구하며, 특정 단어가 다른 데이터에 등장하지 않을 경우 값이 커진다 As a word doesn't appear in other data, IDF increases

TF-IDF란 이 두 값을 곱해서 사용하므로 어떤 단어가 해당 문서에 자주 등장하지만 다른 문서에는 많이 없는 단어일수록 높은 값을 가진다. TF-IDF is computed by the multiplication of TF and IDF.

따라서, 조사나 지시대명사처럼 자주 등장하는 단어는 TF 값은 크지만 IDF 값은 작아지므로 CountVectorizer가 가진 문제점이 해결 가능하다. Thus, TF-IDF overcomes the limits of CountVectorizer.



```python
import pandas as pd
import numpy as np
```


```python
text1 ='''Tracy loves writing about data science.'''
text2 = '''Tracy loves writing for medium publications.'''
text3 = '''Tracy loves studying Python programming.'''
text4 = '''Tracy loves entering Kaggle competitions.'''
text5 = '''Tracy loves making YouTube videos.'''
text6 = '''Tracy loves getting new subscribers.'''
```


```python
corpus = [text1, text2, text3, text4, text5, text6]
corpus
```




    ['Tracy loves writing about data science.',
     'Tracy loves writing for medium publications.',
     'Tracy loves studying Python programming.',
     'Tracy loves entering Kaggle competitions.',
     'Tracy loves making YouTube videos.',
     'Tracy loves getting new subscribers.']




```python
from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer # TfidfTransformer
```


```python
count_vec = CountVectorizer()
```


```python
matrix = count_vec.fit_transform(corpus)
tf_transformer = TfidfTransformer().fit(matrix) # train TF-IDF
word_count_vec_tf = tf_transformer.transform(matrix) # apply TF-IDF
word_count_vec_tf.shape
```


    (6, 21)



```python
df0 = pd.DataFrame(word_count_vec_tf.toarray(), index=['doc1','doc2', 'doc3', 'doc4', 'doc5', 'doc6'],columns=count_vec.get_feature_names_out())
df0
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
        <th>...</th>
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
        <td>0.495894</td>
        <td>0.000000</td>
        <td>0.495894</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.220127</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>...</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.495894</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.220127</td>
        <td>0.000000</td>
        <td>0.40664</td>
        <td>0.000000</td>
      </tr>
      <tr>
        <th>doc2</th>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.495894</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.220127</td>
        <td>0.000000</td>
        <td>0.495894</td>
        <td>...</td>
        <td>0.000000</td>
        <td>0.495894</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.220127</td>
        <td>0.000000</td>
        <td>0.40664</td>
        <td>0.000000</td>
      </tr>
      <tr>
        <th>doc3</th>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.240948</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>...</td>
        <td>0.542798</td>
        <td>0.000000</td>
        <td>0.542798</td>
        <td>0.000000</td>
        <td>0.542798</td>
        <td>0.000000</td>
        <td>0.240948</td>
        <td>0.000000</td>
        <td>0.00000</td>
        <td>0.000000</td>
      </tr>
      <tr>
        <th>doc4</th>
        <td>0.000000</td>
        <td>0.542798</td>
        <td>0.000000</td>
        <td>0.542798</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.542798</td>
        <td>0.240948</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>...</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.240948</td>
        <td>0.000000</td>
        <td>0.00000</td>
        <td>0.000000</td>
      </tr>
      <tr>
        <th>doc5</th>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.240948</td>
        <td>0.542798</td>
        <td>0.000000</td>
        <td>...</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.240948</td>
        <td>0.542798</td>
        <td>0.00000</td>
        <td>0.542798</td>
      </tr>
      <tr>
        <th>doc6</th>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.542798</td>
        <td>0.000000</td>
        <td>0.240948</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>...</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.000000</td>
        <td>0.542798</td>
        <td>0.240948</td>
        <td>0.000000</td>
        <td>0.00000</td>
        <td>0.000000</td>
      </tr>
    </tbody>
  </table>
  <p>6 rows × 21 columns</p>
</div>


'doc1'에서 숫자 0을 가지는 열에 해당되는 단어들은 'doc1' 문장에 포함되지 않은 단어들이다.

TF-IDF 수치는 클수록 다른 문서에서 언급되지 않으면서 해당 문서에서 여러 번 사용되었다는 의미이다.

따라서, 'about'의 0.495894가 'loves'의 0.220127 보다 큰 것은 'about'이 다른 문서에서는 덜 사용됐으면서 해당 문서에서만 많이 사용되었기 때문으로 해석할 수 있다.

### Cosine Similarity
TF-IDF 벡터로 표현된 결과들 끼리의 코사인 연관성을 비교한다.

```python
from sklearn.metrics.pairwise import cosine_similarity
```


```python
cosine_similarity(df0[3:4], df0) # 문서 4와 나머지 문서들과의 코사인 연관성을 비교 consine similarity between docu 4 and others
```




    array([[0.10607812, 0.10607812, 0.1161115 , 1.        , 0.1161115 ,
            0.1161115 ]])



상기 결과는 문서4에 해당하는 인덱스 3 자리에 위치한 1이 자기 자신과의 유사도가 1로 완벽하다는 것을 의미한다.

문서1과 4의 연관성은 0.1061인 것으로 관찰된다.


## 3. Hashing Vectorizer
문장들을 token의 빈도수(= 횟수)로 행렬을 만드는 방법으로, CountVectorizer와 동일한 방식이다.

하지만, 'CountVectorizer'과 다르게 텍스트를 처리할 때 '해시'를 이용하여 '실행시간을 줄인다'.




```python
from sklearn.feature_extraction.text import HashingVectorizer
```


```python
vectorizer = HashingVectorizer(n_features = 2 ** 5) # 'n_features': 피쳐 개수 # features (default = 30,000) 
```

해쉬 함수를 사용하여 토큰 이름들을 맵핑된 32개의 피처를 제어한다. take control of token names that have been mapped into 32 features using hash function
- 해쉬 함수를 통하여 32개의 피처 중 알맞은 피처의 인덱스를 가져온다. hash function helps find the index of appropriate feature

```python
X = vectorizer.fit_transform(corpus)
X.shape
```




    (6, 32)




```python
matrix = pd.DataFrame(X.toarray(), index=['doc1','doc2', 'doc3', 'doc4', 'doc5', 'doc6'])
matrix
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>doc1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>doc2</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.408248</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.408248</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.408248</td>
    </tr>
    <tr>
      <th>doc3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.447214</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.447214</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>doc4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.447214</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.447214</td>
      <td>0.447214</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>doc5</th>
      <td>0.0</td>
      <td>-0.447214</td>
      <td>0.0</td>
      <td>0.447214</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.447214</td>
      <td>0.000000</td>
      <td>0.447214</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>doc6</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.755929</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 32 columns</p>
</div>


## 4. FeatureHasher
Feature Hashing이라는 방법을 이용하여 '약간 메모리를 사용하고' 빠르게 벡터화 하는 방법이다.


```python
from sklearn.feature_extraction import FeatureHasher
```


```python
hasher = FeatureHasher(n_features = 21, alternate_sign=False, input_type="string")
```


```python
vectors = hasher.transform(corpus)
vectors.toarray()
```


    array([[0., 0., 4., 0., 3., 0., 0., 1., 0., 0., 1., 5., 3., 1., 6., 4.,
            6., 1., 3., 0., 1.],
           [0., 2., 3., 0., 4., 0., 0., 1., 0., 0., 1., 4., 3., 1., 6., 4.,
            7., 1., 5., 0., 2.],
           [0., 2., 5., 0., 4., 0., 0., 1., 1., 0., 3., 3., 3., 0., 5., 5.,
            5., 0., 2., 0., 1.],
           [0., 1., 5., 0., 4., 0., 0., 1., 0., 0., 1., 4., 4., 0., 5., 5.,
            7., 0., 3., 0., 1.],
           [0., 1., 2., 0., 4., 0., 0., 2., 0., 0., 1., 3., 0., 0., 5., 4.,
            6., 1., 2., 0., 3.],
           [0., 0., 4., 0., 1., 0., 0., 1., 0., 0., 1., 2., 2., 1., 5., 6.,
            8., 2., 2., 0., 1.]])

## 5. Dict Vectorizer
CountVectorizer과 동일한 방식으로 동작하지만, **딕셔너리** 데이터를 인풋으로 받는다는 점에서 차이가 있다. Dic Vectorizer shares the same way to operate but receives dictionary input data.

```python
from sklearn.feature_extraction import DictVectorizer
```

```python
# sample dict data
dict = [
    {'name': 'Tracy', 'from': 'Little Rock', 'sex':'female', 'age': 60},
    {'name': 'Mike', 'from': 'Reading', 'sex':'male', 'age': 54},
    {'name': 'Martina', 'from': 'Berlin', 'sex':'female', 'age':62},
    {'name': 'Gerry', 'from':'Heerlen', 'sex':'female', 'age': 80},
    {'name': 'Paz', 'from': 'Manila', 'sex':'female', 'age': 61},
    {'name': 'Doug', 'from': 'Aberdeen', 'sex':'male', 'age': 55},
    {'name': 'Jeff', 'from': 'Cardiff', 'sex':'male', 'age': 57},
    {'name': 'Cindy', 'from':'Little Rock', 'sex': 'female', 'age':60},
    {'name': 'Keith', 'from': 'Reading', 'sex': 'male', 'age':57},
    {'name': 'Adrian', 'from': 'Tercera', 'sex': 'male', 'age': 38},
    {'name': 'Katherine', 'from': 'Fayeteville', 'sex': 'female', 'age': 38}
]
```

```python
vec = DictVectorizer()
vectors = vec.fit_transform(dict)
vectors.toarray()
```


    array([[60.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.],
          [54.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.],
          [62.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
          [80.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
          [61.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.],
          [55.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
          [57.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
          [60.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
          [57.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.],
          [38.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
          [38.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]])


```python
vec.get_feature_names_out(), len(vec.get_feature_names_out())
```


    (array(['age', 'from=Aberdeen', 'from=Berlin', 'from=Cardiff',
            'from=Fayeteville', 'from=Heerlen', 'from=Little Rock',
            'from=Manila', 'from=Reading', 'from=Tercera', 'name=Adrian',
            'name=Cindy', 'name=Doug', 'name=Gerry', 'name=Jeff',
            'name=Katherine', 'name=Keith', 'name=Martina', 'name=Mike',
            'name=Paz', 'name=Tracy', 'sex=female', 'sex=male'], dtype=object),
    23)


상기 과정에서 Dict Vectorizer를 통하여 딕셔너리 데이터를 수치에 기반하여 벡터화했다. The result shows how vectorization has been done using Dict Vectorizer.

이후, TF-IDF를 사용하여 횟수 기반 벡터화된 행렬을 변환하여 보다 유의미한 단어를 도출해보자. Now, let's use TF-IDF to find informative words from the output.

```python
from sklearn.feature_extraction.text import TfidfTransformer
```

```python
tf_transformer = TfidfTransformer().fit(vectors)
word_count_vec_tf = tf_transformer.transform(vectors)
word_count_vec_tf.shape
```


  (11, 23)



```python
df0 = pd.DataFrame(word_count_vec_tf.toarray())
df0
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.997805</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039684</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.046427</td>
      <td>0.025594</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.997207</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.044067</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.051555</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031267</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.997673</td>
      <td>0.000000</td>
      <td>0.044924</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.044924</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024765</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.998600</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.034848</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.034848</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.019211</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.997596</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045657</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045657</td>
      <td>0.000000</td>
      <td>0.025169</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.996964</td>
      <td>0.050605</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030691</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.997172</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.04884</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.04884</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029620</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.997805</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039684</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025594</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.997493</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041760</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.048855</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029630</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.993671</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.073002</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.044274</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.993840</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.073015</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.073015</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040250</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 23 columns</p>
</div>
