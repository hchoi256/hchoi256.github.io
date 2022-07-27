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

# 어간추출(Stemmer) vs. 표제어추출(Lemmatizer)
## Stemmer
단어에서 일반적인 형태 및 굴절 어미를 제거하는 프로세스. A process for removing the commoner morphological and inflexional endings from words.


```python
from nltk.stem.porter import PorterStemmer # least strict
from nltk.stem.snowball import SnowballStemmer # average (best)
from nltk.stem.lancaster import LancasterStemmer # most strict 
```

```python
input_words = ['writing', 'calves', 'be', 'branded', 'house', 'randomize', 'possibly', 'extraction', 'hospital', 'kept', 'scratchy', 'code'] # sample

porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()
```

```python
stemmers_names = ["PORTER", "SNOWBALL", "LANCASTER"]
formatted_text = "{:>16}" * (len(stemmers_names) + 1)
print(formatted_text.format("INPUT WORD", *stemmers_names), "\n", "="*68)
for word in input_words:
    output = [word, porter.stem(word), snowball.stem(word), lancaster.stem(word)]
    print(formatted_text.format(*output))
```


          INPUT WORD          PORTER        SNOWBALL       LANCASTER 
    ====================================================================
            writing           write           write            writ
              calves            calv            calv            calv
                  be              be              be              be
            branded           brand           brand           brand
              house            hous            hous            hous
          randomize          random          random          random
            possibly         possibl         possibl            poss
          extraction         extract         extract         extract
            hospital          hospit          hospit          hospit
                kept            kept            kept            kept
            scratchy        scratchi        scratchi        scratchy
                code            code            code             cod

## Lemmatizer
'단어의 원형'을 찾고자 하는 또 다른 형태, 표제어는 단어의 다양한 굴절 형태를 그룹화하여 단일 항목으로 분석할 수 있도록 하는 과정이다. Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item

```python
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
```

```python
lemmatizer_names = ["NOUN LEMMATIZER", "VERB LEMMATIZER"]
formatted_text = "{:>24}" * (len(lemmatizer_names) + 1)
print(formatted_text.format("INPUT WORD", *lemmatizer_names), "\n", "="*75)
for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos="n"), lemmatizer.lemmatize(word, pos="v")]
    print(formatted_text.format(*output))
```


                  INPUT WORD         NOUN LEMMATIZER         VERB LEMMATIZER 
    ===========================================================================
                    writing                 writing                   write
                      calves                    calf                   calve
                          be                      be                      be
                    branded                 branded                   brand
                      house                   house                   house
                  randomize               randomize               randomize
                    possibly                possibly                possibly
                  extraction              extraction              extraction
                    hospital                hospital                hospital
                        kept                    kept                    keep
                    scratchy                scratchy                scratchy
                        code                    code                    code


# Chunking
텍스트 데이터는 일반적으로 추가 분석을 위해 조각으로 나누어야 할 필요가 있다. Text data usually needs to be broken into pieces for further analysis.


```python
import numpy as np
from nltk.corpus import brown
nltk.download("brown")
```

```python
def chunker(input_data, N):
    input_words = input_data.split()
    output = []

    cur_chunk = []
    count = 0

    for word in input_words:
        cur_chunk.append(word)
        count += 1
        if count == N:
            output.append(" ".join(cur_chunk))
            # print(cur_chunk)
            count, cur_chunk = 0, []
        # print(output)

    output.append(" ".join(cur_chunk))

    return output
```

```python
input_data = " ".join(brown.words()[:14000])
chunk_size = 700
chunks = chunker(input_data, chunk_size)
```

상기 코드는 brown 라이브러리 데이터의 14000 단어를 불러와 700개의 단어 단위로 chunk를 생성한다. The above code fetches 14000 words of brown library data and creates chunks in units of 700 words.

```python
print("Number of text chunks =", len(chunks), "\n")
for i, chunk in enumerate(chunks):
    print("Chunk", i + 1, "==>" ,chunk[:50]) # show 50 words out of 700
```

    Number of text chunks = 21 

    Chunk 1 ==> The Fulton County Grand Jury said Friday an invest
    Chunk 2 ==> '' . ( 2 ) Fulton legislators `` work with city of
    Chunk 3 ==> . Construction bonds Meanwhile , it was learned th
    Chunk 4 ==> , anonymous midnight phone calls and veiled threat
    Chunk 5 ==> Harris , Bexar , Tarrant and El Paso would be $451
    Chunk 6 ==> set it for public hearing on Feb. 22 . The proposa
    Chunk 7 ==> College . He has served as a border patrolman and 
    Chunk 8 ==> of his staff were doing on the address involved co
    Chunk 9 ==> plan alone would boost the base to $5,000 a year a
    Chunk 10 ==> nursing homes In the area of `` community health s
    Chunk 11 ==> of its Angola policy prove harsh , there has been 
    Chunk 12 ==> system which will prevent Laos from being used as 
    Chunk 13 ==> reform in recipient nations . In Laos , the admini
    Chunk 14 ==> . He is not interested in being named a full-time 
    Chunk 15 ==> said , `` to obtain the views of the general publi
    Chunk 16 ==> '' . Mr. Reama , far from really being retired , i
    Chunk 17 ==> making enforcement of minor offenses more effectiv
    Chunk 18 ==> to tell the people where he stands on the tax issu
    Chunk 19 ==> '' . Trenton -- William J. Seidel , state fire war
    Chunk 20 ==> not comment on tax reforms or other issues in whic
    Chunk 21 ==> 


# Bag of Words
Bag of Words 모델을 사용해서 텍스트 분석을 하는 주요 목적 중에 하나는 텍스트를 기계학습에서 사용할 수 있도록 텍스트를 숫자 형식으로 변환하는 것이다. One of the main purposes of text analysis using the Bag of Words model is to convert the text into a numeric form so that it can be used in machine learning.

수백만 단어가 포함된 텍스트 문서를 분석하려고 한다. You want to analyze a text document containing millions of words.

그러기 위해선, 텍스트를 추출하고 숫자 표현 형식으로 변환해야 한다. To do that, we need to extract the text and convert it to a numeric representation.

기계 학습 알고리즘은 데이터를 분석하고 의미 있는 정보를 추출할 수 있도록 작업할 숫자 데이터가 필요하다. Machine learning algorithms need numeric data to work with so that they can analyze the data and extract meaningful information.

Bag of Words 모델은 문서의 모든 단어에서 어휘를 추출하고 문서-용어 행렬 (matrix)를 사용하여 모델을 구축한다. The Bag of Words model extracts vocabulary from every word in a document and builds the model using a document-term matrix.
- 이 모델을 사용하면 모든 문서를 단어 모음으로 나타낼 수 있다. This model allows any document to be represented as a collection of words.
- 단어 갯수, 문법적 세부 사항, 단어 순서를 무시한다. Ignoring word count, grammatical details, and word order.

문서-용어 행렬은 기본적으로 문서에서 발생하는 다양한 단어의 수를 제공하는 테이블이다. The document-term matrix is ​​basically a table that gives the number of different words that occur in a document.

텍스트 문서는 다양한 단어의 가중치 조항으로 표현되고, 임계값을 설정하고 더 의미 있는 단어를 선택할 수 있다. Text documents are represented by weighted clauses of various words; more meaningful with thresholds words can be selected.

feature vector로 사용될 문서의 모든 단어들의 히스토그램을 만들고, 이 feature vector는 텍스트 분류에 사용할 수 있다. Create a histogram of all words in the document to be used as a feature vector, and this feature vector can be used for text classification.


```python
from sklearn.feature_extraction.text import CountVectorizer
```

```python
input_data = " ".join(brown.words()[:5500])
chunk_size = 800
text = chunker(input_data, chunk_size)
```

```python
chunks = []
for count, chunk in enumerate(text):
    d = {"index": count, "text": chunk}
    chunks.append(d)
len(chunks)
```


    7



```python
count_vectorizer = CountVectorizer()
document_term_matrix = count_vectorizer.fit_transform([chunk["text"] for chunk in chunks])
vocabulary = np.array(count_vectorizer.get_feature_names_out())
print(vocabulary)
```


    ['000' '10' '100' ... 'york' 'you' 'your']

> [CountVectorizer](#1-countvectorizer)


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
