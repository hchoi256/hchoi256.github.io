---
layout: single
title: "[개발] PART 4: Naive Classifier"
categories: NLP
tag: [NLP, Naive Classifier]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/nlp-thumbnail.jpg
sidebar:
    nav: "docs"
---

# PART 1: Gender Identifier
Corpus의 영어 이름에서 마지막 '몇 글자'를 보고 남자 혹은 여자 이름인지 확인하여 분류한다. <span style="color: yellow"> Identifying sex through the last 'few letters' of the corpus</span>

Naive Bayes Classifier
1. 모든 고유한 단어가 텍스트에서 추출 <span style="color: yellow"> extracting words from texts</span>
2. 라벨 확인 <span style="color: yellow"> checking labels</span>
3. 분류 <span style="color: yellow"> classification </span>

> [Naive Beyas](https://github.com/hchoi256/ai-boot-camp/blob/main/ai/machine-learning/supervised-learning/classification/naive-bayes.md)

## Loading the libraries and dataset
```python
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names
import nltk
nltk.download("names")
```


```python
male_list = [(name, "male") for name in names.words("male.txt")]
female_list = [(name, "female") for name in names.words("female.txt")]
data = (male_list + female_list) # corpus

print(len(male_list))
print(len(female_list))
```


        2943
        5001


```python
data[:30]
```


        [('Aamir', 'male'),
        ('Aaron', 'male'),
        ('Abbey', 'male'),
        ('Abbie', 'male'),
        ('Abbot', 'male'),
        ('Abbott', 'male'),
        ('Abby', 'male'),
        ...
        ('Adnan', 'male'),
        ('Adolf', 'male'),
        ('Adolfo', 'male'),
        ('Adolph', 'male'),
        ('Adolphe', 'male')]


```python
# loading the last few words
def extract_features(word, N=2):
    last_n_letters = worda[-N:]
    return {"feature:":last_n_letters.lower()}
```


```python
random.seed(5)
random.shuffle(data) # avoid overfitting
```


```python
input_names = ["Alexander", "Daniellle", "David", "Cheryl"]  # sample
num_train = int(0.8*len(data)) # train/test ratio
```

## Training the model

```python
for i in range(1, 6):
    print("Number of end letters: ", i)
    features = [(extract_features(n, i), gender) for (n, gender) in data]
    train_data, test_data = features[:num_train], features[num_train:]
    classifier = NaiveBayesClassifier.train(train_data)
    accuracy = round(100 * nltk_accuracy(classifier, test_data), 2)
    print("Accuracy = " + str(accuracy) + '%')

    for name in input_names:
        print(name, "===>", classifier.classify(extract_features(name, i)))
    
```


        Number of end letters:  1
        Accuracy = 74.7%
        Alexander ===> male
        Daniellle ===> female
        David ===> male
        Cheryl ===> male
        Number of end letters:  2
        Accuracy = 78.79%
        Alexander ===> male
        Daniellle ===> female
        David ===> male
        Cheryl ===> female
        ...

상기 결과는 단어의 마지막 문자 1, 2개를 기준으로 성별을 분류했을 때 정확도이다. <span style="color: yellow"> The result represents the accuracy of the classifier that identifies sex based on the last one and two words respectively.</span>

나머지 마지막 문자 3개 이상을 기준으로한 결과값은 편의상 생략하였다. <span style="color: yellow"> The other outcomes are omitted for convenience.</span>

# PART 2: Sentimental Analysis
감성분석(Sentiment Analysis): 텍스트 일부로 감정을 결정하는 과정이다 (좋아, 나뻐). <span style="color: yellow"> Sentiment Analysis studies how to determine sentiment through part of texts (good, poor).</span>

남자 이름인지 여자 이름인지... / 영화 리뷰가 긍정인지... 부정인지... <span style="color: yellow"> Man/Woman, Positive/Negative</span>

제품이 좋은지 나쁜지... 여론조사 등등 확장이 가능 <span style="color: yellow"> Good/Poor, etc.</span>

## Loading the libraries
```python
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import movie_reviews
import nltk
nltk.download("movie_reviews")
```

```python
fileids_pos = movie_reviews.fileids("pos")
fileids_neg = movie_reviews.fileids("neg")
fileids_pos[:1]
```

        ['pos/cv000_29590.txt']


상기 라이브러리의 'pos' 디렉토리의 해당 텍스트에 긍정 단어들이 포함되어 있다. <span style="color: yellow"> The directory above incldues positive words.</span>

어떤 단어들이 속해있는지 지금 바로 확인해보자. <span style="color: yellow"> Let's go check those words now.</span>

```python
movie_reviews.words(fileids=['pos/cv000_29590.txt'])
```


        ['films', 'adapted', 'from', 'comic', 'books', 'have', ...]


```python
def extract_features(words):
    return dict([ (word, True) for word in words])
```


```python
features_pos= [ (extract_features(movie_reviews.words(fileids=[f])), "Positive") for f in fileids_pos ]
features_neg= [ (extract_features(movie_reviews.words(fileids=[f])), "Negative") for f in fileids_neg ]
```

```python
threshold = 0.8 # train/test set ratio
num_pos = int(threshold * len(features_pos))
num_neg = int(threshold * len(features_neg))

features_train = features_pos[:num_pos] + features_neg[:num_neg]
features_test = features_pos[num_pos:] + features_neg[num_neg:]
print("Train Dataset : ", len(features_train))
print("Test Dataset : ", len(features_test))
```


        Train Dataset :  1600
        Test Dataset :  400


```python
classifier = NaiveBayesClassifier.train(features_train) # training the model
accuracy = round(100 * nltk_accuracy(classifier, features_test), 2) # prediction
accuracy
```

        73.5


```python
# test sample
input_reviews = [
    'The costumes in this movie were great', 
    'I think the story was terrible and the characters were very weak',
    'People say that the director of the movie is amazing', 
    'This is such an idiotic movie. I will not recommend it to anyone.' 
]
```

```python
from nltk import probability

probabilities = classifier.prob_classify(extract_features(review.split()))
predicted_sentiment = probabilities.max() # 긍부정 우위 선택 pos vs. neg
print("Predicted sentiment :", predicted_sentiment)
print("Probabilities : ", round(probabilities.prob(predicted_sentiment), 2))

```

        Predicted sentiment : Negative
        Probabilities :  0.8


긍부정 단어 개수를 계산하여 빈도수가 더 많이 등장한 선택지로 결론내린다. <span style="color: yellow"> By counting the number of positive and negative words, we conclude with the option that appears more frequently.</span>

결과는 Negative로 부정 단어가 더 많이 포함된 문장이었고, 그 빈도수는 0.8이며 긍정은 0.2이다. <span style="color: yellow"> The result was a sentence containing more negative words as negative, and the frequency was 0.8 and positive was 0.2.</span>

