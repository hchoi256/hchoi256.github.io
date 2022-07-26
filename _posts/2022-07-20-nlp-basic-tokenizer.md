---
layout: single
title: "NLP - Part 1: Text Mining and Tokenization"
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

# PART 1: Text Mining
## 1-1) 텍스트 마이닝(Text Mining)이란?
비정형(= 구조화 되지 않은) 텍스트 빅 데이터에 대하여 효과적인 탐색 및 분석을 통해 유용한 정보, 패턴 및 실행 가능한 통찰력을 도출하는 과정이다.

'Text Mining' utilizes effective exploration and analysis to produce feasible insights (i.e., valuable patterns) into the extensive dataset of unstructured texts.

### 텍스트 마이닝의 필요성(Why We Need Text Mining?)

일반적으로 텍스트 데이터는 구조화가 되어있지 않아서 비지도학습이 활용된다.

하지만, 실생활에는 엄청나게 많은 양의 텍스트 데이터가 순간마다 생성되고 있다 ([차원의 저주](https://github.com/hchoi256/ai-terms)).

따라서, 텍스트 마이닝은 이러한 텍스트 데이터에서 business insight를 식별하여 목표 성취에 위험을 줄이는 것에 뜻이 있다.

> 정보에 입각한 결정을 내리고, 프로세스를 자동화하고, 감정 분석을 사용한 시장 조사 등을 수행하고자 한다.

### 텍스트 마이닝의 분야

* **정보 추출** (Information Extraction)
* **문서 분류 및 클러스터링** (Document classification and clustering)
* **정보 검색** (Information retrieval)
* **자연어 처리** (Natural Language Processing)

> NLP (Processing) -> NLU (Understaing) -> NLG (Generate)

### NLP: 텍스트 마이닝의 핵심

컴퓨터를 사용하여 인간의 언어를 이해, 해석 및 조작하는 것을 다루는 연구 분야이다.

> Natural Languae : 인간이 의사소통하는 모든 언어를 지칭한다!

NLP은 인공지능의 **하위 도메인**이고, 중요한 정보의 대부분은 **자연어**로 작성되어 편리하게 태그가 지정되지 않기 때문에 텍스트 분석을 위한 컨텐츠를 식별하고 추출한 후 다양한 NLP기술을 사용하여 의미 있는 정보를 추출하고자 한다.

#### NLTK

NLTK이란 NLP를 위해 가장 널리 사용되는 패키지 중에 하나로, 컴퓨터가 자연어와 텍스트를 전처리, 분석 및 이해하는데 도움이 되는 여러 알고리즘이 포함된 강력한 파이썬 패키지이다.

#### NLTK에서 사용 가능한 일반적인 알고리즘
* **Tokenization** : 토큰화
* **Part of Speech Tagging** : 품사 태깅
* **Named Entity Recognition** : NER (명명된 엔티티 인식)
* **Sentiment analysis** : 감정 분석


```python
import nltk
```


```python
nltk.download("names")
```


```python
from nltk.corpus import names
```


```python
names.words()
```




    ['Abagael',
     'Abagail',
     'Abbe',
     'Abbey',
     'Abbi',
     'Abbie',
     'Abby',
     'Abigael',
     'Abigail',
     'Abigale',
     'Abra',
     ...]



> *corpus(말뭉치)*
>> 논리적으로 **최소한**의 의미를 전달할 수 있는 단일 텍스트들의 모음이다.

# PART 2: Tokenization
## 2-1) Tokenization
* **word tokenization** : 대량의 텍스트를 단어로 분할하는 과정 각 단어를 인식하고 특정 감정에 대한 분류 및 계산과 같은 추가 분석을 거쳐야 하는 자연어 처리 작업
* **sentence tokenization** : 문장을 기준으로 토큰화
* **Regex tokenization** : Regex를 기준으로 해서 토큰화
* **Blank line tokenization** : 빈 줄을 기준으로 토큰화


### Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize
```

```python
nltk.download("gutenberg")
```

```python
from nltk.corpus import gutenberg
```


```python
sample = gutenberg.raw("carroll-alice.txt")
```


```python
sent_tokens = sent_tokenize(sample)
```


```python
len(sent_tokens)
```




    1625




```python
sent_tokens[0:10]
```




    ["[Alice's Adventures in Wonderland by Lewis Carroll 1865]\n\nCHAPTER I.",
     "Down the Rabbit-Hole\n\nAlice was beginning to get very tired of sitting by her sister on the\nbank, and of having nothing to do: once or twice she had peeped into the\nbook her sister was reading, but it had no pictures or conversations in\nit, 'and what is the use of a book,' thought Alice 'without pictures or\nconversation?'",
     'So she was considering in her own mind (as well as she could, for the\nhot day made her feel very sleepy and stupid), whether the pleasure\nof making a daisy-chain would be worth the trouble of getting up and\npicking the daisies, when suddenly a White Rabbit with pink eyes ran\nclose by her.',
     "There was nothing so VERY remarkable in that; nor did Alice think it so\nVERY much out of the way to hear the Rabbit say to itself, 'Oh dear!",
     'Oh dear!',
     "I shall be late!'",
     '(when she thought it over afterwards, it\noccurred to her that she ought to have wondered at this, but at the time\nit all seemed quite natural); but when the Rabbit actually TOOK A WATCH\nOUT OF ITS WAISTCOAT-POCKET, and looked at it, and then hurried on,\nAlice started to her feet, for it flashed across her mind that she had\nnever before seen a rabbit with either a waistcoat-pocket, or a watch\nto take out of it, and burning with curiosity, she ran across the field\nafter it, and fortunately was just in time to see it pop down a large\nrabbit-hole under the hedge.',
     'In another moment down went Alice after it, never once considering how\nin the world she was to get out again.',
     'The rabbit-hole went straight on like a tunnel for some way, and then\ndipped suddenly down, so suddenly that Alice had not a moment to think\nabout stopping herself before she found herself falling down a very deep\nwell.',
     'Either the well was very deep, or she fell very slowly, for she had\nplenty of time as she went down to look about her and to wonder what was\ngoing to happen next.']



### Word Tokenization

```python
from nltk.tokenize import word_tokenize
```


```python
tok = word_tokenize(sample)
```


```python
len(tok)
```




    33494




```python
tok[0:20]
```




    ['[',
     'Alice',
     "'s",
     'Adventures',
     'in',
     'Wonderland',
     'by',
     'Lewis',
     'Carroll',
     '1865',
     ']',
     'CHAPTER',
     'I',
     '.',
     'Down',
     'the',
     'Rabbit-Hole',
     'Alice',
     'was',
     'beginning']


### Regex Tokenization


```python
sample = """
Volodymyr Oleksandrovych Zelenskyy (Ukrainian: Володимир Олександрович Зеленський; Russian: Владимир Александрович Зеленский, romanized: Vladimir Aleksandrovich Zelenskyy,[a] born 25 January 1978), also transliterated as Zelensky or Zelenskiy,[b] is a Ukrainian politician and former comedic actor[5] who has served as the 6th and current president of Ukraine since 2019.

Born to a Jewish family, Zelenskyy grew up as a native Russian speaker in Kryvyi Rih, a major city of Dnipropetrovsk Oblast in central Ukraine. Prior to his acting career, he obtained a degree in law from the Kyiv National Economic University. He then pursued a career in comedy and created the production company Kvartal 95, which produced films, cartoons, and TV shows including the TV series Servant of the People, in which Zelenskyy played the role of the Ukrainian president. The series aired from 2015 to 2019 and was immensely popular. A political party bearing the same name as the television show was created in March 2018 by employees of Kvartal 95.

Zelenskyy announced his candidacy in the 2019 Ukrainian presidential election on the evening of 31 December 2018, alongside the New Year's Eve address of then-president Petro Poroshenko on the TV channel 1+1. A political outsider, he had already become one of the frontrunners in opinion polls for the election. He won the election with 73.23 percent of the vote in the second round, defeating Poroshenko. He has positioned himself as an anti-establishment and anti-corruption figure.

As president, Zelenskyy has been a proponent of e-government and of unity between the Ukrainian- and Russian-speaking parts of the country's population.[6]: 11–13  His communication style heavily uses social media, particularly Instagram.[6]: 7–10  His party won a landslide victory in the snap legislative election held shortly after his inauguration as president. During his administration, Zelenskyy oversaw the lifting of legal immunity for members of parliament (the Verkhovna Rada),[7] the country's response to the COVID-19 pandemic and subsequent economic recession, and some progress in tackling corruption in Ukraine.[8][9]

During his presidential campaign, Zelenskyy promised to end Ukraine's protracted conflict with Russia, and he has attempted to engage in dialogue with Russian president Vladimir Putin.[10] His administration faced an escalation of tensions with Russia in 2021, culminating in the launch of the ongoing full-scale Russian invasion in February 2022. Zelenskyy's strategy during the Russian military buildup was to calm the Ukrainian populace and assure the international community that Ukraine was not seeking to retaliate.[11] He initially distanced himself from warnings of an imminent war, while also calling for security guarantees and military support from NATO to "withstand" the threat.[12] After the start of the invasion, Zelenskyy declared martial law across Ukraine and a general mobilisation of the armed forces. His leadership during the crisis has won him widespread international praise, and he has been described as a symbol of the Ukrainian resistance.[13][14]


"""
```


```python
from nltk.tokenize import RegexpTokenizer
```


```python
capword_tokenizer = RegexpTokenizer("[A-Z]\w+")
capword_tokenizer.tokenize(sample)
```




    ['Volodymyr',
     'Oleksandrovych',
     'Zelenskyy',
     'Ukrainian',
     'Russian',
     'Vladimir',
     'Aleksandrovich',
     'Zelenskyy',
     'January',
     'Zelensky',
     'Zelenskiy',
     'Ukrainian',
     'Ukraine',
     'Born',
     'Jewish',
     'Zelenskyy',
     'Russian',
     'Kryvyi',
     'Rih',
     'Dnipropetrovsk',
     'Oblast',
     'Ukraine',
     'Prior',
     'Kyiv',
     ...]




```python
capword_tokenizer = RegexpTokenizer("[0-9]\w+")
capword_tokenizer.tokenize(sample)
```




    ['25',
     '1978',
     '6th',
     '2019',
     '95',
     '2015',
     '2019',
     '2018',
     '95',
     '2019',
     '31',
     '2018',
     '73',
     '23',
     '11',
     '13',
     '10',
     '19',
     '10',
     '2021',
     '2022',
     '11',
     '12',
     '13',
     '14']


### Blankline Tokenizer


```python
from nltk.tokenize import BlanklineTokenizer
```


```python
sample = """
Good muffins cost £1.50 in London.\n\n Please can you buy me two of them. \n\n Thanks.
"""
```


```python
BlanklineTokenizer().tokenize(sample)
```




    ['\nGood muffins cost £1.50 in London.',
     'Please can you buy me two of them.',
     'Thanks.\n']




```python
sample = r"""
Good muffins cost £1.50 in London.\n\n Please can you buy me two of them. \n\n Thanks.
"""
```


```python
BlanklineTokenizer().tokenize(sample)
```




    ['\nGood muffins cost £1.50 in London.\\n\\n Please can you buy me two of them. \\n\\n Thanks.\n']



> **f""** : f-string (format을 중시)
>
> **r""** : r-string (문자열의 문자 자체를 유지하려한다)
> 
> **b""** : b-string (사람이 보기에는 일반 문자열처럼 보이지만, 내부적으로 byte형 문자열을 처리)

## 2-2) Frequency Distribution (빈도 분포)
텍스트에서 단어 빈도를 알아본다. Key가 단어이고, Value가 단어와 관련된 개수이다.


```python
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg
from nltk.probability import FreqDist # 문서에서 각 단어가 발생하는 횟수를 계산
```


```python
sample = """
The Bible (from Koine Greek τὰ βιβλία, tà biblía, 'the books') is a collection of religious texts or scriptures sacred in Christianity, Judaism, Samaritanism, and many other religions. The Bible is an anthology—a compilation of texts of a variety of forms—originally written in Hebrew, Aramaic, and Koine Greek. These texts include instructions, stories, poetry, and prophesies, among other genres. The collection of materials that are accepted as part of the Bible by a particular religious tradition or community is called a biblical canon. Believers in the Bible generally consider it to be a product of divine inspiration, while understanding what that means in different ways.

The origins of the oldest writings of the Israelites are lost to antiquity. The religious texts were compiled by different religious communities into various official collections. The earliest contained the first five books of the Bible, called the Torah, which was accepted as Jewish canon by the 5th century BCE. A second collection of narrative histories and prophesies was canonized in the 3rd century BCE. A third collection containing psalms, proverbs, and narrative histories, was canonized sometime between the 2nd century BCE and the 2nd century CE.[1] The transmission history of these combined collections spans approximately 3000 years, and there is no scholarly consensus as to when the Jewish Hebrew Bible canon was settled in its present form.[2] Some scholars argue that it was fixed by the Hasmonean dynasty (140–40 BCE),[a] while others argue it was not fixed until the second century CE or even later.[3] The Dead Sea scrolls are approximately dated to 250 BCE–100 CE and are the oldest existing copies of the books of the Hebrew Bible. Tanakh is an alternate term for the Hebrew Bible composed of the first letters of the three parts of the Hebrew scriptures: the Torah ("Teaching"), the Nevi'im ("Prophets"), and the Ketuvim ("Writings"). The Torah is also known as the Pentateuch. The Masoretic Text, in Hebrew and Aramaic, is considered the authoritative text by Rabbinic Judaism; the Septuagint, a Koine Greek translation from the third and second centuries BCE, largely overlaps with the Hebrew Bible.

Christianity began as an outgrowth of Judaism, using the Septuagint as the basis of the Old Testament. The early Church continued the Jewish tradition of writing and incorporating what it saw as inspired, authoritative religious books. The gospels, Pauline epistles and other texts coalesced into the "New Testament" very early. In the first three centuries CE, the concept of a closed canon emerged in response to heretical writings in the second century. The list of books included in the Catholic Bible was established as canon by the Council of Rome in 382, followed by those of Hippo in 393 and Carthage in 397. Christian biblical canons range from the 73 books of the Catholic Church canon, and the 66-book canon of most Protestant denominations, to the 81 books of the Ethiopian Orthodox Tewahedo Church canon, among others.

With estimated total sales of over five billion copies, the Bible is widely considered to be the best-selling publication of all time.[4][5] It has had a profound influence both on Western culture and history and on cultures around the globe.[6] "Simply put, the Bible is the most influential book of all-time."[7] The study of the Bible through biblical criticism has indirectly impacted culture and history as well. The Bible is currently translated or being translated into about half of the world's languages.
"""
```


```python
token_words = word_tokenize(sample)
```


```python
token_words[:20]
```




    ['The',
     'Bible',
     '(',
     'from',
     'Koine',
     'Greek',
     'τὰ',
     'βιβλία',
     ',',
     'tà',
     'biblía',
     ',',
     "'the",
     'books',
     "'",
     ')',
     'is',
     'a',
     'collection',
     'of']




```python
fdist = FreqDist(token_words)
```


```python
fdist
```




    FreqDist({'the': 51, ',': 35, 'of': 32, '.': 27, 'and': 18, 'The': 15, 'Bible': 14, 'in': 13, 'is': 10, 'a': 9, ...})




```python
capword_tokenizer = RegexpTokenizer("[A-Za-z0-9]\w+")
fdist = FreqDist(capword_tokenizer.tokenize(sample))
```


```python
fdist
```




    FreqDist({'the': 52, 'of': 32, 'and': 18, 'The': 15, 'Bible': 14, 'in': 13, 'is': 10, 'as': 9, 'canon': 8, 'books': 7, ...})




```python
%pip install matplotlib
```


```python
fdist.plot(10, title="Top 10 Most Common Words")
```


    
![output_66_0](https://user-images.githubusercontent.com/39285147/180378796-f8a9a87f-2f43-4f6a-9d3b-64b415132eba.png)
    



## 2-3) Stop Words (불용어)
일반적으로 자연어를 처리하기 전에 걸러내는 단어이다.

실제로 모든 언어에서 가장 흔한 단어 (관사, 전치사, 대명사, 접속사 등)이며 텍스트에 많은 정보를 추가하지 않는다는 것이 특징이다.

주어진 말뭉치의 텍스트에서 불용어를 제거하여 데이터를 정리하고, 더 희귀하고 우리가 관심 있는 것과 잠재적으로 더 관련이 있는 단어를 식별하고자 하는 것이 목적이다!


```python
import nltk
nltk.download("stopwords")
```


```python
from nltk.corpus import stopwords
```


```python
stop_words = set(stopwords.words("english"))
type(stop_words), len(stop_words)
```




    (set, 179)




```python
from nltk.tokenize import word_tokenize, sent_tokenize
```


```python
filtered_word = []
```


```python
sample = """
The Bible (from Koine Greek τὰ βιβλία, tà biblía, 'the books') is a collection of religious texts or scriptures sacred in Christianity, Judaism, Samaritanism, and many other religions. The Bible is an anthology—a compilation of texts of a variety of forms—originally written in Hebrew, Aramaic, and Koine Greek. These texts include instructions, stories, poetry, and prophesies, among other genres. The collection of materials that are accepted as part of the Bible by a particular religious tradition or community is called a biblical canon. Believers in the Bible generally consider it to be a product of divine inspiration, while understanding what that means in different ways.

The origins of the oldest writings of the Israelites are lost to antiquity. The religious texts were compiled by different religious communities into various official collections. The earliest contained the first five books of the Bible, called the Torah, which was accepted as Jewish canon by the 5th century BCE. A second collection of narrative histories and prophesies was canonized in the 3rd century BCE. A third collection containing psalms, proverbs, and narrative histories, was canonized sometime between the 2nd century BCE and the 2nd century CE.[1] The transmission history of these combined collections spans approximately 3000 years, and there is no scholarly consensus as to when the Jewish Hebrew Bible canon was settled in its present form.[2] Some scholars argue that it was fixed by the Hasmonean dynasty (140–40 BCE),[a] while others argue it was not fixed until the second century CE or even later.[3] The Dead Sea scrolls are approximately dated to 250 BCE–100 CE and are the oldest existing copies of the books of the Hebrew Bible. Tanakh is an alternate term for the Hebrew Bible composed of the first letters of the three parts of the Hebrew scriptures: the Torah ("Teaching"), the Nevi'im ("Prophets"), and the Ketuvim ("Writings"). The Torah is also known as the Pentateuch. The Masoretic Text, in Hebrew and Aramaic, is considered the authoritative text by Rabbinic Judaism; the Septuagint, a Koine Greek translation from the third and second centuries BCE, largely overlaps with the Hebrew Bible.

Christianity began as an outgrowth of Judaism, using the Septuagint as the basis of the Old Testament. The early Church continued the Jewish tradition of writing and incorporating what it saw as inspired, authoritative religious books. The gospels, Pauline epistles and other texts coalesced into the "New Testament" very early. In the first three centuries CE, the concept of a closed canon emerged in response to heretical writings in the second century. The list of books included in the Catholic Bible was established as canon by the Council of Rome in 382, followed by those of Hippo in 393 and Carthage in 397. Christian biblical canons range from the 73 books of the Catholic Church canon, and the 66-book canon of most Protestant denominations, to the 81 books of the Ethiopian Orthodox Tewahedo Church canon, among others.

With estimated total sales of over five billion copies, the Bible is widely considered to be the best-selling publication of all time.[4][5] It has had a profound influence both on Western culture and history and on cultures around the globe.[6] "Simply put, the Bible is the most influential book of all-time."[7] The study of the Bible through biblical criticism has indirectly impacted culture and history as well. The Bible is currently translated or being translated into about half of the world's languages.
"""
```


```python
tokenized_word = word_tokenize(sample)
```


```python
fdist = FreqDist(tokenized_word)
fdist_top10 = fdist.most_common(10)
fdist_top10
```




    [('the', 51),
     (',', 35),
     ('of', 32),
     ('.', 27),
     ('and', 18),
     ('The', 15),
     ('Bible', 14),
     ('in', 13),
     ('is', 10),
     ('a', 9)]




```python
fdist.plot(10, title="Top 10 Most Common Words in Sample")
```


    
![output_88_0](https://user-images.githubusercontent.com/39285147/180379028-cfc18f85-faa9-4d0c-b693-a2544f8457bf.png)
    



```python
for w in tokenized_word:
    if w not in stop_words:
        if len(w) > 3:
            filtered_word.append(w)

fdist = FreqDist(filtered_word)
fdist_top10 = fdist.most_common(10)
fdist_top10
```




    [('Bible', 28),
     ('canon', 16),
     ('books', 14),
     ('Hebrew', 14),
     ('century', 12),
     ('religious', 10),
     ('texts', 10),
     ('collection', 8),
     ('second', 8),
     ('Koine', 6)]




```python
fdist.plot(10, title="Top 10 Most Common Words in Sample")
```


    
![output_90_0](https://user-images.githubusercontent.com/39285147/180378703-5c921d67-37ce-45b8-a7ed-edde873e76b1.png)
    



```python
nltk.download("stopwords", download_dir="./data")
```


## 2-4) Unigrams, Bigrams, Trigrams, ngrams
기계는 한 번에 한 단어를 통과해야 할 때 문장의 의미를 완전히 이해할 수 없다.
- 한 번에 하나씩 주어진 단어를 유니그램이라고 합니다.
- 한 번에 두 개의 단어 -> 바이그램
- 한 번에 세 개의 단어 -> 트라이그램


```python
import nltk
nltk.download("webtext")
```


```python
from nltk.corpus import webtext, stopwords
```


```python
from nltk import bigrams
```


```python
text_words = []

for w in webtext.words("firefox.txt"):
    text_words.append(w)

type(text_words), len(text_words)
```




    (list, 102457)




```python
text_words = [w.lower() for w in webtext.words("firefox.txt")] 

type(text_words), len(text_words)
```




    (list, 102457)




```python
stop_words = set(stopwords.words("english"))
```


```python
filtered_word = []
```


```python
for w in text_words:
    if w not in stop_words:
        if len(w) > 3:
            filtered_word.append(w)

fdist = FreqDist(filtered_word)
fdist.most_common(20)
```




    [('page', 882),
     ('firefox', 879),
     ('window', 647),
     ('bookmarks', 598),
     ('firebird', 583),
     ('open', 576),
     ('menu', 527),
     ('toolbar', 518),
     ('browser', 484),
     ('bookmark', 482),
     ('download', 441),
     ('work', 421),
     ('manager', 408),
     ('crash', 389),
     ('file', 382),
     ('button', 378),
     ('dialog', 369),
     ('text', 327),
     ('crashes', 316),
     ('mozilla', 314)]




```python
fdist.plot(10)
```


    
![output_101_0](https://user-images.githubusercontent.com/39285147/180378509-0b7f8535-5123-43c3-838d-27f45a609411.png)
    



```python
filtered_word[:20]
```




    ['cookie',
     'manager',
     'allow',
     'sites',
     'removed',
     'cookies',
     'future',
     'cookies',
     'stay',
     'checked',
     'full',
     'screen',
     'mode',
     'pressing',
     'ctrl',
     'open',
     'browser',
     'download',
     'dialog',
     'left']




```python
bigrams_list = bigrams(filtered_word)
freq = FreqDist(bigrams_list)
freq.most_common(20)
```




    [(('download', 'manager'), 150),
     (('context', 'menu'), 105),
     (('bookmarks', 'toolbar'), 103),
     (('mozilla', 'firebird'), 96),
     (('right', 'click'), 89),
     (('firefox', 'crashes'), 72),
     (('print', 'preview'), 71),
     (('password', 'manager'), 63),
     (('open', 'tabs'), 61),
     (('browser', 'window'), 56),
     (('default', 'browser'), 52),
     (('tools', 'options'), 50),
     (('middle', 'click'), 47),
     (('home', 'page'), 47),
     (('toolbar', 'folder'), 46),
     (('bookmarks', 'menu'), 46),
     (('bookmark', 'manager'), 44),
     (('bookmark', 'toolbar'), 43),
     (('page', 'info'), 43),
     (('bookmarks', 'manager'), 42)]




```python
freq.plot(10)
```


    
![output_104_0](https://user-images.githubusercontent.com/39285147/180378599-674ae2aa-ad6a-4f9c-89e0-84189f66228d.png)
    


```python
from nltk import trigrams

trigrams_list = trigrams(filtered_word)
fdist = FreqDist(trigrams_list)
```


```python
fdist.most_common(10)
```




    [(('bookmarks', 'toolbar', 'folder'), 23),
     (('full', 'screen', 'mode'), 19),
     (('view', 'page', 'source'), 15),
     (('bookmark', 'toolbar', 'folder'), 14),
     (('right', 'click', 'context'), 14),
     (('middle', 'mouse', 'button'), 13),
     (('click', 'context', 'menu'), 13),
     (('right', 'click', 'menu'), 12),
     (('save', 'link', 'disk'), 12),
     (('download', 'manager', 'open'), 9)]




```python
fdist.plot(10)
```


    
![output_108_0](https://user-images.githubusercontent.com/39285147/180378674-2bfcf9aa-1aef-4394-8de3-1877773589a3.png)
    



```python
from nltk import ngrams

ngrams_list = ngrams(filtered_word, 4)
fdist = FreqDist(ngrams_list)
```


```python
fdist.most_common(10)
```




    [(('right', 'click', 'context', 'menu'), 13),
     (('error', 'launching', 'browser', 'window'), 7),
     (('launching', 'browser', 'window', 'binding'), 6),
     (('browser', 'window', 'binding', 'browser'), 6),
     (('tree', 'view', 'bookmark', 'dialog'), 4),
     (('bookmarks', 'toolbar', 'folder', 'bookmarks'), 3),
     (('click', 'middle', 'mouse', 'button'), 3),
     (('work', 'full', 'screen', 'mode'), 3),
     (('button', 'download', 'font', 'dialog'), 3),
     (('error', 'establishing', 'encrypted', 'connection'), 3)]




```python
ngrams_list = ngrams(filtered_word, 5)
fdist = FreqDist(ngrams_list)
fdist.most_common(10)
```




    [(('error', 'launching', 'browser', 'window', 'binding'), 6),
     (('launching', 'browser', 'window', 'binding', 'browser'), 6),
     (('save', 'link', 'disk', 'context', 'menu'), 3),
     (('allow', 'sites', 'removed', 'cookies', 'future'), 2),
     (('sites', 'removed', 'cookies', 'future', 'cookies'), 2),
     (('shortcut', 'increase', 'text', 'size', 'broken'), 2),
     (('focus', 'page', 'going', 'page', 'bookmark'), 2),
     (('..."', 'right', 'click', 'context', 'menu'), 2),
     (('webpage', 'loads', 'refreshing', 'page', 'continuous'), 2),
     (('loads', 'refreshing', 'page', 'continuous', 'loop'), 2)