---
layout: single
title: "[개발] 긴 영문 글/기사 요약번역 웹 페이지 구현하기"
categories: NLP
tag: [NLP, Text Summarization, Translator, Streamlit]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/nlp-thumbnail.jpg
sidebar:
    nav: "docs"
---

# style.css
```python
# 폰트 설정
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap'); 

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
    font-weight: 500;
    color: #091747;
}
```

# 텍스트 요약
텍스트 내용에서 가장 많이 등장한 단어들을 포함하는 문장을 취합해서 최종 요약본으로 제시한다.

```python
import bs4 as bs # html 컨트롤
import urllib.request # url 접근
import re # regex 
import nltk
```

```python
class TextSummarization():
    def __init__(self, url='https://en.wikipedia.org/wiki/Korea'): # default url                     
        # 단락 가져오기
        scraped_data = urllib.request.urlopen(url) # url 접근 
        article = scraped_data.read() # url 데이터 가져오기
        parsed_article = bs.BeautifulSoup(article,'lxml') # lxml 형식으로 파싱
        self.paragraphs = parsed_article.find_all('p') # 문단 가져오기

    def summarize(self):
        # 요약
        article_text = ""

        for p in self.paragraphs:
            article_text += p.text # article 내용
        # Removing Square Brackets and Extra Spaces
        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)
        # Removing special characters and digits
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
        sentence_list = nltk.sent_tokenize(article_text) # 문장 토크나이저
        stopwords = nltk.corpus.stopwords.words('english') # 영어 불용어

        word_frequencies = {} # 단어 등장 빈도
        for word in nltk.word_tokenize(formatted_article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
            maximum_frequncy = max(word_frequencies.values()) # 최다 단어 빈도수
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy) # 빈도 확률 계산
        sentence_scores = {} # 문장별 단어 빈도 확률 총합
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30: # 문장 단어 30개 미만
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        # 요약 방식 처리
        sent_max = len(sentence_scores.keys())
        res = 0
        if sent_n > 0:
            if sent_max > sent_n:
                sent_max = sent_n # advanced
            else:
                res = 1
        else:
            if sent_max > 7:
                sent_max = 7 # default
                
        import heapq
        summary_sentences = heapq.nlargest(sent_max, sentence_scores, key=sentence_scores.get) # 우선순위 큐에서 문장 단어 빈도 확률이 가장 높은 7개의 문장 가져옴 

        summary = ' '.join(summary_sentences)
        return (summary)
```

# 웹 페이지 (Streamlit)
```python
import streamlit as st
import validators
import os
import text_summarization as ts
from googletrans import Translator
```

```python
# Preparations

# 폰트 설정
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "css/style.css")) as css:
    st.markdown( f'<style>{css.read()}</style>', unsafe_allow_html= True)

# 사이드 바 크기
# st.markdown(f'<style>.sidebar .sidebar-content {{width: 100px;}}</style>', unsafe_allow_html=True)

# 요약 방식 설정
sidebar_radio = st.sidebar.radio("How to Summarize", ["Simple", "Advanced"])
summ_opt = 0
if sidebar_radio == "Advanced":
    summ_opt = 15
```

```python
# Helper Function
def url_val(url):
    # URL 유효성 검사
    if validators.url(url) != True:
        return -1
    return 0
```

```python
# Web Implementation
st.title("Article Summarizer & Online Translating Tool (ENG/KOR)") # title
st.markdown("---") # division

# url 텍스트 박스 생성
url = st.text_area(
    label = "Press your URL here.",
    # value = "https://en.wikipedia.org/wiki/Korea", # text_input의 초기값
    max_chars = 500,
    height = 10,
    placeholder= "Press Ctrl + Enter to apply." # default
)

if url_val(url) == 0:
    print("URL accepted: {}".format(url))        
    summary, res = ts.TextSummarization(url).summarize(summ_opt)
    
    # 요약
    st.text_area(
    label = "Summary",
    value = summary, # text_input의 초기값
    max_chars = 10000,
    height = 200,
    )
    
    # 번역
    translator = Translator()
    
    st.text_area(
    label = "Translated",
    value = translator.translate(summary, src="en", dest="ko").text, # text_input의 초기값
    max_chars = 10000,
    height = 200,
    )
else:
    if url != "":
        st.warning('Please enter a valid Website URL (e.g., https://en.wikipedia.org/wiki/Korea).')
        
st.markdown("---") # division
st.markdown("Made by Eric Choi")
```

# 결과

아래는 [South Korea](https://en.wikipedia.org/wiki/Korea) 위키피디아 기사를 요약번역 해본 예시이다.

![image](https://user-images.githubusercontent.com/39285147/186647668-865e9aec-1f43-4e7b-9855-d3b5e9e81d0d.png)

장문의 기사가 몇 줄로 요약되고, 그 번역본 또한 손쉽게 확인할 수 있다.

![image](https://user-images.githubusercontent.com/39285147/186647810-cc1edba2-a7c7-4352-9cb9-2eea07b5250d.png)

요약 설정은 간단하게 **Simple, Advanced** 옵션을 선택할 수 있다.

Advanced 옵션은 더 많은 중요 문장 내용을 포함하는 요약본이다.

# References
[Text Summarization](https://www.kaggle.com/code/imkrkannan/text-summarization-with-nltk-in-python/notebook)