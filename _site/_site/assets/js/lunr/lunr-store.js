var store = [{
        "title": "NLP - Part 1: Text Mining and Tokenization",
        "excerpt":"PART 1: Text Mining 1-1) 텍스트 마이닝(Text Mining)이란? 비정형(= 구조화 되지 않은) 텍스트 빅 데이터에 대하여 효과적인 탐색 및 분석을 통해 유용한 정보, 패턴 및 실행 가능한 통찰력을 도출하는 과정이다. ‘Text Mining’ utilizes effective exploration and analysis to produce feasible insights (i.e., valuable patterns) into the extensive dataset of unstructured...","categories": ["NLP"],
        "tags": ["NLP","python"],
        "url": "http://localhost:4000/nlp/nlp-basic-tokenizer/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "NLP - Part 2: Word Embedding",
        "excerpt":"어간추출(Stemmer) vs. 표제어추출(Lemmatizer) Stemmer 단어에서 일반적인 형태 및 굴절 어미를 제거하는 프로세스. A process for removing the commoner morphological and inflexional endings from words. from nltk.stem.porter import PorterStemmer # least strict from nltk.stem.snowball import SnowballStemmer # average (best) from nltk.stem.lancaster import LancasterStemmer # most strict input_words = ['writing', 'calves', 'be',...","categories": ["NLP"],
        "tags": ["NLP","Word Embedding"],
        "url": "http://localhost:4000/nlp/nlp-basic-word-embedding/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "NLP - Part 3: Voice Recognition",
        "excerpt":"PART 1: STT(Speech to Text) !pip install SpeechRecognition !pip install PyAudio import speech_recognition as sr def transform(): r = sr.Recognizer() with sr.Microphone() as source: r.pause_threshold = 0.8 # 0.8초 동안 정적이면 자동으로 음성 녹음을 종료한다 terminate recording in 0.8 seconds of silence said = r.listen(source) # 녹음본 'said'에 저장하기 save...","categories": ["NLP"],
        "tags": ["NLP","Voice Recognition","STT","TTS"],
        "url": "http://localhost:4000/nlp/nlp-voice-recognition/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "ANN - Car Sales Prediction",
        "excerpt":"Code [Notice] download here Learning Goals Artificial Neural Network (ANN)을 이용한 회귀 작업 처리를 이해한다. Understanding of how ANN solves regression tasks. 순방향/역전파를 동반하는 가중치 학습의 과정에 대해 보다 나은 이해를 도모한다. Better understanding of deep learning through forward/backward propagation Description 여러분이 자동차 딜러 혹은 차량 판매원이라 가정해보자. Now, you...","categories": ["SL"],
        "tags": ["ANN","Regression"],
        "url": "http://localhost:4000/sl/projects-1/",
        "teaser": "http://localhost:4000/assets/images/posts/dl-thumbnail.jpg"
      },{
        "title": "Deep Learning - CIFAR-10 Classification",
        "excerpt":"Code [Notice] download here Learning Goals 합성곱 신경망 모델 설계하여 케라스로 이미지 분류 Building the CNN and Image classification using keras Adam 옵티마이저로 신경망 가중치 최적화 Optimizing weights using ‘Adam’ 드롭아웃을 통한 과적합 개선 Drop-out 모델 평가 진행 (confusion matrix) Model evaluation Image Augmentation으로 신경망 일반화 성능 개선 Improving generalization...","categories": ["SL"],
        "tags": ["CNN","Classification","CIFAR-10"],
        "url": "http://localhost:4000/sl/projects-2/",
        "teaser": "http://localhost:4000/assets/images/posts/dl-thumbnail.jpg"
      },{
        "title": "Profit Time Series",
        "excerpt":"Code [Notice] download here Learning Goals 시계열 예측을 위한 페이스북 Propjet 이해 Understanding Facebook Propjet for Time Series Prediction PART 1: Chicago Crime Rate Description 절도범이 어느 시간대에 가장 잘 잡히는지, 범죄율이 올라가는 가장 높은 시간대는 언제인지 등을 관찰해보고 Prophet 활용하여 미래 ‘Crime’ 결과도 예측해본다. Observing when thieves are best...","categories": ["SL"],
        "tags": ["LSTM","Prophet","Time Series"],
        "url": "http://localhost:4000/sl/projects-3/",
        "teaser": "http://localhost:4000/assets/images/posts/dl-thumbnail.jpg"
      },{
        "title": "LeNet - Traffic Signs Classification",
        "excerpt":"Code [Notice] download here Observing the dataset Classes are as listed below: ( 0, b’Speed limit (20km/h)’) ( 1, b’Speed limit (30km/h)’) ( 2, b’Speed limit (50km/h)’) ( 3, b’Speed limit (60km/h)’) ( 4, b’Speed limit (70km/h)’) ( 5, b’Speed limit (80km/h)’) ( 6, b’End of speed limit (80km/h)’) (...","categories": ["SL"],
        "tags": ["LeNet","Classification"],
        "url": "http://localhost:4000/sl/projects-4/",
        "teaser": "http://localhost:4000/assets/images/posts/dl-thumbnail.jpg"
      },{
        "title": "NLP Basic",
        "excerpt":"Code [Notice] download here Learning Goals Naive Beyas Theorem 베이즈 정리에 기반한 분류 기술 Classification techniques with beyas theorem 자연어 처리의 기초 Understanding NLP tokenization with NLTK Extracting features with Count Vectorizer 우도, 사전 확률, 주변 우도의 차이점 Likelihood, prior, marginal likelihood 불균형 데이터 처리 방법 How to handle unbalanced...","categories": ["NLP"],
        "tags": ["NLP","python"],
        "url": "http://localhost:4000/nlp/projects-5/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "사용자 기반 협업 필터링 (Collaborative Filtering) - 영화 추천 시스템 (Movie Recommender Systems)",
        "excerpt":"Code [Notice] download here Description 추천 시스템이란 가령, 아마존에서 제품을 구매하면 이 제품을 구매하는 고객들이 다른 상품에도 관심이 있을 것이라 판단해 추천해주는 방법이다. Providing suggestions for items that are most pertinent to a particular user ‘아이템’을 기반으로 사용자에게 추천을 제공한다. Item-based recommendation systems, not user-based 이게 무슨 말인고 하니, 세상...","categories": ["Others"],
        "tags": ["Correlation","Movie Recommendation"],
        "url": "http://localhost:4000/others/ml-projects-6/",
        "teaser": "http://localhost:4000/assets/images/posts/ml-thumbnail.jpg"
      },{
        "title": "AI 학습자료",
        "excerpt":"AI 대학원 전공 면접 질문 모음: Preparing for AI Graduate School Math, Statistics Linear Algebra(KAIST) or Linear Algebra(YouTube) Prababilities and Statistics Programming &amp; Data Analysis Python R AI AI Machine Learning ML(SNU) or ML(Stanford) Reinforcemnet Learning(Stanford) or Reinforcemnet Learning(Stanford) Deep Learning Deep Learning(MIT)) CNN(Stanford) Computer Vision(Georgia Tech) NLP(Stanford) Deep NLP(Stanford)...","categories": ["Star"],
        "tags": ["AI","KAIST","SNU","Graduate School"],
        "url": "http://localhost:4000/star/ai-study-guide/",
        "teaser": "http://localhost:4000/assets/images/posts/data-thumbnail.jpg"
      },{
        "title": "NLP - Part 4: Naive Classifier",
        "excerpt":"PART 1: Gender Identifier Corpus의 영어 이름에서 마지막 ‘몇 글자’를 보고 남자 혹은 여자 이름인지 확인하여 분류한다. Identifying sex through the last ‘few letters’ of the corpus Naive Bayes Classifier 모든 고유한 단어가 텍스트에서 추출 extracting words from texts 라벨 확인 checking labels 분류 classification Naive Beyas Loading the libraries...","categories": ["NLP"],
        "tags": ["NLP","Naive Classifier"],
        "url": "http://localhost:4000/nlp/nlp-basic-naive-beyas/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "NLP - Part 5: Topic Modeling",
        "excerpt":"What is Topic Modeling? 주제 모델링(Topic Modeling)이란 주제에 해당하는 텍스트 데이터의 패턴을 식별하는 과정이다. Topic Modeling is the process of identifying a pattern of text data corresponding to a topic. 텍스트에 여러 주제가 포함된 경우 이 기술을 사용하여 입력 텍스트 내에서 해당 주제를 식별하고 분리할 수 있다. If the text...","categories": ["NLP"],
        "tags": ["NLP","Topic Modeling"],
        "url": "http://localhost:4000/nlp/nlp-basic-topic-modeling/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "NLP: Transformer - GPT-2",
        "excerpt":"Transformer w/ Tensorflow 번역, 질문 답변, 텍스트 요약 등과 같은 작업을 위해 순차적으로 데이터를 처리하도록 설계된 2017년에 구글이 소개한 딥러닝 아키텍처이다. It is a deep learning architecture introduced by Google in 2017 designed to process data sequentially for tasks such as translation, question answering, text summarization, and more. 텍스트 생성기로...","categories": ["GPT"],
        "tags": ["NLP","GPT","Transformer"],
        "url": "http://localhost:4000/gpt/nlp-basic-transformer/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "[논문 분석] Learning loss for active learning (CVPR 2019)",
        "excerpt":"Learning loss for active learning 논문 Background We need big dataset to train a deep learning model Low cost: image collection (unlabeled dataset) High cost: image annotation (labeled dataset) Semi-Supervised Learning vs. Active Learning Semi-Supervised Learning How to train unlabeled dataset based on labeled data Active Learning What data to...","categories": ["AIPaperCV"],
        "tags": ["AI Research Papers","CV","NLP"],
        "url": "http://localhost:4000/aipapercv/ai-paper-learning-loss-for-active-learning/",
        "teaser": "http://localhost:4000/assets/images/posts/ai-thumbnail.jpg"
      },{
        "title": "CNN - Classification: MNIST with Tensorflow",
        "excerpt":"Code [Notice] download here Observing the dataset MNIST 데이터셋에는 0부터 9까지 10종류의 숫자가 여러 2차원 이미지 데이터 형태로 저장되어 있으며, RGB 차원은 따로 없는 흑백사진이다. In the MNIST dataset, 10 types of numbers from 0 to 9 are stored in the form of several two-dimensional image data, and it is...","categories": ["SL"],
        "tags": ["CNN","MNIST","Tensorflow","PyTorch"],
        "url": "http://localhost:4000/sl/dl-mnist/",
        "teaser": "http://localhost:4000/assets/images/posts/dl-thumbnail.jpg"
      },{
        "title": "RNN - Time Series: LSTM",
        "excerpt":"[Notice] Reference RNN 다른 딥러닝 모델처럼 은닉층 (Hidden Layer)을 제공할 뿐만 아니라, 자체적으로 지원하는 임시적인 루프가 존재한다; 시간이라는 차원이 추가된다. Not only does it provide a hidden layer like other deep learning models, but it also has its own ad hoc loops; A dimension of time is added. 따라서, RNN은...","categories": ["SL"],
        "tags": ["RNN","LSTM","Time Series"],
        "url": "http://localhost:4000/sl/dl-rnn-lstm/",
        "teaser": "http://localhost:4000/assets/images/posts/dl-thumbnail.jpg"
      },{
        "title": "NLP: Data Preprocessing",
        "excerpt":"Loading the libraries and dataset import pandas as pd import numpy as np raw_data = pd.read_csv(\"Corona_NLP_train.csv\", encoding = \"latin-1\") # load the dataset raw_data.shape # check the shape raw_data.head() # view the dataset briefly raw_data.drop([\"UserName\", \"ScreenName\", \"Location\", \"TweetAt\"], axis = 1) # remove unnecessary columns raw_data = raw_data [ [\"OriginalTweet\",...","categories": ["NLP"],
        "tags": ["NLP","Data Preprocessing"],
        "url": "http://localhost:4000/nlp/nlp-basic-text-classification/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      },{
        "title": "Data Preprocessing Techniques",
        "excerpt":"‘raw_data’ is the temporary dataset, and we are going to address various preprocessing tasks with it. 데이터 확인 (Describing the dataset) raw_data.info() &lt;class 'pandas.core.frame.DataFrame'&gt; Int64Index: 23 entries, 0 to 29 Data columns (total 8 columns): # Column Non-Null Count Dtype --- ------ -------------- ----- 0 customer_id 23 non-null int64 1...","categories": ["Study"],
        "tags": ["AI","Data Preprocessing"],
        "url": "http://localhost:4000/study/ai-data-preprocessing/",
        "teaser": "http://localhost:4000/assets/images/posts/data-thumbnail.jpg"
      },{
        "title": "Python: PART 1 - Web Application without Server",
        "excerpt":"파이썬으로 웹 서버 구동없이 웹 어플리케이션을 만들기 위해 ‘streamlit’ 라이브러리가 활용된다. To create a web application without a server, we use the ‘streamlit’ library. Loading the library for web server import streamlit as st Texts st.write(\"Hello, World!\") # print out the message to the web page st.markdown(\"\"\"This is an H1=============\"\"\")...","categories": ["Python"],
        "tags": ["Python Web App","Streamlit"],
        "url": "http://localhost:4000/python/python-web-server/",
        "teaser": "http://localhost:4000/assets/images/posts/streamlit-thumbnail.png"
      },{
        "title": "[논문 분석] Deep Learning-Based Vehicle Anomaly Detection by Combining Vehicle Sensor Data (KAIS 2021)",
        "excerpt":"Deep Learning-Based Vehicle Anomaly Detection by Combining Vehicle Sensor Data 논문 PART 1: Background 기존 이상탐지 방법은 제한된 데이터를 다루는 전통적인 통계 방법에 의존한다. 이 논문은 **AI 기반 보다 효과적인 이상탐지 방법을 제안한다. 자동차의 공회전 센서 및 **이상 탐지 간의 상관관계를 분석하여 인공지능 모델을 설계하였다. 기존의 SVM이나 PCA 모델 등은...","categories": ["AIPaperOthers"],
        "tags": ["AI Research Paper","Self-driving"],
        "url": "http://localhost:4000/aipaperothers/ai-paper-dl-behicle-detection-by-sensor/",
        "teaser": "http://localhost:4000/assets/images/posts/ai-thumbnail.jpg"
      },{
        "title": "Part 1: BERT Language Model",
        "excerpt":"자연어 처리(Natural Language Processing, NLP) 자연어란? 부호화(Encoding) 해독(Decoding) 자연어 처리는 상기 도표에서 컴퓨터가 텍스트를 해독하는 과정을 의미한다. 일상에서 사용하는 모든 인간의 언어로, 한국어, 영어와 같은 것들이 예시이다. 인공언어: 프로그래밍 언어, etc. 자연어 처리의 두 종류 규칙 기반 접근법 (Symbolic approach) 확률 기반 접근법 (Statistical approach) TF-IDF TF(Term frequency): 단어가 문서에...","categories": ["BERT"],
        "tags": ["NLP","BERT","Language Model"],
        "url": "http://localhost:4000/bert/bert-1/",
        "teaser": "http://localhost:4000/assets/images/posts/bert-thumbnail.png"
      },{
        "title": "Part 2: BERT Language Model",
        "excerpt":"언어 모델 (Language Model, LM) ‘자연어’의 법칙을 컴퓨터로 모사하는 모델로, 다음에 등장할 단어 예측을 수행한다(= ‘맥락’을 고려한다). Markov 확률 모델 이전 단어의 형태를 통하여 확률적으로 다음에 나올 단어를 예측한다 가령, ‘like’ 다음에 ‘rabbit’이라는 단어가 나타날 확률은 주어진 학습 데이터에 기반하여 33%로 나타난다. RNN (Recurrent Neural Network) 모델 Markov 체인 모델을...","categories": ["BERT"],
        "tags": ["NLP","BERT","Language Model"],
        "url": "http://localhost:4000/bert/bert-2/",
        "teaser": "http://localhost:4000/assets/images/posts/bert-thumbnail.png"
      },{
        "title": "Part 3: BERT Language Model",
        "excerpt":"BERT Bi-directional transformer로 이루어진 언어모델로, BERT 언어모델 위에 1개의 classification layer만 부착하여 다양한 NLP task를 수행한다. WordPiece tokenizing 입력 문장을 toeknizing하고, 그 token들로 ‘token sequence’를 만들어 학습에 사용한다. BPE와는 다르게 WordPiece는 우도로 병합을 진행하여, 두 문자가 같이 오는 문자 단위를 중요시한다. 우도: 전체 글자 중 각 단어가 따로 등장한 것을...","categories": ["BERT"],
        "tags": ["NLP","BERT","Language Model"],
        "url": "http://localhost:4000/bert/bert-3/",
        "teaser": "http://localhost:4000/assets/images/posts/bert-thumbnail.png"
      },{
        "title": "[논문 분석] A Multi-Task Benchmark for Korean Legal Language Understanding and Judgement Prediction (arXiv 2022)",
        "excerpt":"논문 들어가면서 법률 계약서는 일반인이 독해하기 어려운 단어들 뿐만 아니라, 한 문장이 한 페이지를 차지할 정도로 긴 문장들을 포함한다. 이전에 참가한 한 세미나에서는 이러한 법률 계약서에 존재하는 오류를 검사하는 AI 모델을 실제 변호사와 대결시킨 사례를 소개했다. AI 모델은 26초 만에 94%의 정확도로 오류를 검증해내었고, 반면 사람 변호사는 96분 동안 86%의...","categories": ["AIPaperNLP"],
        "tags": ["AI Research Paper","Kaist"],
        "url": "http://localhost:4000/aipapernlp/ai-paper-ko-legal-nlp/",
        "teaser": "http://localhost:4000/assets/images/posts/ai-thumbnail.jpg"
      },{
        "title": "Python: PART 2 - Web Application without Server",
        "excerpt":"강아지 품종 분류 AI 웹페이지 % pip install opencv-python # Loading the libraries from distutils.command.install_egg_info import to_filename import numpy as np import streamlit as st import cv2 # *opencv import tensorflow as tf from tensorflow import keras 상기 라이브러리 중에 눈에 띄는 것이 있다; cv2 cv2 라이브러리는 opencv 패키지를 설치해서...","categories": ["Python"],
        "tags": ["Python Web App","Streamlit","Opencv"],
        "url": "http://localhost:4000/python/python-web-server-exercise/",
        "teaser": "http://localhost:4000/assets/images/posts/streamlit-thumbnail.png"
      },{
        "title": "LeNet 신경망 - MiniPlaces 이미지 분류",
        "excerpt":"LeNet 신경망을 활용해서 MiniPlaces 데이터셋 이미지 분류 작업을 수행한다. ‘MiniPlaces’ 데이터셋은 캐글과 같은 온라인에서 손쉽게 구할 수 있다 여기. Code [Notice] download here CNN이나 LeNet 신경망에 대한 보다 자세한 내용은 여기를 참조하자. 라이브러리 불러오기 # python imports import os from tqdm import tqdm # torch imports import torch import torch.nn...","categories": ["SL"],
        "tags": ["LeNet","Classification","PyTorch"],
        "url": "http://localhost:4000/sl/dl-miniplace-classification/",
        "teaser": "http://localhost:4000/assets/images/posts/lenet.JPG"
      },{
        "title": "A* Search - 8-tile Puzzle Game",
        "excerpt":"A-star 알고리즘을 활용해서 8-tile Puzzle 게임을 구현한다.      8-tile Puzzle 정보는 여기서 참고하자.    이 게임에서 당신은 AI 로봇과 Teeko 보드 게임을 펼치게 될 것이다.   이 프로젝트는 A* 탐색 알고리즘을 이해하고 있다는 전제로 진행한다.      A* 탐색 알고리즘이란?    Code  [Notice] download here   ","categories": ["Others"],
        "tags": ["Machine Learning","Game","A* Search"],
        "url": "http://localhost:4000/others/ml-8-tile-puzzle-a-search/",
        "teaser": "http://localhost:4000/assets/images/posts/8-tile-puzzle.png"
      },{
        "title": "Hierarchical Agglomerate Clustering(HAC) - 포켓몬 군집화",
        "excerpt":"Hierarchical Agglomerate Clustering(HAC) 이용해서 서로 다른 특성을 공유하는 Pokemon들을 군집으로 묶어보자. ‘Pokemon.csv’는 온라인에서 손쉽게 구할 수 있다. 이 프로젝트는 군집화(Clustering)의 개념을 숙지하고 있다는 전제로 수행한다. Code [Notice] download here 데이터셋 관찰 Columns: Attack, Sp. Atk, Speed, Defense, Sp. Def, and HP 라이브러리 불러오기 from scipy.cluster.hierarchy import dendrogram, linkage import csv...","categories": ["USL"],
        "tags": ["HAC","Clustering","Pokemon"],
        "url": "http://localhost:4000/usl/ml-hca/",
        "teaser": "http://localhost:4000/assets/images/posts/hac.png"
      },{
        "title": "PCA - Image Compression(이미지 압축)",
        "excerpt":"PCA를 이용해서 이미지 압축을 진행해보자. Code [Notice] download here 이번 프로젝트는 PCA를 활용해서 이미지 압축 문제를 해결한다. 이론적으로 PCA의 개념을 이해하고 있다는 전제로 프로젝트를 수행한다. PCA 개념 숙지는 여기! 데이터 불러오기 from scipy.linalg import eigh import numpy as np import matplotlib.pyplot as plt def load_and_center_dataset(filename): f = np.load(filename) dc =...","categories": ["USL"],
        "tags": ["PCA","Image Compression"],
        "url": "http://localhost:4000/usl/ml-pca/",
        "teaser": "http://localhost:4000/assets/images/posts/pca-image-compression.png"
      },{
        "title": "Reinforcement Learning - Q-Learning 알고리즘",
        "excerpt":"Q-Learning 알고리즘을 활용한 예제이다.   이 프로젝트는 Q-Learning 알고리즘을 이해하고 있다는 전제로 진행한다.   Code  [Notice] download here   ","categories": ["RL"],
        "tags": ["Reinforcement Learning","Q-learning"],
        "url": "http://localhost:4000/rl/ml-reinforcement-learning-q-learning/",
        "teaser": "http://localhost:4000/assets/images/posts/q-learning.png"
      },{
        "title": "Minimax 알고리즘 - Teeko Game",
        "excerpt":"Minimax 알고리즘을 활용해서 Teeko 게임을 구현한다. Teeko 게임은 한국인들에게는 익숙하지 않은 보드 게임일 수도 있다 (하기 설명 참조). Teeko Game이란? It is a game between two players on a 5x5 board. Each player has four markers of either red or black. Beginning with black, they take turns placing markers (the...","categories": ["Others"],
        "tags": ["Machine Learning","Teeko Board Game","Minimax"],
        "url": "http://localhost:4000/others/ml-teeko-minimax/",
        "teaser": "http://localhost:4000/assets/images/posts/teeko.png"
      },{
        "title": "NLP - Part 7: Name Entity Recognition (NER) 앱",
        "excerpt":"사전 설정 % pip install spacy spacy_streamlit % python -m spacy download en_core_web_trf 1) 하나하나 직접 입력! -&gt; 노동력이 필요 -&gt; 시간도 필요 -&gt; 실수가 있을 수 있다. -&gt; 입력 도중에 값이 변할 수도 있다. 2) Copy &amp; Paste 로 할 때 총 1640번의 반복작업을 해야한다. 붙여넣기 하고 정리가 필요하다!...","categories": ["NLP"],
        "tags": ["NLP","NER"],
        "url": "http://localhost:4000/nlp/nlp-NER/",
        "teaser": "http://localhost:4000/assets/images/posts/nlp-thumbnail.jpg"
      }]
