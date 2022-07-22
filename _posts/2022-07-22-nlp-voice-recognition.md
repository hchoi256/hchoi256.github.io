---
layout: single
title: "NLP - Part 3: Voice Recognition"
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

# Voice Recognition
## PART 1: SST(Speech to Text)

```python
!pip install SpeechRecognition
!pip install PyAudio

import speech_recognition as sr
```

```python
def transform():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.pause_threshold = 0.8 # 0.8초 동안 정적이면 자동으로 음성 녹음을 종료한다
        said = r.listen(source) # 녹음본 'said'에 저장하기
        try:
            q = r.recognize_google(said, language="ko") # google 언어팩 사용
            return q
        except sr.UnknownValueError:
            return "무슨 말인지 이해를 잘 못 했어요"
        except:
            return "대기중입니다."
```


```python
transform() # STT 작업 시작
```

    마이크를 켜고, 음성을 녹음하면 메시지로 출력되는 것을 보실 수 있습니다.

## PART 2: TTS(Text to Speech)

```python
!pip install pyttsx3 # TTS 클래스
```

```python
import pyttsx3

def speaking(message):
    engine = pyttsx3.init() # TTS 엔진 가동
    engine.say(message) # 메시지 내용 엔진에 전달
    engine.runAndWait() # 메시지 음성출력
```


```python
speaking("이제 STT가 준비가 되었습니다. 계속 해서 Voice Assistance를 만들어 보겠습니다.")
```

    스피커에서 상기 메시지가 출력되는 것을 보실 수 있습니다.


### TTS 실습 1: 요일/시간 음성 출력하기

```python
def query_day():
    day = datetime.date.today()
    weekday = day.weekday()
    week_mapping = {
        0: "월요일",
        1: "화요일",
        2: "수요일",
        3: "목요일",
        4: "금요일",
        5: "토요일",
        6: "일요일",
    }

    speaking(f"오늘은 {week_mapping[weekday]}입니다. {week_mapping[weekday]}에도 공부하느라 고생이시네요!" )
```


```python
query_day()
```

    오늘 요일이 음성 출력되는 것을 보실 수 있습니다.


```python
def query_time():
    time = datetime.datetime.now().strftime("%H:%M:%S")
    speaking(f"현재는 {time[:2]}시 {time[3:5]}분입니다.")
```


```python
query_time()
```

    현재 시간이 음성 출력되는 것을 보실 수 있습니다.


### TTS 실습 2: STT + TTS - 마이크 출력을 다시 읊어주기

```python
import webbrowser
```


```python
while (True):
    q = transform()

    if  "무슨 요일" in q:
        query_day()
        continue
    elif "몇 시" in q:
        query_time()
        continue 
    elif "유튜브 시작" in q:
        speaking("유튜브를 시작하겠습니다. 잠시만 기다려주세요!")
        webbrowser.open("https://www.youtube.com")
    elif "네이버 시작" in q:
        speaking("네이버를 시작하겠습니다. 잠시만 기다려주세요!")
        webbrowser.open("https://www.naver.com")
    elif "이동" in q:
        speaking("윈도우키를 실행하겠습니다. 잠시만 기다려주세요!")
        pyautogui.moveTo(700, 1050, 3)
        pyautogui.click(button="left")        
    elif "이제 그만" in q:
        speaking("아쉽지만 다음에 또 뵙겠습니다")
        break
```

    가령, '네이버 시작'이라는 마이크 입력을 넣으면, '네이버 시작'이라는 말이 스피커로 반향되며 webbrowser 라이브러리를 통하여 네이버를 브라우저에서 띄워준다.



## Bonus: Keyboard I/O

```python
!pip install pyautogui
```


```python
import pyautogui
```


```python
screenWidth, screenHeight = pyautogui.size() # 현재 화면 해상도

pyautogui.moveTo(700, 1050, 3) # # 마우스 커서를 x, y 좌표로 이동
pyautogui.click(button="left") # 해당 위치 버튼을 좌클릭
```