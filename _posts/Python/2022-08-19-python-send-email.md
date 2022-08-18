---
layout: single
title: "Python: 이메일 보내기 (SMTP)"
categories: Python
tag: [Python, Email]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
# author_profile: false
header:
    teaser: /assets/images/posts/streamlit-thumbnail.png
sidebar:
    nav: "docs"
---

# 실습 예제: 단순 내용 보내기

본문 작성을 위해 <*https://yeolco.tistory.com/93*> 참조했다.

## 라이브러리

```python
import smtplib # Send Mail Transport Protocol Library
from email.message import EmailMessage
```

# 내용 작성

```python

email = EmailMessage()
# print(dir(email))
email["from"] = "메일을 보내는 자 <fermat39@gmail.com>"
email["to"] = "이메일을 받는자 <fermat39@naver.com>"
email["subject"] = "이메일을 보냅니다!"
email.set_content("""Hello, 
World!
""")
```

# 이메일 보내기

```python
with smtplib.SMTP(host="smtp.gmail.com", port=587) as smtp: # 세션 생성
    smtp.starttls() # TLS 보안 시작
    smtp.login('<지메일 계정>', '<앱 비밀번호>')
    smtp.send_message(email)
    print("메일 발송 완료!")
```

상기 과정에서 주의할 점은 'smtp.login'에서 본인의 비밀번호를 그대로 입력하면 해킹될 위험이 있다.

따라서, '앱 비밀번호'라는 것을 대신 기재해야 한다.

> [앱 비밀번호 얻는 방법](https://yeolco.tistory.com/93)

