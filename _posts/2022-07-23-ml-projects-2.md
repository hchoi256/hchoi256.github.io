---
layout: single
title: "ML Project 1: ANN - Car Sales Prediction"
categories: Machine Learning
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

# Car Sales Prediction

## Learning Goals
Artificial Neural Network (ANN)을 이용한 회귀 작업 처리를 이해한다.

순방향/역전파를 동반하는 가중치 학습의 과정에 대해 보다 나은 이해를 도모한다.

## Description
여러분이 자동차 딜러 혹은 차량 판매원이라 가정하고, 상기 고객들의 특정 데이터(나이, 연봉, etc.)를 참고하여 고객들이 차량 구매에 사용할 금액을 예측하여 특정 집단에 대한 타깃 마케팅을 이루고자 한다.

### Dataset
<table border="0" cellpadding="0" cellspacing="0" id="sheet0" class="sheet0 gridlines">
    <col class="col0">
    <col class="col1">
    <col class="col2">
    <col class="col3">
    <col class="col4">
    <col class="col5">
    <col class="col6">
    <col class="col7">
    <col class="col8">
    <tbody>
        <tr class="row0">
        <td class="column0 style0 s">Customer Name</td>
        <td class="column1 style0 s">Customer e-mail</td>
        <td class="column2 style0 s">Country</td>
        <td class="column3 style0 s">Gender</td>
        <td class="column4 style0 s">Age</td>
        <td class="column5 style0 s">Annual Salary</td>
        <td class="column6 style0 s">Credit Card Debt</td>
        <td class="column7 style0 s">Net Worth</td>
        <td class="column8 style0 s">Car Purchase Amount</td>
        </tr>
        <tr class="row1">
        <td class="column0 style0 s">Martina Avila</td>
        <td class="column1 style0 s">cubilia.Curae.Phasellus@quisaccumsanconvallis.edu</td>
        <td class="column2 style0 s">Bulgaria</td>
        <td class="column3 style0 n">0</td>
        <td class="column4 style0 n">41.8517198</td>
        <td class="column5 style0 n">62812.09301</td>
        <td class="column6 style0 n">11609.38091</td>
        <td class="column7 style0 n">238961.2505</td>
        <td class="column8 style0 n">35321.45877</td>
        </tr>
        <tr class="row2">
        <td class="column0 style0 s">Harlan Barnes</td>
        <td class="column1 style0 s">eu.dolor@diam.co.uk</td>
        <td class="column2 style0 s">Belize</td>
        <td class="column3 style0 n">0</td>
        <td class="column4 style0 n">40.87062335</td>
        <td class="column5 style0 n">66646.89292</td>
        <td class="column6 style0 n">9572.957136</td>
        <td class="column7 style0 n">530973.9078</td>
        <td class="column8 style0 n">45115.52566</td>
        </tr>
        <tr class="row3">
        <td class="column0 style0 s">Naomi Rodriquez</td>
        <td class="column1 style0 s">vulputate.mauris.sagittis@ametconsectetueradipiscing.co.uk</td>
        <td class="column2 style0 s">Algeria</td>
        <td class="column3 style0 n">1</td>
        <td class="column4 style0 n">43.15289747</td>
        <td class="column5 style0 n">53798.55112</td>
        <td class="column6 style0 n">11160.35506</td>
        <td class="column7 style0 n">638467.1773</td>
        <td class="column8 style0 n">42925.70921</td>
        </tr>
        <tr class="row4">
        <td class="column0 style0 s">Jade Cunningham</td>
        <td class="column1 style0 s">malesuada@dignissim.com</td>
        <td class="column2 style0 s">Cook Islands</td>
        <td class="column3 style0 n">1</td>
        <td class="column4 style0 n">58.27136945</td>
        <td class="column5 style0 n">79370.03798</td>
        <td class="column6 style0 n">14426.16485</td>
        <td class="column7 style0 n">548599.0524</td>
        <td class="column8 style0 n">67422.36313</td>
        </tr>
        <tr class="row5">
        <td class="column0 style0 s">Cedric Leach</td>
        <td class="column1 style0 s">felis.ullamcorper.viverra@egetmollislectus.net</td>
        <td class="column2 style0 s">Brazil</td>
        <td class="column3 style0 n">1</td>
        <td class="column4 style0 n">57.31374945</td>
        <td class="column5 style0 n">59729.1513</td>
        <td class="column6 style0 n">5358.712177</td>
        <td class="column7 style0 n">560304.0671</td>
        <td class="column8 style0 n">55915.46248</td>
        </tr>
    </tbody>
</table>

**독립변수**
- Customer Name
- Customer e-mail
- Country
- Gender
- Age
- Annual Salary
- Credit Card Debt.
- Net Worth

**종속변수**
- Car Purchase Amount

## Import Dataset

```python
import pandas as pd # 데이터 프레임 조작
import numpy as np # 수치 해석
import matplotlib.pyplot as plt # 그래프 시각화
import seaborn as sns # 그래프 시각화

car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1') # 데이터셋이 '@'와 같은 특수문자를 포함하기 때문에 상기 인코딩 설정을 해줘야한다.
```

## Data Visualization

### Seaborn

```python
sns.pairplot(car_df) # 씨본 덕분에 분석 작업을 여러 번 할 필요없이 여러 종류의 시각화를 보여준다
```

![screensht](https://user-images.githubusercontent.com/39285147/180380848-d3772ba0-a21b-416b-8139-534c7a3aa721.JPG)

데이터 분포에서 맨 아래 위치한 행은 'Car Purchase Amount'이고, 각 열은 순서대로 Gender, Age, Annual Salary, Credit Car Debt, Net Worth, Car Purchase Amount이다.

따라서, 나이가 증가함에 따라 차량 구매 예상 금액이 증가하는 선형적 형태의 데이터 분포를 보여주고, 반대로 Credit Card Debt은 종속변수와 뚜렷한 상관관계를 나타내지 않는 것으로 관찰된다.

## Data Preprocessing
## Remove Unnecessary Variables
```python
X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1) # 종속변수에 영향을 끼치지 않는 불필요한 입력피처를 제거한다.
y = car_df['Car Purchase Amount'] # 종속변수

X # 정제된 훈련 데이터 관찰
```

![image](https://user-images.githubusercontent.com/39285147/180381916-2051d577-51ec-4ff8-be80-0685754f456b.png)

### Data Scaling
나이와 연봉과 같은 입력피처의 수치가 차이가 커서, 특정 피처에 과중화된 결과가 나올 수 있으므로 [0, 1] 값으로 정규화하는 스케일링(Scailing)을 적용해야 한다.

이번 프로젝트에서, 우리는 **MinMaxScaler**를 사용한다.

기존 StandardScaler와 MinMaxScaler의 차이점은 데이터가 **정규분포를 따르는지 혹은 따라야 하는지**에 달려있다.

[참고](https://velog.io/@ljs7463/%ED%94%BC%EC%B2%98-%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81StandardScalerMinMaxScaler)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

```

## Model Training

**Dense**
- *첫번째 인자* : 출력 뉴런의 수를 설정합니다.
- *input_dim* : 입력 뉴런의 수를 설정합니다.
- *init* : 가중치 초기화 방법 설정합니다.
  - 'uniform' : 균일 분포
  - 'normal' : 가우시안 분포
- *activation* : 활성화 함수 설정합니다.
  - 'linear' : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
  - 'relu' : rectifier 함수, 은익층에 주로 쓰입니다.
  - 'sigmoid' : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
  - 'softmax' : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25) # Create training and test set

import tensorflow.keras
from keras.models import Sequential # 신경망을 순차적 형태로 설계
from keras.layers import Dense # 뉴런의 입출력을 연결해주는 완전 연결 신경망 생성
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu')) # 순차적 망이기 때문에 'input)dim'은 다시 쓰지 않아도 된다.
model.add(Dense(1, activation='linear')) # output 값
model.summary()
```


![image](https://user-images.githubusercontent.com/39285147/180387209-a42f1385-dacc-45a3-a844-0ea2c3384262.png)


**입력값 개수**: 5 (나이, etc.)

**뉴런 개수**: 25

> 뉴런 개수가 적을수록 손실이 크게 발생한다 (에포크 수를 늘림으로써 보완 가능하다).

**bias**: 은닉층 뉴런 개수에 맞게 할당된다 (i.e., 은닉층 뉴런 개수 25개 --> bias 역시 25개가 존재한다).

**훈련 가능한 파라미터**: 딥러닝 모델 학습의 역전파 과정에서 피라미터 업데이트의 대상이 되는 가중치와 bias를 말한다.

1. 초기 입력값에서 첫 번째 은닉층까지, **훈련 가능한 피라미터* 개수 = 5(입력값 개수) * 25(첫 은닉층 뉴런개수) + 25(bias) = 150

2. 초기 입력값에서 두 번째 은닉층까지, **훈련 가능한 피라미터* 개수 =  25(첫 은닉층 뉴런개수) * 25(두 번째 은닉층 뉴런개수) + 25(bias) = 650

3. 출력, **훈련 가능한 피라미터* 개수 = 25(두 번째 은닉층 뉴런개수) * 1(출력값은 하나) + 1(bias) = 150

> Toal params: 826
>> 입출력 값에 대한 최선의 상관관계 도출을 위해 훈련되거나 조정되는 피라미터 총 개수이다.


```python
model.compile(optimizer='adam', loss='mean_squared_error') # 모델 학습 방법 제시

```
![image](https://user-images.githubusercontent.com/39285147/180389686-0fd6c3e2-8ee8-4e0f-999c-7686f8f89d41.png)

> Optimizer
>> 모델이 학습과정에서 어떻게 가중치 최적화를 이뤄내는지에 대한 방법을 제시한다.
>>
>> [adam이란?](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/gradient-discent.md)

> loss (손실함수)
>> 모델의 정확도를 판단하는데 사용되는 방법론이다.
>>
>> mean_squared_error (평균제곱오차)
>>>>
>>>> 예측값과 실제값의 차이를 나타내는 정도로, 그 값이 작을수록 실제값과 유사하여 정확한 예측을 해냈다고 볼 수 있다.


```python
epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)
```

![image](https://user-images.githubusercontent.com/39285147/180390662-b659a0d9-6f49-46cd-bb01-acd9cf2bd4b5.png)

모델이 학습하면서 epoch를 거듭함에 따라 loss(여기서는 평균제곱오차 방법을 사용)의 값이 줄어드는 것을 볼 수 있다.


- epoch: 배치 사이즈만큼의 하나의 학습을 몇번 시행할지 결정한다. 그 크기가 모델 성능을 향상시키는 최대 임계치에 가까워 질수록 더 정확한 예측을 해낼 수 있다.
- batch_size: 한 번에 학습할 훈련 데이터 개수
- verbose: 디폴트 0. 1로 지정하면 Epoch의 상황과, loss의 값이 output에 보여준다.
- [validation_split](https://github.com/hchoi256/ai-terms/blob/main/README.md)


## Model Evaluation

```python
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

```

![image](https://user-images.githubusercontent.com/39285147/180432355-41591eba-81ac-4625-a7a9-52b8dcba0c7a.png)

위 그래프에서 손실함수의 분포가 에포크가 [0, 4] 사이의 어느 임계치에서부터 크게 줄어들지 않는 것을 볼 수 있다.

이를 통하여 우리는 적당한 에포크 개수를 도출할 수 있을 것이다.


## Model Prediction

```python
y_predict = model.predict(np.array([[1, 50, 50000, 10000, 600000]]))
print('Expected Purchase Amount=', y_predict)
```

    Expected Purchase Amount= [35656.47]