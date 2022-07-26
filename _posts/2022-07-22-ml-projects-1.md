---
layout: single
title: "ML Project 1: ANN - Car Sales Prediction"
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

# 코드
**[Notice]** [download here](https://github.com/hchoi256/machine-learning-development)
{: .notice--danger}


# Learning Goals
Artificial Neural Network (ANN)을 이용한 회귀 작업 처리를 이해한다. Understanding of how ANN solves regression tasks.

순방향/역전파를 동반하는 가중치 학습의 과정에 대해 보다 나은 이해를 도모한다. Better understanding of deep learning through forward/backward propagation

# Description
여러분이 자동차 딜러 혹은 차량 판매원이라 가정해보자. Now, you are a car seller.

상기 고객들의 특정 데이터(나이, 연봉, etc.)를 참고하여 고객들이 차량 구매에 사용할 금액을 예측하여 특정 집단에 대한 타깃 마케팅을 이루고자 한다. Your job is to analyze the customer dataset, then predict how much money a new customer would like to spend.

## Dataset
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

# Import Dataset

```python
import pandas as pd # data frame
import numpy as np # numbers
import matplotlib.pyplot as plt
import seaborn as sns # data visualization

car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1') # 데이터셋이 '@'와 같은 특수문자를 포함하기 때문에 상기 인코딩 설정을 해줘야한다. Encoding process (i.e., remove '@')
```

# Data Visualization

## Seaborn

```python
sns.pairplot(car_df) # 씨본 덕분에 분석 작업을 여러 번 할 필요없이 여러 종류의 시각화를 보여준다 Easily produce data visualization using seaborn
```

![screensht](https://user-images.githubusercontent.com/39285147/180380848-d3772ba0-a21b-416b-8139-534c7a3aa721.JPG)

데이터 분포에서 맨 아래 위치한 행은 'Car Purchase Amount'이고, 각 열은 순서대로 Gender, Age, Annual Salary, Credit Car Debt, Net Worth, Car Purchase Amount이다.

따라서, 나이가 증가함에 따라 차량 구매 예상 금액이 증가하는 선형적 형태의 데이터 분포를 보여주고, 반대로 Credit Card Debt은 종속변수와 뚜렷한 상관관계를 나타내지 않는 것으로 관찰된다.

# Data Preprocessing
# Remove Unnecessary Variables
```python
X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1) # 종속변수에 영향을 끼치지 않는 불필요한 입력피처를 제거한다. remove unncessary columns not affecting 'y'
y = car_df['Car Purchase Amount'] # 종속변수

X
```

![image](https://user-images.githubusercontent.com/39285147/180381916-2051d577-51ec-4ff8-be80-0685754f456b.png)

## Data Scaling
나이와 연봉과 같은 입력피처의 수치가 차이가 커서, 특정 피처에 과중화된 결과가 나올 수 있으므로 [0, 1] 값으로 정규화하는 스케일링(Scailing)을 적용해야 한다. Since there is a gap between age and salary (maybe overfitting), we should apply scaling that normalizes data into the range [0, 1]

이번 프로젝트에서, 우리는 **MinMaxScaler**를 사용한다. We are using **MinMaxScaler**

기존 StandardScaler와 MinMaxScaler의 차이점은 데이터가 **정규분포를 따르는지 혹은 따라야 하는지**에 달려있다. The main difference between MinMaxScaler and StandardScaler relies on normal distribution.

[reference](https://velog.io/@ljs7463/%ED%94%BC%EC%B2%98-%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81StandardScalerMinMaxScaler)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

```

# Model Training

**Dense**
- *첫번째 인자* : 출력 뉴런의 수를 설정합니다. # output neurons
- *input_dim* : 입력 뉴런의 수를 설정합니다. # input neurons
- *init* : 가중치 초기화 방법 설정합니다. method to init weights
  - 'uniform' : 균일 분포
  - 'normal' : 가우시안 분포
- *activation* : 활성화 함수 설정합니다.
  - 'linear' : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다. default setting
  - 'relu' : rectifier 함수, 은닉층에 주로 쓰입니다. used in hidden layers
  - 'sigmoid' : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다. used to solve binary classification tasks
  - 'softmax' : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다. used in output layers to solve multinomial classification tasks


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25) # Create training and test set

import tensorflow.keras
from keras.models import Sequential # 신경망을 순차적 형태로 설계 sequential network
from keras.layers import Dense # 뉴런의 입출력을 연결해주는 완전 연결 신경망 생성 fully connected layers
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu')) # 순차적 망이기 때문에 'input_dim'은 다시 쓰지 않아도 된다. since it is a sequential network, we don't need to set 'input_dim' here again
model.add(Dense(1, activation='linear')) # output
model.summary()
```


![image](https://user-images.githubusercontent.com/39285147/180387209-a42f1385-dacc-45a3-a844-0ea2c3384262.png)


**입력값 개수(# input)**: 5 (age, etc.)

**뉴런 개수(# neurons**: 25

> 뉴런 개수가 적을수록 손실이 크게 발생한다 (에포크 수를 늘림으로써 보완 가능하다). As # neurons decreases, loss increases (can be alleviated by increasing # epoches)

**bias**: 은닉층 뉴런 개수에 맞게 할당된다 (i.e., 은닉층 뉴런 개수 25개 --> bias 역시 25개가 존재한다). has the same number of neurons of the previous hidden layer

**훈련 가능한 파라미터(learnable parameters)**: 딥러닝 모델 학습의 역전파 과정에서 피라미터 업데이트의 대상이 되는 가중치와 bias를 말한다. i.e., weights and bias

1. 초기 입력값에서 첫 번째 은닉층까지(from initial input to first hidden layer), **# learnable parameters* = 5(# input) * 25(# neurons of first hidden layer) + 25(bias) = 150

2. 초기 입력값에서 두 번째 은닉층까지(from initial input to second hidden layer), **# learnable parameters* =  25(# neurons of first hidden layer) * 25(# neurons of second hidden layer) + 25(bias) = 650

3. Ouput, **# learnable parameters* = 25( neurons of second hidden layer) * 1(output) + 1(bias) = 150

> Tot. params: 826
>> 입출력 값에 대한 최선의 상관관계 도출을 위해 훈련되거나 조정되는 피라미터 총 개수이다. # parameters that have been adjusted for drawing better correlation between input and output


```python
model.compile(optimizer='adam', loss='mean_squared_error') # present how to tarin model

```
![image](https://user-images.githubusercontent.com/39285147/180389686-0fd6c3e2-8ee8-4e0f-999c-7686f8f89d41.png)

> Optimizer
>> 모델이 학습과정에서 어떻게 가중치 최적화를 이뤄내는지에 대한 방법을 제시한다. present how to achieve weight optimization in the process of training model
>>
>> [adam](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/gradient-discent.md)

> loss (손실함수)
>> 모델의 정확도를 판단하는데 사용되는 방법론이다. method to determine model's accuracy
>>
>> mean_squared_error (평균제곱오차)
>>>>
>>>> 예측값과 실제값의 차이를 나타내는 정도로, 그 값이 작을수록 실제값과 유사하여 정확한 예측을 해냈다고 볼 수 있다. difference between estimate and actual value (if low value, then better precdiction)


```python
epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)
```

![image](https://user-images.githubusercontent.com/39285147/180390662-b659a0d9-6f49-46cd-bb01-acd9cf2bd4b5.png)

모델이 학습하면서 epoch를 거듭함에 따라 loss(여기서는 평균제곱오차 방법을 사용)의 값이 줄어드는 것을 볼 수 있다. Loss is getting smaller as epoches go by.


- epoch: 배치 사이즈만큼의 하나의 학습을 몇번 시행할지 결정한다. 그 크기가 모델 성능을 향상시키는 최대 임계치에 가까워 질수록 더 정확한 예측을 해낼 수 있다. Train the batch size of data in one epoch
- batch_size: 한 번에 학습할 훈련 데이터 개수 # training data at one epoch
- verbose: 디폴트 0. 1로 지정하면 Epoch의 상황과, loss의 값이 output에 보여준다. 1: show epoch and loss in the output
- [validation_split](https://github.com/hchoi256/ai-terms/blob/main/README.md)


# Model Evaluation

```python
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

```

![image](https://user-images.githubusercontent.com/39285147/180432355-41591eba-81ac-4625-a7a9-52b8dcba0c7a.png)

위 그래프에서 손실함수의 분포의 에포크가 [0, 4] 사이의 어느 임계치에서부터 크게 줄어들지 않는 것을 볼 수 있다. In the graph above, you can see epoch at around 3 has stopped plummeting.

이를 통하여 우리는 적당한 에포크 개수를 도출할 수 있을 것이다. This gives us information about the best number of epoch to train the model.


# Model Prediction

```python
y_predict = model.predict(np.array([[1, 50, 50000, 10000, 600000]]))
print('Expected Purchase Amount=', y_predict)
```

    Expected Purchase Amount= [35656.47]