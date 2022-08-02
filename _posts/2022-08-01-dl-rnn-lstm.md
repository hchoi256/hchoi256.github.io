---
layout: single
title: "RNN - Time Series: LSTM"
categories: DL
tag: [deep learning, rnn, lstm, python]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/dl-thumbnail.jpg
sidebar:
    nav: "docs"
---

**[Notice]** [Reference](https://github.com/Dataweekends/zero_to_deep_learning_udemy)
{: .notice--danger}


# Loading the dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../assets/data/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows=6, skipfooter=9,
                 engine='python')
df
```


![image](https://user-images.githubusercontent.com/39285147/182352758-1db7300d-249f-4ec2-b7f4-2b1f4e942a6f.png)



상기 디렉토리에 *참조* 사이트에서 파일을 다운받자. <span style="color: blue"> Download the file from the *reference* site, then save it into the above directory. </span>


```python
df.describe()
```


![image](https://user-images.githubusercontent.com/39285147/182352030-57851bf4-8910-4dcc-89f0-fd90475b217a.png)


```python
df.corr()
```

![image](https://user-images.githubusercontent.com/39285147/182361418-701dae63-639d-46dc-8532-0d1c8a6ca33b.png)


```python
sns.pairplot(df)
```

![image](https://user-images.githubusercontent.com/39285147/182352282-eaafad09-1ce0-4114-b6e7-fa80f923a533.png)


```python
f, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(df.corr(), annot=True)
```

![image](https://user-images.githubusercontent.com/39285147/182361578-93cd0623-48b6-4191-8ddf-cf72e3c11e16.png)


씨본 라이브러리를 활용하여 한 눈에 변수들 간 상관관계를 시각화했다. <span style="color: blue"> The correlation between variables was visualized at a glance using the Seabone library. </span>

'Unadjusted'와 'Seasonally adjusted'는 서로 **선형적인** 상관관계가 있음을 발견할 수 있다. <span style="color: blue"> It can be found that 'Unadjusted' and 'Seasonally adjusted' have a **linear** correlation with each other. </span>


그렇다면, 이제 시간의 흐름에 따른 두 변수의 변화를 확인해보자. <span style="color: blue"> Now, let's check the change of the two variables with the passage of time. </span>


# Preprocessing

시계열 데이터를 학습하기 위해서는 시간과 관련된 변수를 인덱스로 삼는 전처리 작업이 필수다. <span style="color: blue"> In order to address time series data, we should go through the process to have time-related variables as indexes. </span>

```python
from pandas.tseries.offsets import MonthEnd

df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1) # date formatting
df = df.set_index('Adjustments') # preprocessing for time series
print(df.head())
df.plot()
```


                    Unadjusted  Seasonally adjusted
        Adjustments                                 
        1991-01-31     12588862             15026890
        1991-02-28     12154321             15304585
        1991-03-31     14337072             15413591
        1991-04-30     15108570             15293409
        1991-05-31     17225734             15676083


![image](https://user-images.githubusercontent.com/39285147/182353479-97b1ba63-fa6f-4301-b5c7-722d18cd6c42.png)


```python
time_pivot = pd.Timestamp('01-01-2012')

train = df.loc[:time_pivot, ['Unadjusted']]
test = df.loc[time_pivot:, ['Unadjusted']]
```

```python
# visualizing the datasets
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
```

![image](https://user-images.githubusercontent.com/39285147/182354623-4dd81c45-f931-4d14-96b0-0b3d2a96d1b6.png)


**Training set**: 2012년 까지의 데이터 <span style="color: blue"> Data before 2012 </span>

**Test set**: 2012년 이후 데이터 <span style="color: blue"> Data after 2012 </span>


파란 선은 'train', 주황색은 'test'를 나타낸다. <span style="color: blue"> blue: train, orange: test</span>

# Feature Scaling

```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

train
```

        array([[0.01402033],
            [0.        ],
            [0.0704258 ],
            [0.09531795],
            [0.16362761],
            [0.13514108],
        ...
            [0.81439355],
            [0.79916654],
            [0.80210057],
            [0.81482896],
            [1.        ]])


> [MinMaxScaler](https://github.com/hchoi256/ai-terms)

신경망 학습에 이용하기 전 테이블 형태로써 데이터를 확인하기 위해 dataframe 형태로 변형하자. <span style="color: blue"> Before using it for neural network training, let's transform it into a dataframe format to check the data in a table format.</span>

직관적으로 확인해보기 위해 12달의 시계열 분석을 위한 테이블을 형성해보자. <span style="color: blue"> For intuition, let's form a table for time series analysis of 13 months.</span>

```python
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)

for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)

```


![image](https://user-images.githubusercontent.com/39285147/182356292-733ab690-76e4-47a3-94f2-e43a66de6855.png)


테이블에서 시간의 흐름에 따라 시계열 값들 또한 누적됨을 확인해볼 수 있다. <span style="color: blue"> In the table, it can be seen that the time series values ​​are accumulated over time.</span>

# Splitting the dataset into the Training set and Test set

```python
X_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]

X_test = test_sc_df.dropna().drop('Scaled', axis=1)
y_test = test_sc_df.dropna()[['Scaled']]
```

결측치가 존재하는 행을 제외한 후, 신경망 학습에 불필요한 'Scaled'를 제거한다. <span style="color: blue"> After excluding rows with missing values, 'Scaled' unnecessary for neural network training is removed. </span>


```python
# Converting to ndarray
X_train = X_train.values
X_test= X_test.values

y_train = y_train.values
y_test = y_test.values
```

시각화 목적을 달성한 dataframe 형태를 다시 ndarray로 바꿔준다. <span style="color: blue"> Changing the dataframe type that has achieved the purpose of visualization back to ndarray. </span>


```python
# reshape(size, timestep, feature)
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)
```


기존 신경망 학습에 사용되는 X는 2차원 배열 형태여야 하지만, RNN은 '시간'이라는   새로운 차원이 존재한다.<span style="color: blue"> Neural network training usually requires 2-dimensional input data, but RNNs have a new dimension called 'time'. </span>

따라서, 2차원이 아닌 3차원 배열 형태로 바꿔줘야 한다. <span style="color: blue"> Therefore, the input must be converted to a 3D array rather than a 2D. </span>

# Building the model

```python
# LSTM
import tensorflow as tf

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(12, 1))) # (timestep, feature)
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
```


        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        lstm (LSTM)                 (None, 20)                1760      
                                                                        
        dense (Dense)               (None, 1)                 21        
                                                                        
        =================================================================
        Total params: 1,781
        Trainable params: 1,781
        Non-trainable params: 0
        _________________________________________________________________


LSTM 모델의 입력층 차원으로 (12, 1)을 넣어줬다. <span style="color: blue"> We put (12, 1) as the dimension of the input layer of the LSTM model. </span>


여기서, 12는 시간 차원으로 12달을 의미한다. <span style="color: blue"> Here, 12 means 12 months in the time dimension.</span>


# Training the model

```python
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train_t, y_train, epochs=100,
          batch_size=30, verbose=1, callbacks=[early_stop])
```


        Epoch 1/100
        8/8 [==============================] - 2s 6ms/step - loss: 0.2733
        Epoch 2/100
        8/8 [==============================] - 0s 6ms/step - loss: 0.1831
        Epoch 3/100
        8/8 [==============================] - 0s 6ms/step - loss: 0.1105
        Epoch 4/100
        8/8 [==============================] - 0s 5ms/step - loss: 0.0547
        Epoch 5/100
        8/8 [==============================] - 0s 5ms/step - loss: 0.0227
        Epoch 6/100
        8/8 [==============================] - 0s 5ms/step - loss: 0.0172
        Epoch 7/100
        8/8 [==============================] - 0s 5ms/step - loss: 0.0176
        Epoch 7: early stopping


EarlyStopping을 활용해서 손실함수 변화가 1번 이상 없다면 학습을 중지하게 설계했다.<span style="color: blue"> By using EarlyStopping, it is designed to stop learning if there is no change in the loss function more than once.</span>

뉴런 개수와 같은 요소들은 **자율적으로** 설정한다.<span style="color: blue"> Factors such as the number of neurons are set **autonomously**.</span>

# Predicting the Test set

```python
print(X_test_t)
```

        [[[1.06265011]
        [0.87180554]
        [0.84048091]
        [0.86220767]
        [0.88363094]
        [0.89302107]
        [0.92552046]
        [0.89993326]
        [0.83505683]
        [0.77259579]
        [0.56926634]
        [0.61423187]]
        ...
        [1.05593537]
        [0.9437244 ]
        [0.75806325]
        [0.78276721]]]

        

```python
y_pred = model.predict(X_test_t)
print(y_pred)
```

        [[0.7675418 ]
        [0.7773328 ]
        [0.79216003]
        [0.79674876]
        [0.7971103 ]
        [0.7974134 ]
        [0.7950487 ]
        [0.7938808 ]
        [0.79453826]
        [0.7947075 ]
        [0.797763  ]
        [0.802348  ]
        [0.7896755 ]
        [0.7992326 ]
        [0.80933905]
        [0.8092701 ]
        [0.81345344]
        [0.81029963]
        [0.8067038 ]
        [0.80851555]
        [0.8075502 ]
        [0.81002915]
        [0.8135476 ]
        [0.81820446]
        ...
        [0.88950527]
        [0.8895281 ]
        [0.89572793]
        [0.8883731 ]]

