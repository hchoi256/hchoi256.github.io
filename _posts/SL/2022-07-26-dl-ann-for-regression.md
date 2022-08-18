---
layout: single
title: "ANN - Regression"
categories: SL
tag: [ANN, Regression]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/dl-thumbnail.jpg
sidebar:
    nav: "docs"
search: false
---

# Code
**[Notice]** [download here](https://github.com/hchoi256/ai-workspace/blob/main/codes/artificial_neural_network.ipynb)
{: .notice--danger}

# Observing the dataset
        AT	    V	    AP	    RH	    PE
        8.34	40.77	1010.84	90.01	480.48
        23.64	58.49	1011.4	74.2	445.75
        29.74	56.9	1007.15	41.91	438.76
        19.07	49.69	1007.22	76.79	453.09
        11.8	40.66	1017.13	97.2	464.43
        13.97	39.16	1016.05	84.6	470.96
        22.1	71.29	1008.2	75.38	442.35
        ...

*AT*: Average Temperature

*V*: Vacuum

*AP*: Average Pressure

*RH*: Relative Huminity

*PE*: Predicted Energy

상기 데이터셋을 활용하여 하기 인공 신경망을 구축하고 새로운 데이터에 대해 효과적으로 예측값을 도출하는 것이 목표이다. <span style="color: yellow"> Our goal is to build an ANN that refers to the dataset above and make predictions effectively.</span>

<img width="2560" alt="ANN_Architecture" src="https://user-images.githubusercontent.com/39285147/181073579-8e6497c7-1ab6-4e37-859c-e024655c5d44.png">

# Data Preprocessing
## Importing the dataset
```python
import numpy as np
import pandas as pd
import tensorflow as tf
```

```python
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

## Splitting the dataset into the Training set and Test set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

# Building the ANN
## Initializing the ANN

```python
ann = tf.keras.models.Sequential()
```

```python
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # the second hidden layer
ann.add(tf.keras.layers.Dense(units=1)) # the ouput layer
```


# Training the ANN

```python
ann.compile(optimizer = 'adam', loss = 'mean_squared_error') # Compiling the ANN
ann.fit(X_train, y_train, batch_size = 32, epochs = 100) # Training the ANN model on the Training set
```

        ...
        Epoch 12/100
        240/240 [==============================] - 0s 1ms/step - loss: 265.4893
        Epoch 13/100
        ...
        Epoch 99/100
        240/240 [==============================] - 0s 1ms/step - loss: 26.8439
        Epoch 100/100
        240/240 [==============================] - 0s 1ms/step - loss: 26.7392
        <tensorflow.python.keras.callbacks.History at 0x7f895a5850b8>


```python
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2) # precision: decimal places
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```


        [[430.79 431.23]
        [461.8  460.01]
        [465.29 461.14]
        ...
        [472.51 473.26]
        [439.39 438.  ]
        [458.55 463.28]]

        
예측값과 실제값이 수직으로 배열된 2차원 행렬을 통해 가독성 좋게 비교가 가능하다. <span style="color: yellow"> Through our 2-dimensional matrix, we can easily compare estimates with actual values. </span>