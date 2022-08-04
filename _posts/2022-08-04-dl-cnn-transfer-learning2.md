---
layout: single
title: "Transfer Learning #2 - Classification"
categories: DL
tag: [deep learning, cnn, transfer learning, classification]
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

# Learning Goals

전이학습을 어떻게 우리의 task에 맞게 이용할 수 있는지 알아본다. <span style="color: blue">Let's see how transfer learning can be used for our tasks. </span>

가령, 우리는 수많은 사람의 얼굴 데이터를 학습한 모델을 전이학습으로 불러왔다. <span style="color: blue"> For example, we loaded a model that learned face data from a large number of people (transfer learning). </span>

이후, 우리 회사 부서 사람 몇명을 그 모델의 output layer 클래스로 할당해서 적용하면, 예측 결과로 부서 사람들 얼굴 중 가장 비슷한 사람 얼굴을 결과로 배출할 것이다. <span style="color: blue"> After that, if we assign a few people from our company department as the output layer class of the model, the most similar human face among department people faces will be output as a result of the prediction. </span>

그렇다면, 내가 수행하려는 task에 맞는 사전학습 모델은 어떻게 찾아올까 (가령, 사람 얼굴 학습한 모델)? <span style="color: blue"> Then, how can we download the pre-trained model suitable for our task (i.e., a model that learned face data)?  </span>

'TensorFlow Hub'라는 사이트는 다양한 사전학습 모델을 제공한다. <span style="color: blue"> 'TensorFlow Hub' provides diverse pre-trained models. </span>

가령, 모바일 데이터를 학습하고자 이와 관련된 사전 모델인 MobileNet을 전이학습해보자. <span style="color: blue"> Let's load a pre-trained model 'MobileNet' that trained mobile data. </span>

# Example 1: 'Watch'

```python
# Loading the pre-trained model
!pip install tensorflow_hub
import tensorflow_hub as hub

Trained_MobileNet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
```

```python
# Building the model
Trained_MobileNet = tf.keras.Sequential([
    hub.KerasLayer(Trained_MobileNet_url, input_shape = (224, 224, 3))
])

Trained_MobileNet.summary()
```


        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        keras_layer (KerasLayer)    (None, 1001)              3540265   
                                                                        
        =================================================================
        Total params: 3,540,265
        Trainable params: 0
        Non-trainable params: 3,540,265
        _________________________________________________________________


MobileNet 모델을 가져와 우리의 목적에 맞게 마지막 레이어만 하나 추가했다. <span style="color: blue"> We took the MobileNet model and added just one last layer for our purposes. </span>

이제 이 모델에 이미지를 넣어 예측을 수행해보자.  <span style="color: blue"> Now let's put an image into this model to make predictions. </span>

```python
# Predicting the image
Sample_Image = tf.keras.preprocessing.image.load_img("watch.jpg", target_size=(224, 224))
```

![image](https://user-images.githubusercontent.com/39285147/182831989-d3678292-8342-4652-adf4-251ec039a67a.png)


TensorFlow에서 제공하는 이미지 데이터 하나를 끌고왔다. <span style="color: blue"> We are using one image data provided by TensorFlow. </span>

신경망 학습을 위해 알맞은 형태로 이미지 데이터를 변환시켜주자. <span style="color: blue"> Let's transform the image data into an appropriate form for neural network training. </span>


```python
Sample_Image = np.array(Sample_Image) # to ndarray
Sample_Image = Sample_Image / 255.0 # normalization
Sample_Image = np.expand_dims(Sample_Image, axis = 0) # (224, 224, 3) --> (1, 224, 224, 3)
Sample_Image.shape
```

        (1, 224, 224, 3)


```python
predicted_class = Trained_MobileNet.predict(Sample_Image)
print(predicted_class)
predicted_class.shape
```

        array([[ 0.31899276,  0.69766045, -0.4810167 , ...,  0.22585514,
        -1.4412354 , -0.02521752]], dtype=float32)

        (1, 1001)


예측 결과로 높은 연관성을 가지는 클래스 1001개가 도출되었다. <span style="color: blue"> As a result of the prediction, 1001 classes with high correlation were derived. </span>

> ImageNet dataset has '1001' classes

이제, 가장 높은 확률로 연관성을 가지는 클래스 이름을 가져와보자. <span style="color: blue"> Let's get the class name with the highest probability. </span>


```python
predicted_class = np.argmax(predicted_class)
```

        827

우리 모델은 클래스 인덱스 827번을 차지하는 것이 인풋 사진의 클래스라고 예측했다. <span style="color: blue"> Our model predicted that the class occupying class index 827 is the class of the input picture.</span>

여기서 MobileNet 사이트에 들어가보면, 이 모델은 ImageNet 데이터셋에 기반해서 학습을 수행했다.  <span style="color: blue"> If you go to the MobileNet site, the model was trained based on the ImageNet dataset. </span>

따라서, 우리는 ImageNet 데이터셋의 클래스 인덱스 827을 확인해야 한다. <span style="color: blue"> Therefore, we need to check the class index 827 of the ImageNet dataset. </span>

```python
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt") # loading the dataset from the website

imagenet_labels = np.array( open(labels_path).read().splitlines() )
print(imagenet_labels)
imagenet_labels.shape
```


        array(['background', 'tench', 'goldfish', ..., 'bolete', 'ear',
       'toilet tissue'], dtype='<U30')

       (1001,)


```python
predicted_class_name = imagenet_labels[predicted_class]
predicted_class_name
```

        'watch'


모델이 맞게 예측한 모습이다. <span style="color: blue"> Our model got the correct answer. </span>

하지만, 시계는 사실 ImageNet dataset 안에 포함된 학습 클래스 중 하나로, 전이학습으로 불러온 모델은 이미 시계에 대한 분류를 잘한다.  <span style="color: blue"> However, 'clock' is actually one of the classes included in the ImageNet dataset, and the model imported by transfer learning is already good at classifying clocks. </span>

만약, 그 모델이 한 번도 분류해보지 못한 클래스로 분류해야 하는 새로운 이미지가 주어진다면 주어진 클래스 풀 안에서 그나마 비슷한 이상한 답을 도출할 것이다.  <span style="color: blue"> If given a new image that the model should classify into a class that it has never classified before, it will derive a similar but strange answer within the given class pool. </span>


# Example 2: 'Flower'

이번에는 이 모델에게 TensorFlow 라이브러리에서 완전히 새로운 꽃 사진 하나를 가져와서 기존 모델에 예측시켜보자. <span style="color: blue"> This time, let's get this model an entirely new picture of a flower from the TensorFlow library and make predictions on the old model. </span>

```python
# Loading the dataset of flowers
flowers_data_url = tf.keras.utils.get_file("flower_photos", "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", untar = True) # auto-untar when loading

# adjusting the image
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255) # scale to fit 255 pixels

flowers_data = image_generator.flow_from_directory(str(flowers_data_url), target_size=(224, 224), batch_size = 64, shuffle = True)
```


'image_generator.flow_from_directory'를 활용하여 한 batch(묶음)에 64개의 224x224 사이즈 이미지가 포함된 집합을 만들었다. <span style="color: blue"> Making a batch with 64 images of 224x224 using 'image_generator.flow_from_directory'. </span>

여기서 *shuffle*은 매번 이미지를 섞어서 모델이 같은 순서로 같은 이미지를 매번 학습하는 불상사를 피하기 위함이다. <span style="color: blue"> We set *shuffle* to True so that our model won't train the images in the same order. </span>

```python
for flowers_data_input_batch, flower_data_label_batch in flowers_data:
    print("Image batch shape", flowers_data_input_batch.shape)
    print("Label batch shape", flower_data_label_batch.shape)
    break
```


        Image batch shape (64, 224, 224, 3)
        Label batch shape (64, 5)

64개의 꽃 이미지 사진이 주어졌고, 그들은 5개의 클래스로 각각 분류되어야 한다. <span style="color: blue"> 64 flower image pictures were given, and they should each be classified into 5 classes. </span>

```python
# Predicing the image
predictions_batch = Trained_MobileNet.predict(flowers_data_input_batch)
predictions_batch.shape
```

        2/2 [==============================] - 2s 2s/step

        (64, 1001)


앞서 전이학습을 통해 불러온 MobileNet 모델을 활용해서 꽃 이미지를 분류해보자. <span style="color: blue"> Let's classify flower images using the MobileNet model loaded through transfer learning. </span>

```python
predicted_class_name = imagenet_labels[np.argmax(predictions_batch, axis = -1)]
```

여기서 'axis=-1' 은 64개의 행(=이미지 개수)에 속한 열(데이터)들 중에서 가장 큰 값의 인덱스를 가져오라는 말이다. <span style="color: blue"> Here, 'axis=-1 'means to get the index of the largest value among the columns (data) belonging to 64 rows (= number of images). </span>

무슨 말인지 헷갈린다면 아래 예시를 보자. <span style="color: blue"> If you are still confused, don't worry and look at the following example. </span>

```python
import numpy as np
a = np.arange(12).reshape(4,3) + 10
print(a.shape)

print("Max elements", np.argmax(a, axis=0))
print("Max elements", np.argmax(a, axis=-1))
```

        [[10 11 12]
        [13 14 15]
        [16 17 18]
        [19 20 21]]
        Max elements [3 3 3]
        Max elements [2 2 2 2]


이 예시는 4개의 행을 가지고 있고, 각 행에서 가장 높은 데이터의 인덱스를 가져온다. <span style="color: blue"> This example has 4 rows, and in each row we get the index of the highest data. </span>

'axis=0'이라면, **세로**를 기준으로 가령 '10 13 16 19' 중에서 가장 큰 값의 인덱스인 3을 가져온다. <span style="color: blue"> If 'axis=0', for example, 3, the index of the largest value among '10 13 16 19', is taken based on the vertical. </span>

반대로 'axis=-1'이라면, **가로**를 기준으로 가령 '10 11 12' 중에서 가장 큰 값의 인덱스인 2을 가져온다. <span style="color: blue"> If 'axis=-1', for example, 2, the index of the largest value among '10 11 12', is taken based on the horizontal. </span>

> 'axis=1' is same as 'axis=-1' in this example.

자, 이제 원래 문제로 돌아와서 예측 클래스 이름을 확인해보자. <span style="color: blue"> Now, back to the original problem, let's check the prediction class name. </span>

```python
predicted_class_name
```


        array(['daisy', 'rapeseed', 'feather boa', 'tray', 'daisy', 'bonnet',
            'daisy', 'daisy', 'barrow', 'picket fence', 'buckeye', 'orange',
            'daisy', 'jackfruit', 'bonnet', 'sea urchin', 'picket fence',
            'daisy', 'sea urchin', 'daisy', 'chainlink fence', 'daisy',
            'daisy', 'picket fence', 'picket fence', 'head cabbage', 'daisy',
            'daisy', 'bakery', 'red-backed sandpiper', 'vase', 'sea urchin',
            'daisy', 'teddy', 'cardoon', 'vase', 'daisy', 'daisy', 'daisy',
            'bee', 'daisy', 'strawberry', 'rapeseed', 'cauliflower', 'bakery',
            'quill', 'oxcart', 'vine snake', 'artichoke', 'pot', 'monarch',
            'daisy', 'spindle', 'chime', 'volcano', 'velvet', 'quill', 'daisy',
            'picket fence', 'cardoon', 'daisy', 'daisy', 'picket fence', 'hip'],
            dtype='<U30')
            

```python
plt.figure(figsize = (15, 15))

for n in range(64):
    plt.subplot(8, 8, n+1)
    plt.imshow(flowers_data_input_batch[n])
    plt.title(predicted_class_name[n])
    plt.axis("off")
```


![image](https://user-images.githubusercontent.com/39285147/182841799-08d3230c-52b2-40ef-a6c3-4732921e8642.png)


64개의 꽃 이미지들에 대한 모델의 예측 결과를 시각화했다. <span style="color: blue"> The prediction results of the model for 64 flower images were visualized. </span>

가장 첫 번째 이미지 분류만 봐도 결과가 이상하다. <span style="color: blue"> Just looking at the first image classification, the result is strange. </span>

'해바라기' 사진에 대하여 데이지라는 잘못된 분류 결과가 나타났다. <span style="color: blue"> An incorrect classification result of daisy was found for the sunflower picture. </span>

아마, ImageNet dataset에서 학습한 몇몇의 꽃 관련 사진들은 '해바라기' 클래스가 없어서 '데이지'라는 꽃으로 분류된 것 같다. <span style="color: blue"> Perhaps, some flower-related pictures learned from the ImageNet dataset are classified as a flower called 'daisy' because there was no class 'sunflower'. </span>

우리는 '데이지'가 아닌 '해바라기'로 해당 이미지의 예측 결과를 보고싶다. <span style="color: blue"> We would like to see the predictions for that image as sunflowers, not daisies. </span>


```python
MobileNet_feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" # base model

MobileNet_feature_extractor_layer = hub.KerasLayer(MobileNet_feature_extractor_url, input_shape=(224, 224, 3)) # convert images to a form of 224x224x3

feature_batch = MobileNet_feature_extractor_layer(flowers_data_input_batch) # apply the flower images we want to predict

feature_batch.shape,
```

        TensorShape([64, 1280])


아까와 같은 64개의 꽃 사진들을 1280개의 클래스로 구분하는 base model이다. <span style="color: blue"> This is a base model that identifies the previous 64 flower photos into 1280 classes. </span>

```python
MobileNet_feature_extractor_layer.trainable = False # not modifying base model's parameters

print(flowers_data.num_classes)

model = tf.keras.Sequential([
    MobileNet_feature_extractor_layer,
    tf.keras.layers.Dense(flowers_data.num_classes, activation = "softmax")
])
```

        5


        Model: "sequential_1"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        keras_layer_1 (KerasLayer)  (None, 1280)              2257984   
                                                                        
        dense (Dense)               (None, 5)                 6405      
                                                                        
        =================================================================
        Total params: 2,264,389
        Trainable params: 6,405
        Non-trainable params: 2,257,984
        _________________________________________________________________


base model가 학습하여 얻은 가중치와 편향은 그대로 가져가고, 마지막에 output layer만 수정하여 우리가 원하는 분류 결과를 보여주도록 한다. <span style="color: blue"> We don't want to change the weights and biases obtained from the base model and only the output layer to be modified. </span>


```python
model.compile(optimizer="Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
history = model.fit(flowers_data, epochs = 20)
print(flowers_data.class_indices.items())
```



        Epoch 1/20
        58/58 [==============================] - 183s 3s/step - loss: 0.8617 - accuracy: 0.6768
        Epoch 2/20
        58/58 [==============================] - 136s 2s/step - loss: 0.4060 - accuracy: 0.8608
        Epoch 3/20
        58/58 [==============================] - 132s 2s/step - loss: 0.3253 - accuracy: 0.8894
        ...
        Epoch 19/20
        58/58 [==============================] - 145s 2s/step - loss: 0.0857 - accuracy: 0.9826
        Epoch 20/20
        58/58 [==============================] - 137s 2s/step - loss: 0.0812 - accuracy: 0.9845


        dict_items([('daisy', 0), ('dandelion', 1), ('roses', 2), ('sunflowers', 3), ('tulips', 4)])


상기 결과처럼 모델은 총 다섯 개의 꽃 클래스로 인풋 이미지들을 분류하는 학습을 수행한다.  <span style="color: blue"> As shown above, the model classifies input images into a total of five flower classes. </span>

이제 아까 해바라기를 데이지라 잘못 분류했던 예측이 어떻게 바뀔지 확인해보자.  <span style="color: blue"> Now let's see how the prediction that our model had previously misclassified sunflowers as daisies would change. </span>



```python
predicted_batch = model.predict(flowers_data_input_batch) # predicing the images
predicted_id = np.argmax(predicted_batch, axis = -1) # find the index of the most proabable class

# get the class names (must be 5 in total)
class_names = sorted(flowers_data.class_indices.items(), key = lambda pair:pair[1])
class_names = np.array( [ key.title() for key, value in class_names ] )

predicted_label_batch = class_names[predicted_id] # get the name of the class
```


```python
class_names = sorted(flowers_data.class_indices.items(), key = lambda pair:pair[1])
class_names = np.array( [ key.title() for key, value in class_names ] )
class_names
```

        array(['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips'],
            dtype='<U10')


```python
plt.figure(figsize = (15, 15))
plt.subplots_adjust(hspace=0.5)

for n in range(64):
    plt.subplot(8, 8, n+1)
    plt.imshow(flowers_data_input_batch[n])
    plt.title(predicted_label_batch[n].title())
    plt.axis("off")
```


![image](https://user-images.githubusercontent.com/39285147/182860851-4d9bcd69-e7a3-44b2-950d-783d282b1069.png)



이제서야 '해바라기'를 맞게 예측하는 모습이다. <span style="color: blue"> Now, our model does the trick in prediction tasks! </span>


