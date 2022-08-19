---
layout: single
title: "[논문 분석] Learning loss for active learning (CVPR 2019)"
categories: AIPaperCV
tag: [CVPR, CV, Active Learning]
toc: true
toc_sticky: true
toc_label: "쭌스log"
#author_profile: false
header:
    teaser: /assets/images/posts/ai-thumbnail.jpg
sidebar:
    nav: "docs"
---

# Learning loss for active learning
![image](https://user-images.githubusercontent.com/39285147/178131203-306385b4-13e0-4f23-b109-d041767b2cb7.png)

[**논문**](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf)

# Background
We need <u>big dataset</u> to train a deep learning model

![image](https://user-images.githubusercontent.com/39285147/178131277-d19251a9-5054-4165-9dcc-af68d5358798.png)

- **Low cost**: image collection (*unlabeled dataset*)
- **High cost**: image annotation (*labeled dataset*)

## Semi-Supervised Learning vs. Active Learning
- Semi-Supervised Learning
  - How to train *unlabeled dataset* based on labeled data
- Active Learning
  - What data to be labeled for better performance

## Passive Learning
![image](https://user-images.githubusercontent.com/39285147/178131377-71c2414e-7582-4c7a-8b65-669ce7650224.png)

****
# Related Researches
## ① Least Confident Sampling
![image](https://user-images.githubusercontent.com/39285147/178131566-821935b5-d41b-47e3-a6aa-1642982be95e.png)

## ② Margin Sampling
![image](https://user-images.githubusercontent.com/39285147/178131577-52945ac3-de4d-4b6c-bbb1-858fa8dcf28d.png)

## ③ Entropy Sampling
![image](https://user-images.githubusercontent.com/39285147/178131582-4170e264-fad4-461b-9e1c-94e06e90cd2a.png)

****
# Active Learning
![image](https://user-images.githubusercontent.com/39285147/178131383-7055e839-530a-4fed-9f66-882a19fa2f56.png)
- Using the *Welsh corgi* image for better performance in training model

## Active Learning Process
![image](https://user-images.githubusercontent.com/39285147/178131445-137e6839-25eb-45c4-8f5b-9643aa331946.png)
![image](https://user-images.githubusercontent.com/39285147/178131462-893920d8-546e-4370-9231-96830b151e26.png)
- **Random Sampling**
  - No observable changes in decision boundary after adding labeled data
- **Active Learning**
  - Training model much faster by selecting unnecessary data near decision boundary

## Loss Prediction Module
![image](https://user-images.githubusercontent.com/39285147/178131741-1602b16d-7592-49fd-bf87-01eec060b739.png)

- Loss prediction for a single unlabeled data
- Smaller network than target model; can be trained under the target model
- No need to define uncertainty through computation
- i.e., Classification, Regression, Object Detection 

### Loss Prediction: How to Work?
![image](https://user-images.githubusercontent.com/39285147/178131754-1622dd12-5d8d-4924-8355-6bfcd4aa3485.png)

### Loss Prediction: Architecture
![image](https://user-images.githubusercontent.com/39285147/178131774-d34b90fc-fb80-4f8d-993a-58df059409a7.png)
-	**GAP**: Global Avg. Pooling
-	FC: Fully Connected Layers
-	ReLU

### Method to Learn the Loss
![image](https://user-images.githubusercontent.com/39285147/178131782-be74272b-2f5f-4776-9539-743d4d46d18d.png)
- **MSE Loss Function** Limitation
  - Since target loss decreases, predicted loss is just adapting the change in target loss's size

### Margin Loss
![image](https://user-images.githubusercontent.com/39285147/178131827-718cfb0c-ce0f-4232-88c2-625051c325ba.png)
![image](https://user-images.githubusercontent.com/39285147/178131831-05dd7565-e456-43d5-a802-b500ddd20a32.png)

### Loss Prediction: Evaluation
![image](https://user-images.githubusercontent.com/39285147/178131902-2a585e0e-160d-4fdc-91b6-bbce8edbaec7.png)

#### Image Classification
![image](https://user-images.githubusercontent.com/39285147/178131907-76742e5f-0340-498f-9078-e51b70776ee4.png)
![image](https://user-images.githubusercontent.com/39285147/178131910-b89aaabd-8ac3-40fd-ad9f-9bd1b9c9b41f.png)

#### Object Detection
![image](https://user-images.githubusercontent.com/39285147/178131912-a61636f4-8366-472f-bfa9-ae888b853085.png)

#### Human Pose Estimation
![image](https://user-images.githubusercontent.com/39285147/178131914-f08af03e-bc94-44fe-a068-5706927dc939.png)
![image](https://user-images.githubusercontent.com/39285147/178131917-4cc6d0bf-7a4e-463b-8b81-18eb2451aafb.png)

****
# Limitation
![image](https://user-images.githubusercontent.com/39285147/178131923-923b0b5c-fa4d-400e-ae86-5f5480fe2e29.png)

# Note
referenced by 나동빈



