---
layout: single
title: "[논문분석] Entropy-Driven Mixed-Precision Quantization for Deep Network Design"
categories: AIPaperLightWeight
tag: [Model Compression, Mixed Precision, Quantization]
toc: true
toc_sticky: true
toc_label: "쭌스log"
author_profile: false
header:
    teaser: /assets/images/posts/qntzn.png
sidebar:
    nav: "docs"
---

[논문링크](https://openreview.net/forum?id=E28hy5isRzC)

<span style="color:yellow"> QAT 경우 overhead 줄일 수만 있다면, PTQ를 대체해도 좋을까?</span>

****
# 한줄요약 ✔
A one-stage solution that optimizes both the architecture and the corresponding quantization jointly and automatically. The key idea of our approach is to cast the joint architecture design and quantization as an Entropy Maximization process.

1. Its representation capacity measured by entropy is maximized under the given computational budget.
    1. `Quantization Entropy Score (QE-Score`) with calibrated initialization to measure the expressiveness of the system.
    2. `Quantization Bits Refinement` within evolution algorithm to adjust mixed-precision quantization.
2. Each layer is assigned with a proper quantization precision.
    1. The `Entropy-based ranking strategy` of mixed-precision quantization networks.
3. The overall design loop can be made on the CPU; no GPU is required.

****
# Introduction 🙌
![image](https://user-images.githubusercontent.com/39285147/222670139-efbe18e9-bd7d-4401-bb2d-f1ca7de6f653.png)
## Why Need Quantization?
- Most IoT devices have very limited on-chip memory.
- Deploying deep CNN on Internet-of-Things (IoT) devices is challenging due to the limited computational resources, such as limited SRAM memory and Flash storage.

**The key is to control the peak memory during inference.**

## Trends in Traditional Lightweight CNN
(1)Re-design a small network for IoT devices, then (2)compress the network size by mixed-precision quantization.

## Limitations
The incoherence of such a two-stage design procedure leads to the inadequate utilization of resources, therefore producing sub-optimal models within tight resource requirements for IoT devices. 

****
# Related Work 😉
## Training-free NAS methods
- Accelerates the progress of the model design using a proxy mechanism instead of a training-based accuracy indicator.

**Limitation:**
- Still lacks key techniques for cooperating mixed-precision quantization.

****
# Challenges and Main Idea💣
**C1)** Designing models under limited resources remains a challenging issue.

**C2)** Low-precision has a short range of expressible values, producing chronic accuracy degradation.

**Idea)** Build a training-free NAS on mixed-precision quantization for selected IoT devices.

****
# Proposed Method 🧿
## Quantization Entropy
### Maximum Entropy for Full-precision Models
### Quantization Entropy for Mixed-Precision Models
### Gaussian Initialization Calibration
### Resource Maximization for IoT Devices


****
# Experiment 👀
## Mixed-Precision Comparison
### Random Correlation Study

### Comparison with SOTA Models

## Tiny Image Classification
### Large-scale Classification on ImageNet

### Low-energy Application on Visual Wake Words

### Resource Maximization

## Tiny Object Detection on WIDER FACE

****
# Open Reivew 💗


****
# Major Takeaways 😃

****
# Conclusion ✨
## Strength

## Weakness

****
# Reference
