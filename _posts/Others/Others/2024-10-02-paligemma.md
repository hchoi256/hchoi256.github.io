---
layout: single
title: "Evaluating on Image Hallucination for TTI Generative Models in I-HallA via PaliGemma"
categories: Others
tag: [Generative AI, GemmaSprint]
toc: true
toc_sticky: true
toc_label: "쭌스log"
# author_profile: false
# header:
#     teaser: /assets/images/posts/data-visual.jpg
sidebar:
    nav: "docs"
---

[Paper](https://arxiv.org/abs/2409.12784)
[Github](https://github.com/hchoi256/I-HallA-PaliGemma)

****
# I-HallA via PaliGemma ✨

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aligning-bag-of-regions-for-open-vocabulary/open-vocabulary-object-detection-on-mscoco&#41;]&#40;https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=aligning-bag-of-regions-for-open-vocabulary&#41;)

This is an **unofficial** release of the paper **Evaluating Image Hallucination in Text-to-Image Generation with Question-Answering**.

> [**Evaluating Image Hallucination in Text-to-Image Generation with Question-Answering**](https://arxiv.org/abs/2409.12784),            
> Youngsun Lim, Hojun Choi, Pin-Yu Chen, Hyunjung Shim
> [[Paper](https://arxiv.org/abs/2409.12784)][[Supp](https://arxiv.org/abs/2409.12784)][[project page(TBD)](https://github.com/hchoi256/evaluate-hallucination-PaliGemma)][[Bibetex](https://github.com/hchoi256/evaluate-hallucination-PaliGemma#Citation)]

****
# Installation

This project is based on [PaliGemma](https://huggingface.co/docs/transformers/main/en/model_doc/paligemma)

It requires the following packages:

- python==3.10.0
- transformers==4.45.1
- numpy==1.26.3

****
# Testing
## PaliGemma
The implementation based on PaliGemma achieves comparable results compared to the GPT-4o results reported in the paper.

| Models   | I-HallA Score (Science)      | I-HallA Score (History)      | I-HallA Score† (Science)     | I-HallA Score† (History)    |
|----------|------------------------------|------------------------------|------------------------------|-----------------------------|
| SD v1.4  | 0.253                        | 0.435                        | -                            | -                           |
| SD v1.5  | 0.209                        | 0.433                        | -                            | -                           |
| SD v2.0  | 0.236                        | 0.440                        | -                            | -                           |
| SD XL    | 0.298                        | 0.479                        | -                            | -                           |
| DallE-3  | 0.561                        | 0.566                        | -                            | -                           |
| Factual  | **0.756**                    | **0.773**                    | -                            | -                           |


## GPT-4o
| Models   | I-HallA Score (Science)      | I-HallA Score (History)      | I-HallA Score† (Science)     | I-HallA Score† (History)    |
|----------|------------------------------|------------------------------|------------------------------|-----------------------------|
| SD v1.4  | 0.353 ± 0.002                | 0.535 ± 0.013                | 0.033 ± 0.012                | 0.110 ± 0.010               |
| SD v1.5  | 0.309 ± 0.011                | 0.533 ± 0.004                | 0.030 ± 0.017                | 0.117 ± 0.021               |
| SD v2.0  | 0.336 ± 0.006                | 0.540 ± 0.014                | 0.027 ± 0.021                | 0.120 ± 0.010               |
| SD XL    | 0.398 ± 0.015                | 0.579 ± 0.012                | 0.077 ± 0.050                | 0.110 ± 0.066               |
| DallE-3  | 0.661 ± 0.020                | 0.666 ± 0.003                | 0.227 ± 0.029                | 0.133 ± 0.031               |
| Factual  | **0.856 ± 0.002**            | **0.873 ± 0.006**            | **0.517 ± 0.038**            | **0.533 ± 0.015**           |

# Citation

```bibtex
@inproceedings{wu2023baron,
    title={Evaluating Image Hallucination in Text-to-Image Generation with Question-Answering},
    author={Youngsun Lim, Hojun Choi, Pin-Yu Chen, Hyunjung Shim},
    year={2024},
    booktitle={arXiv},
}
```