# Vocabulary-free few-shot learning for Vision-Language Models [8<sup>th</sup>MULA@CVPR 2025]
  
The official implementation of [*Vocabulary-free few-shot learning for Vision-Language Models*](https://arxiv.org/abs/2501.03729).

Authors:
[Maxime Zanella*](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Clément Fuchs*](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=fr&oi=ao),
[Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=fr&oi=ao),
[Christophe De Vleeschouwer](https://scholar.google.ca/citations?user=xb3Zc3cAAAAJ&hl=en).

*Denotes equal contribution

## Quick Overview

In many real-world applications, class names may be unavailable, ambiguous, or poorly defined — yet we still want to adapt Vision-Language Models (VLMs) using only a few visual examples. As shown in the figure below, traditional few-shot learning methods rely heavily on known class names and handcrafted prompts, whereas **vocabulary-free few-shot learning** eliminates this dependency by leveraging only generic prompts (e.g., from ImageNet classes).

<div align="center" style="margin-top:20px; margin-bottom:20px;">
  <img src="intro.png" alt="vocab-free-fsl" width="500">
</div>


**We introduce _SiM_ (Similarity Mapping)** — a simple yet powerful baseline for vocabulary-free few-shot learning with VLMs.

Unlike conventional approaches that rely on textual labels and manual prompt design, SiM bypasses vocabulary entirely: it learns a **linear mapping** between similarity scores (computed between input images and a fixed set of generic prompts) and target classes — enabling classification without access to any class names.

### In short:
- 🔤 **Vocabulary-free**: no class names or handcrafted prompts required.
- ⚡ **Lightweight**: the mapping is learned in under one second.
- 🔍 **Interpretable**: learned weights reveal how target classes align with known concepts.


---


## Table of Contents

1. [Installation](#installation)  
2. [Basic Usage](#basic-usage)  
3. [Reproducing Results](#reproducing-results)  
4. [Citation](#citation)  
5. [Contact](#contact)


---

## Installation
This repository requires to install an environment and datasets:
### Environment
Create a Python environment with your favorite environment manager. For example, with `conda`: 
```bash
conda create -y --name my_env python=3.10.0
conda activate my_env
pip3 install -r requirements.txt
```
And install Pytorch according to your configuration:
```bash
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```
### Datasets
Please follow [DATASETS.md](DATASETS.md) to install the datasets.
You will get a structure with the following dataset names:
```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
|–– oxford_flowers/
|–– food-101/
|–– fgvc_aircraft/
|–– sun397/
|–– dtd/
|–– eurosat/
|–– ucf101/
```

## Citation

If you find this repository useful, please consider citing our paper:
```
@article{zanella2025realistic,
title={Realistic Test-Time Adaptation of Vision-Language Models},
author={Zanella, Maxime and Fuchs, Cl{\'e}ment and De Vleeschouwer, Christophe and Ben Ayed, Ismail}
journal={arXiv preprint arXiv:2501.03729},
  year={2025}
}
```


## Contact

For any inquiries, please contact us at [maxime.zanella@uclouvain.be](mailto:maxime.zanella@uclouvain.be) and [clement.fuchs@uclouvain.be](mailto:clement.fuchs@uclouvain.be) or feel free to [create an issue](https://github.com/MaxZanella/vocabulary-free-FSL/issues).


## License

[AGPL-3.0](https://github.com/MaxZanella/vocabulary-free-FSL/blob/main/LICENSE)

## Acknowledgment
This repository is mainly based on [CLIP](https://github.com/openai/CLIP) and [TransCLIP](https://github.com/MaxZanella/transduction-for-vlms). 
