# Vocabulary-free few-shot learning for Vision-Language Models [8<sup>th</sup>MULA@CVPR 2025]
  
The official implementation of [*Vocabulary-free few-shot learning for Vision-Language Models*](https://arxiv.org/abs/2501.03729).

Authors:
[Maxime Zanella*](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Clément Fuchs*](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=fr&oi=ao),
[Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=fr&oi=ao),
[Christophe De Vleeschouwer](https://scholar.google.ca/citations?user=xb3Zc3cAAAAJ&hl=en).

*Denotes equal contribution

## Quick Overview

In many real-world applications, class names may be unavailable, ambiguous, or poorly defined — yet we still want to adapt Vision-Language Models (VLMs) with just a few visual examples.

<div align="center" style="margin-top:20px; margin-bottom:20px;">
  <img src="intro.png" alt="vocab-free-fsl" width="500">
  <p style="font-size:75%;"><em>Current few-shot learning methods assume that target class names are known, often requiring handcrafted or fine-tuned prompts. In vocabulary-free few-shot learning, we remove this constraint and classify images using only generic prompts, such as ImageNet classes.</em></p>
</div>

We introduce **SiM** (Similarity Mapping), a novel and efficient baseline for **vocabulary-free few-shot learning** with Vision-Language Models (VLMs). Unlike conventional approaches that rely on known class names and handcrafted textual prompts, SiM operates **without any vocabulary**: it learns a linear mapping from similarity scores (between input images and a set of generic prompts) to class assignments.

Key features of SiM:
- 🔤 **Vocabulary-free**: no access to class names or manually designed prompts is required.
- ⚡ **Lightweight**: training the mapping typically takes less than one second.
- 🔍 **Interpretable**: learned weights offer semantic insights by linking target classes to generic concepts.

SiM provides a strong and interpretable baseline for scenarios where class names are unavailable, ambiguous, or evolving — laying the groundwork for a new class of few-shot learning methods that decouple adaptation from predefined vocabularies.


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
