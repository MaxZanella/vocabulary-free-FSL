# Vocabulary-free few-shot learning for Vision-Language Models
  
The official implementation of [*Vocabulary-free few-shot learning for Vision-Language Models*]().

Authors:
[Maxime Zanella*](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Clément Fuchs*](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=fr&oi=ao),
[Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=fr&oi=ao),
[Christophe De Vleeschouwer](https://scholar.google.ca/citations?user=xb3Zc3cAAAAJ&hl=en).

*Denotes equal contribution

## Quick Overview

In many real-world applications, class names may be unavailable, ambiguous, or poorly defined — yet we still want to adapt Vision-Language Models (VLMs) using only a few visual examples. As shown in the figure below, traditional few-shot learning methods rely heavily on known class names and handcrafted prompts, whereas **vocabulary-free few-shot learning** eliminates this dependency by leveraging only generic prompts (e.g., from ImageNet classes):

<div align="center" style="margin-top:20px; margin-bottom:20px;">
  <img src="figures/intro.png" alt="vocab-free-fsl" width="500">
</div>

\
**We introduce _SiM_ (Similarity Mapping)** — a simple yet powerful baseline for vocabulary-free few-shot learning with VLMs. Unlike conventional approaches that rely on textual labels and manual prompt design, SiM bypasses vocabulary entirely: it learns a **linear mapping** between similarity scores (computed between input images and a fixed set of generic prompts) and target classes — enabling classification without access to any class names.

### In short:
- 🔤 **Vocabulary-free**: no class names or manually handcrafted prompts required.
- ⚡ **Lightweight**: the mapping is learned in under one second.
- 🔍 **Interpretable**: learned weights can reveal how target classes align with known concepts.

---



## Table of Contents

1. [Method](#method)
2. [Installation](#installation)
3. [Reproducing Results](#reproducing-results)  
4. [Citation](#citation)  
5. [Contact](#contact)


---

## Method

**Similarity Mapping (SiM)** classifies target images by learning a linear mapping between similarity scores and class labels, using a fixed set of generic prompts (textual or visual).

The method works in two steps:

1. **Training**:  
   A small set of labeled images (shots) is compared to a predefined set of `K` generic prompts.  
   The resulting similarity matrix `L ∈ ℝ^{N × K}` is linearly mapped to the one-hot labels `Y ∈ ℝ^{N × C}` using regularized least-squares:

   ```math
   W^* = \arg\min_W \| Y - L W \|_F^2 + \lambda \|W\|_F^2
   ```
   
2. **Inference**:  
For a test image, similarity scores `l ∈ ℝ^{1 × K}` are computed with the same prompts.  
The predicted class scores are given by `s = l W`, and classification is done by `argmax(s)`.

This procedure is summarized in the figure below:
<div align="center" style="margin-top:20px; margin-bottom:20px;">
<img src="figures/method.png" alt="method" width="1000">
</div>

\
No class names are required: all operations are based on similarity scores with generic prompts. The method assumes access only to a pretrained Vision-Language Model and does not modify its internal weights.

### Interpretation

The linear mapping `W` not only enables classification but can also be used to inspect which generic prompts are most influential for each class. Below are examples of such interpretations for different types of prompts used in the paper:

**With ImageNet textual prompts (e.g., “a photo of a {class}”)**:
<div align="center" style="margin-top:10px; margin-bottom:20px;">
<img src="figures/visu_text.png" alt="text-prompts" width="600">
</div>

**With WordNet-based textual prompts**:
<div align="center" style="margin-top:10px; margin-bottom:20px;">
<img src="figures/visu_wordnet.png" alt="wordnet-prompts" width="600">
</div>

**With images from ImageNet**:
<div align="center" style="margin-top:10px; margin-bottom:20px;">
<img src="figures/visu_images.png" alt="image-prompts" width="600">
</div>

\
These visualizations illustrate how the learned mapping can highlight semantically related concepts even in the absence of class names.

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

---

## Reproducing Results

The performance of **SiM** can be reproduced using the scripts. To run with prompts using the classes of ImageNet and the ViT-L/14 backbone in the 16-shot setting, run:
```bash 
bash SiM_imagenet_text.sh --root /path_to_datasets/ --backbone ViT-L/14 --n_shots 16
```


For instance, the table below reports **Top-1 accuracy** for the **CLIP ViT-L/14** backbone with **16-shot** training across 10 datasets. **best vocabulary-free results** appear in bold.


| Method           | Vocab-free | SUN  | Aircraft | EuroSAT | Cars | Food | Pets | Flowers | Caltech | DTD  | UCF  | Avg  |
|------------------|----------|------|----------|---------|------|------|------|---------|---------|------|------|------|
| CoOp             | ✗        | 77.9 | 53.0     | 86.7    | 87.4 | 90.2 | 94.5 | 98.6    | 97.5    | 73.7 | 86.7 | 84.6 |
| TIP-Adapter-F    | ✗        | 79.6 | 55.8     | 86.1    | 88.1 | 91.6 | 94.6 | 98.3    | 97.5    | 74.0 | 87.4 | 85.3 |
| TaskRes          | ✗        | 76.9 | 55.0     | 84.3    | 87.6 | 91.5 | 94.7 | 97.8    | 97.3    | 74.4 | 86.6 | 84.6 |
| MaPLe            | ✗        | 78.8 | 46.3     | 85.4    | 83.6 | 92.0 | 95.4 | 97.4    | 97.2    | 72.7 | 86.5 | 83.5 |
| CLIP-LoRA        | ✗        | 79.4 | 66.2     | 93.1    | 90.9 | 89.9 | 94.3 | 99.0    | 97.3    | 76.5 | 89.9 | 87.7 |
| One-to-One       | ✅       | 26.7 | 2.9      | 47.1    | 5.9  | 20.5 | 72.8 | 18.6    | 77.1    | 29.4 | 43.6 | 34.4 |
| BLM              | ✅       | 34.5 | 3.1      | 53.4    | 2.7  | 25.0 | 58.8 | 24.5    | 89.9    | 43.7 | 50.8 | 38.7 |
| centroids        | ✅       | **74.9** | 51.8     | 81.8    | 81.3 | 88.2 | 88.6 | **98.5** | 96.6    | 67.3 | 83.8 | 81.3 |
| **SiM (ours)**   | ✅       | 74.3 | **52.8** | **90.1**| **84.0** | **89.6** | **93.5** | **98.8** | **97.3** | **73.1** | **85.7** | **83.9** |

*Note: ✅ = vocabulary-free method.*



---

## Citation

If you find this repository useful, please consider citing our paper:
```
TODO
```


## Contact

For any inquiries, please contact us at [maxime.zanella@uclouvain.be](mailto:maxime.zanella@uclouvain.be) and [clement.fuchs@uclouvain.be](mailto:clement.fuchs@uclouvain.be) or feel free to [create an issue](https://github.com/MaxZanella/vocabulary-free-FSL/issues).


## License

[AGPL-3.0](https://github.com/MaxZanella/vocabulary-free-FSL/blob/main/LICENSE)

## Acknowledgment
This repository is mainly based on [CLIP](https://github.com/openai/CLIP) and [TransCLIP](https://github.com/MaxZanella/transduction-for-vlms). 
