<div align="center">
<h5>

<h2><a href="https://arxiv.org/" style="color:#9C276A">
ViSCoP: Visual Probing for Domain Adapatation of Vision Language Models</a></h2>

[![arXiv](https://img.shields.io/badge/arXiv-ViSCoP%20Paper-b31b1b?style=flat&logo=arxiv)](https://arxiv.org)
[![HuggingFace Ego-in-Exo-PerceptionMCQ](https://img.shields.io/badge/🤗%20HuggingFace-Training%20Data-FFD21F?style=flat)]([https://huggingface.co/datasets/dreilly/Ego-in-Exo-PerceptionMCQ](https://huggingface.co/datasets/dreilly/ViSCoP_data))

</h5>

</div>

## Installation
Create a conda environment and install the required Python packages
```shell
conda create --name=viscop python=3.10
conda activate viscop

git clone https://github.com/dominickrei/ViSCoP.git
cd ViSCoP
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
```

## Preparing data for training
### Egocentric Viewpoint and Depth Modality
We provide the instruction pairs as well as videos for training through [HuggingFace](https://huggingface.co/datasets/dreilly/ViSCoP_data).

### Robot Control
Please follow [LLaRA's instructions](https://github.com/LostXine/LLaRA/blob/main/datasets/README.md) to prepare the VIMA dataset for ViSCoP training.

