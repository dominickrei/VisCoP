<div align="center">
<h5>

<h2><a href="https://arxiv.org/" style="color:#9C276A">
ViSCoP: Visual Probing for Domain Adapatation of Vision Language Models</a></h2>

[![arXiv](https://img.shields.io/badge/arXiv-ViSCoP%20Paper-b31b1b?style=flat&logo=arxiv)](https://arxiv.org)
[![HuggingFace ](https://img.shields.io/badge/🤗%20HuggingFace-Training%20Data-FFD21F?style=flat)](https://huggingface.co/datasets/dreilly/ViSCoP_data)

</h5>

</div>

## Installation
1. Create a conda environment
```shell
conda create --name=viscop python=3.10
conda activate viscop
```

2. Clone ViSCoP and install the required Python packages (we use `torch 2.4.0 + cuda 12.4` in our experiments)
```
git clone https://github.com/dominickrei/ViSCoP.git
cd ViSCoP
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
```

## Training ViSCoP
### Preparing Training Data for Egocentric Viewpoint and Depth Modality
We provide the instruction pairs as well as videos for training through [HuggingFace](https://huggingface.co/datasets/dreilly/ViSCoP_data). After downloading the data, update the following varaibles in `scripts/train/ego_depth_video/train_viscop.sh`:
* `DATA_DIR`: Update with the path to either egocentric or depth videos
* `TRAINING_JSON`: Update with the path to either egocentric or depth instructions

### Preparing Training Data for Simulated Robot Control
Firstly, download and extract the VIMA data through [HuggingFace](https://huggingface.co/datasets/VIMA/VIMA-Data). Next, generate the training instruction pairs using the [conversion script provided by LLaRA](https://github.com/LostXine/LLaRA/blob/main/datasets/convert_vima.ipynb). We use the `D-inBC-text-multi-train-8k-front` instructions for all of our simulated robot control experiments.
* After extracting the VIMA data and generating the instructions, update `DATA_DIR` and `TRAINING_JSON` in `scripts/train/robotic_control/train_viscop.sh`

### Preparing Training Data for Real-world Robot Control

**Coming soon!**

### Launch the training


## Evaluating ViSCoP


## Building on ViSCoP


## Acknowledgements

