<div align="center">
<h5>

<h2><a href="https://arxiv.org/abs/2510.13808" style="color:#9C276A">
VisCoP: Visual Probing for Domain Adapatation of Vision Language Models</a></h2>

[![arXiv](https://img.shields.io/badge/arXiv-VisCoP%20Paper-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2510.13808)
[![HuggingFace ](https://img.shields.io/badge/🤗%20HuggingFace-Training%20Data-FFD21F?style=flat)](https://huggingface.co/datasets/dreilly/VisCoP_data)
[![HuggingFace ](https://img.shields.io/badge/🤗%20HuggingFace-VisCoP%20Models-FFD21F?style=flat)](https://huggingface.co/dreilly/viscop-models/tree/main)

</h5>

<p align="center">
<img src="teaser.jpg" width="1000px" >
</p>

</div>

<p align="center">
  <video src="https://github.com/user-attachments/assets/3478722b-e98b-40e8-a82a-b15a9770c542" autoplay loop muted></video>
</p>


## ⚙️ Installation
1. Create a conda environment
```shell
conda create --name=viscop python=3.10
conda activate viscop
```

2. Clone VisCoP and install the required Python packages (we use `torch 2.4.0 + cuda 12.4` in our experiments)
```
git clone https://github.com/dominickrei/VisCoP.git
cd VisCoP
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
```

## Inference
First, download pre-trained VisCoP models through [HuggingFace](https://huggingface.co/dreilly/viscop-models/tree/main): `hf download dreilly/viscop-models --local-dir ./viscop_trained_models`

```python
from viscop import model_init, mm_infer
from viscop.mm_utils import load_video

## Load model
model_path = './viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert'

model, processor = model_init(
    model_path=model_path,
    device_map={"": "cuda"}
)

## Load video
video_path = './assets/ego_cut_carrot.mp4'

frames, timestamps = load_video(video_path, fps=1, max_frames=180)

## Create conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
            {"type": "text", "text": "What vegetable is the person cutting in the video?"},
        ]
    }
]

## Perform inference
inputs = processor(
    images=[frames],
    text=conversation,
    merge_size=2,
    return_tensors="pt",
)

prediction = mm_infer(
    inputs,
    model=model,
    tokenizer=processor.tokenizer,
    do_sample=False,
    modal='video'
)

print(prediction)
```

### Gradio Demo
We provide a Gradio interface to interact with VisCoP models. After downloading the models following the instructions above, run the following command to launch the demo:

```python inference/launch_gradio_demo.py --model-path ./viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert```

Note: if you are running the demo on a remote machine, be sure to forward the client port (9999 by default) so you can access the web interface through your local network. For example, `ssh -L 9999:localhost:9999 user@host`

## 🏋️ Training VisCoP
### 🎥 Preparing Training Data for Egocentric Viewpoint and Depth Modality
We provide the instruction pairs as well as videos for training through [HuggingFace](https://huggingface.co/datasets/dreilly/VisCoP_data). After downloading the data, update the following variables in `scripts/train/ego_depth_video/train_viscop.sh`:
* `DATA_DIR`: Update with the path to either egocentric or depth videos
* `TRAINING_JSON`: Update with the path to either egocentric or depth instructions

### 🤖 Preparing Training Data for Simulated Robot Control
Firstly, download and extract the VIMA data through [HuggingFace](https://huggingface.co/datasets/VIMA/VIMA-Data). Next, generate the training instruction pairs using the [conversion script provided by LLaRA](https://github.com/LostXine/LLaRA/blob/main/datasets/convert_vima.ipynb). We use the `D-inBC-text-multi-train-8k-front` instructions for all of our simulated robot control experiments.
* After extracting the VIMA data and generating the instructions, update `DATA_DIR` and `TRAINING_JSON` in `scripts/train/robotic_control/train_viscop.sh`

### 🤖 Preparing Training Data for Real-world Robot Control

**Coming soon!**

### 🔥 Update Training Script and Launch Training
In `scripts/train/ego_depth_video/train_viscop.sh` and `scripts/train/robotic_control/train_viscop.sh`, update the following arguments to match your system settings and paths:
* `INIT_MODEL`: This is the path to weights of the base VLM (VideoLLaMA3). Please use the following command to download and save the weights `python scripts/save_basevlm_for_finetuning.py --save-path-for-local-basevlm /path/to/save/base_vlm`
* `DATA_DIR`: The path to your data directory containing the egocentric, depth, or robot control data
* `TRAINING_JSON`: The path to a json file containing the egocentric, depth, or robot control instructions
* (Optional) `NUM_VISUAL_PROBES`: The number of Visual Probes to use in VisCoP
* (Optional) `INTERACTION_MODULE_POS`: The positions of the interaction modules. Acceptable values are `all` or a comma-separated list of integers (denoting zero-indexed layer indices of the vision encoder)

(**Single node training**) After updating the training scripts, initiate the training with the following command:
```shell
bash scripts/train/ego_depth_video/train_viscop.sh 1 <NUM_GPUS>
```

(**Multi-node training with SLURM**) After updating the training scripts, update the arguments in `ex_multi_node_slurm_job.sh` and submit the job:
```shell
sbatch ex_multi_node_slurm_job.sh
```

## ❄️ Evaluating VisCoP
### 💾 Preparing Source and Target Domain Data
| Target domain | Target Domain Data | Source Domain Data |
|-----------|-----------|-----------|
| Egocentric Viewpoint | [Ego-in-Exo PerceptionMCQ](https://huggingface.co/datasets/dreilly/Ego-in-Exo-PerceptionMCQ), [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema) | [NeXTQA](https://huggingface.co/datasets/lmms-lab/NExTQA), [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME), [ADL-X](https://github.com/ADL-X/LLAVIDAL/tree/main?tab=readme-ov-file#quantitative-evaluation-) |
| Depth Modality | [Ego-in-Exo PerceptionMCQ (Exo Depth)](https://huggingface.co/datasets/dreilly/VisCoP_data) (contained in `depth_videos.zip`) | [Ego-in-Exo PerceptionMCQ (Exo RGB)](https://huggingface.co/datasets/dreilly/Ego-in-Exo-PerceptionMCQ), [NeXTQA](https://huggingface.co/datasets/lmms-lab/NExTQA), [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME), [ADL-X](https://github.com/ADL-X/LLAVIDAL/tree/main?tab=readme-ov-file#quantitative-evaluation-) |
| Robot Control | [VIMA-Bench](https://github.com/vimalabs/VIMABench?tab=readme-ov-file#installation) | [Ego-in-Exo PerceptionMCQ (Exo RGB)](https://huggingface.co/datasets/dreilly/Ego-in-Exo-PerceptionMCQ), [NeXTQA](https://huggingface.co/datasets/lmms-lab/NExTQA), [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME), [ADL-X](https://github.com/ADL-X/LLAVIDAL/tree/main?tab=readme-ov-file#quantitative-evaluation-) |

<details>
<summary>Click to view our evaluation directory structure</summary>
  
    /path/to/vlm_eval_bench/
    ├── adlx
    │   ├── Charades-AR.json
    │   ├── Charades-Description.json
    │   ├── LEMMA-TC.json
    │   ├── Smarthome-AR.json
    │   ├── TSU-Description.json
    │   ├── TSU-TC.json
    │   └── videos
    │       ├── ADLMCQ-TC-TSU
    │       ├── Charades_v1_480
    │       ├── lemma_cropped
    │       └── SH_cropped224x224_better
    ├── egoperceptionmcq
    │   ├── all_category_qas.json
    │   ├── keystep_segments
    │   └── depth_videos
    ├── egoschema
    │   ├── GENERATION
    │   ├── MC
    │   ├── MC_PPL
    │   ├── questions.json
    │   ├── Subset
    │   ├── subset_answers.json
    │   └── videos
    ├── nextqa
    │   ├── NExTVideo
    │   └── test.csv
    └── videomme
        ├── subtitle
        ├── test-00000-of-00001.parquet
        └── videos

</details>

### 🏃🎥 Run the video understanding evaluations
After downloading the data, update the following variables in `scripts/eval/eval_video.sh`:
* `DATA_ROOT`: Update with the path to your evaluation directory, ensure it follows the same structure as shown above

After updating the evaluation script, run the following command:
```shell
bash scripts/eval/eval_video.sh /path/to/trained/viscop/model <BENCHMARKS> 1 <NUM_GPUS>
e.g., bash scripts/eval/eval_video.sh /path/to/trained/viscop/model egoperceptionmcq,egoschema,nextqa,videomme,adlx_mcq 1 8
```

**NOTE:** Evaluations on Ego-in-Exo PerceptionMCQ and ADL-X require Llama 3.1. You will need to install [Ollama](https://ollama.com/download) and download the Llama 3.1 model by running the command `ollama run llama3.1` prior to running the evaluations.
* If you are using an HPC environment and can not install Ollama, you will need to run an Ollama server locally
  * To do this, download the Ollama server that matches your system architecture from their [releases page](https://github.com/ollama/ollama/releases). Then update and uncomment lines `59-62` in `scripts/eval/eval_video.sh`
 
### 🏃🤖 Run the simulated robotics evaluations
We provide code to evaluate VisCoP models on simulated robotics tasks (VIMA-Bench) in `robotics_evaluation`, adapted from [LLaRA](https://github.com/LostXine/LLaRA)
1. Install the VIMA-Bench dependencies from [their official reposiotory](https://github.com/vimalabs/VIMABench?tab=readme-ov-file#installation)
2. Run the following command
```shell
torchrun --nproc_per_node=<NUM_GPUS> ./robotics_evaluation/eval-viscop.py <OUTPUT NAME> --model-path /path/to/trained/viscop_model --output-path ./robotics_evaluation/results/ --prompt-mode hso
```
3. After the evaluation is complete, run the cells in `robotics_evaluation/viscop-results.ipynb` to see the final success rates across each VIMA-Bench subset

## 🛠️ Building on VisCoP
If you would like to build on VisCoP, we mention the specific codes that might be helpful:
* `viscop/model/viscop_vision_encoder/modeling_viscop_vision_encoder.py` contains the implementations of visual probes and interaction modules
* `viscop/model/viscop_arch.py` contains the vision-language connector for the visual features and visual probes
* `viscop/model/viscop_qwen2.py` contains the code where visual probes are passed to the LLM
* `viscop/model/processor.py` contains the code to add placeholders for visual features and visual probes to the language instruction
* `viscop/train_viscop.py` contains the training logic for VisCoP

Please consider citing VisCoP if it is helpful for your project!
```bibtex
@article{viscop2025,
  title={VisCoP: Visual Probing for Video Domain Adaptation of Vision Language Models}, 
  author={Dominick Reilly and Manish Kumar Govind and Le Xue and Srijan Das},
  journal={arXiv Preprint},
  year={2025}
}
```

## 🙏 Acknowledgements
We thank the researchers behind the following codebases and model releases for their great open source work which VisCoP builds upon! [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3), [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [SigLIP](https://arxiv.org/abs/2303.15343), and [Qwen2.5](https://arxiv.org/abs/2412.15115).







