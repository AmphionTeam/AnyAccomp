## AnyAccomp: Generalizable Accompaniment Generation via Quantized Melodic Bottleneck

<div class="flex justify-center items-center gap-2 mt-6 mb-8">
<a href="https://arxiv.org/abs/2509.14052" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper"></a>
<a href="https://github.com/AmphionTeam/AnyAccomp" target="_blank"><img src="https://img.shields.io/badge/GitHub-Code-blue" alt="Code"></a>
<a href="https://huggingface.co/amphion/anyaccomp" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Model"></a>
<a href="https://anyaccomp.github.io"><img alt="Online Demo" src="https://img.shields.io/badge/Online-Demo-brightgreen">
  </a>
</div>

## Overview

AnyAccomp is an accompaniment generation framework that addresses two critical challenges faced by existing accompaniment generation models:

1. **Generalization**: Traditional models heavily rely on vocal separation, leading to suboptimal performance in real-world applications. AnyAccomp resolves this through its innovative quantized melodic bottleneck design.

2. **Versatility**: The framework extends beyond vocal accompaniment to support solo instruments, significantly broadening its application scenarios.

The workflow of AnyAccomp involves first extracting core melodic features using chromagrams and VQ-VAE, followed by generating matching accompaniments through a flow matching model based on these features.

Experimental results demonstrate that AnyAccomp not only excels on traditional vocal separation datasets but also significantly outperforms existing solutions for clean vocals and solo instruments, providing a more flexible tool for music creation.

<img src="https://anyaccomp.github.io/data/framework.jpg" alt="framework" width="500">

## Issues

If you encounter any issue when using AnyAccomp, feel free to open an issue in this repository. But please use **English** to describe, this will make it easier for keyword searching and more people to participate in the discussion.

## Demo Audio

Before diving into the local setup, you can listen to our demo directly in your browser at **[anyaccomp.github.io](https://anyaccomp.github.io)**

## Quickstart

To run this model, you need to follow the steps below:

1. Clone the repository and install the environment.
2. Run the Gradio demo / Inference script.

### 1. Clone and Environment

In this section, follow the steps below to clone the repository and install the environment.

1. Clone the repository.
2. Install the environment following the guide below.

```bash
git clone https://github.com/AmphionTeam/AnyAccomp.git

# enter the repositry directory
cd AnyAccomp
```

#### 2. Download the Pretrained Models

We provide a simple Python script to download all the necessary pretrained models from Hugging Face into the correct directory.

Before running the script, make sure you are in the `AnyAccomp` root directory.

Run the following command:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='amphion/anyaccomp', local_dir='./pretrained', repo_type='model')"
```

If you have trouble connecting to Hugging Face, you can try switching to a mirror endpoint before running the command:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

#### 3. Install the Environment

Before start installing, make sure you are under the `AnyAccomp` directory. If not, use `cd` to enter.

```bash
conda create -n anyaccomp python=3.9
conda activate anyaccomp
conda install -c conda-forge ffmpeg=4.0
pip install -r requirements.txt 
```

### Run the Model

Once the setup is complete, you can run the model using either the Gradio demo or the inference script.

#### Run Gradio 🤗 Playground Locally

You can run the following command to interact with the playground:

```bash
python gradio_app.py
```

#### Inference Script

If you want to infer several audios, you can use the python inference script from folder.


```bash
python infer_from_folder.py
```

By default, the script loads input audio from `./example/input` and saves the results to `./example/output`. You can customize these paths in the [inference script](./anyaccomp/infer_from_folder.py).


## Model Introduction

We provide the following pretrained checkpoints:

| Model Name                                                   | Description                                |
| ------------------------------------------------------------ | ------------------------------------------ |
| [VQ](https://huggingface.co/amphion/anyaccomp/tree/main/pretrained/vq) | Extracting core melodic features           |
| [Flow Matching](https://huggingface.co/amphion/anyaccomp/tree/main/pretrained/flow_matching) | Generating matching accompaniments         |
| [Vocoder](https://huggingface.co/amphion/anyaccomp/tree/main/pretrained/vocoder) | Generating matching accompaniments' audios |

You can download all pretrained checkpoints from [HuggingFace](https://huggingface.co/amphion/anyaccomp/tree/main) or use huggingface API.


```python
from huggingface_hub import hf_hub_download

# download semantic codec ckpt
semantic_code_ckpt = hf_hub_download("amphion/anyaccomp", filename="")

# same for other models
```

If you having trouble connecting to HuggingFace, you try switch endpoint to mirror site:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Citations

If you use AnyAccomp in your research, please cite the following paper:

```bibtex
@article{zhang2025anyaccomp,
  title={AnyAccomp: Generalizable Accompaniment Generation via Quantized Melodic Bottleneck},
  author={Zhang, Junan and Zhang, Yunjia and Zhang, Xueyao and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2509.14052},
  year={2025}
}
```

