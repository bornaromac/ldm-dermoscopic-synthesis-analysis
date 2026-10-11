# Latent Diffusion for Dermoscopic Image Synthesis

Research code for class-conditional synthesis of 256 × 256 dermoscopic images using latent diffusion models (LDMs) trained on the HAM10000 dataset.

## Overview

Medical image datasets often exhibit substantial class imbalance, particularly in categories with limited numbers of examples. Generative models offer a potential way to investigate synthetic data generation in this setting. However, generating visually plausible images does not necessarily mean that the generated samples are diverse, representative of the real data distribution, or useful for downstream tasks.

This project investigates latent diffusion models for dermoscopic image synthesis across seven diagnostic categories from HAM10000. It forms part of a broader study comparing diffusion-model configurations using complementary evaluation criteria.

The project focuses on the generation pipeline and the evaluation of synthetic images rather than on clinical diagnosis.

## Research objectives

The project supports the investigation of the following questions:

- **Image fidelity:** How closely do generated samples resemble the distribution of real dermoscopic images?
- **Sample diversity:** Does the model represent the variability within each diagnostic category, including underrepresented classes?
- **Potential memorization:** How similar are generated samples to images in the training data?
- **Human evaluation:** How do human judgments of image authenticity relate to computational evaluation?
- **Downstream utility:** Does synthetic data provide measurable benefits when used for a downstream classification task?

These criteria address different properties of a generative model and should not be interpreted as interchangeable measures of overall quality.

## Dataset

The experiments use **HAM10000**, a publicly available dataset of dermatoscopic images covering seven diagnostic categories:

| Abbreviation | Diagnostic category |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

The dataset is class-imbalanced, which makes class-conditional generation and per-class evaluation particularly relevant.

The dataset itself is not included in this repository. Users must obtain it from its authorized source and comply with its terms of use.

## Method

The model operates in the latent space of a pretrained variational autoencoder (VAE), rather than directly in pixel space.

The pipeline consists of:

1. Preprocessing and loading dermoscopic images.
2. Encoding images into a lower-dimensional latent representation using a pretrained VAE.
3. Training a class-conditional diffusion model in latent space.
4. Generating latent samples conditioned on diagnostic categories.
5. Decoding generated latents into image space.
6. Evaluating the generated samples using complementary quantitative and qualitative procedures.

The implementation includes training and checkpoint management, image sampling, and evaluation utilities. The experiment configuration is maintained in `configs/ldm_ham_256.yaml`.

**Important:** Exact training settings, sampling parameters, and hardware requirements should be taken from the configuration and scripts associated with the experiment being reproduced.

## Evaluation

The repository contains utilities for evaluating generated images. The broader research study considers several complementary dimensions:

- **Fréchet Inception Distance (FID):** distribution-level feature-space comparison between real and generated images.
- **Precision and recall for generative models:** complementary indicators of sample fidelity and coverage of the reference distribution.
- **Nearest-neighbour analysis:** inspection of similarity between generated samples and training images to investigate potential memorization.
- **Human evaluation:** assessment of whether participants can distinguish real from generated images.
- **Downstream classification:** investigation of whether synthetic samples provide practical value for a classification task.

These methods have different assumptions and limitations. In particular, FID depends on the feature extractor and image preprocessing protocol, and nearest-neighbour similarity alone cannot establish that a model does or does not memorize its training data.

Human-evaluation results and downstream experiments are part of the broader study and should be interpreted in the context of their experimental protocols.

## Repository structure

```text
.
├── configs/
│   └── ldm_ham_256.yaml
├── src/
│   ├── train.py
│   ├── model.py
│   ├── data.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── resize_images.py
│   ├── sample.py
│   ├── final_sample.py
│   ├── compute_fid.py
│   ├── compute_pr.py
│   └── fid_reference.py
├── requirements.txt
├── requirements_locked.txt
├── supek_ldm_256.pbs
├── .gitignore
└── README.md
```

The exact file list may differ between repository versions. Survey-specific scripts are excluded from this public version.

## Installation

Create and activate a Python virtual environment using a Python version supported by the project.

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and torchvision must also be installed using versions compatible with the available hardware and CUDA environment. Consult the official PyTorch installation instructions and verify compatibility with the project dependencies before running the experiments.

## Pretrained VAE

The model configuration references the pretrained VAE `stabilityai/sd-vae-ft-mse`.

The required model weights must be available in the local Hugging Face cache or downloaded before running in offline mode. Access to pretrained weights may be subject to the relevant model repository's terms.

## Training and generation

Training, sampling, and evaluation are implemented in separate scripts under `src/`.

Before executing a command, inspect the corresponding script's argument parser and the configuration file to confirm the supported arguments, data paths, checkpoint locations, and output directories. Paths in the HPC job script must be adapted to the target environment.

Training from scratch may require substantial GPU time and storage. Existing checkpoints and generated samples are not included in this repository.

## Reproducibility

Reproducing an experiment requires more than the source code alone. Record and preserve:

- The exact code revision and configuration used.
- The dataset version and data-split definitions.
- The pretrained VAE checkpoint and its version.
- The random seeds and sampling settings.
- The software environment, GPU type, and relevant CUDA versions.
- The evaluation implementation and preprocessing protocol.

The training and evaluation scripts are provided as research code. Exact numerical reproduction should be verified against the recorded experimental environment.

## Limitations

Synthetic dermoscopic images are not a substitute for clinical evidence or expert validation. Visual plausibility does not establish diagnostic validity, representativeness, privacy preservation, or downstream utility.

Results can vary by diagnostic category, particularly when training data are imbalanced. Quantitative metrics, human judgments, and downstream performance may therefore lead to different conclusions.

This repository is intended for research and experimentation, not for clinical decision-making.

## Acknowledgements

The experiments were conducted using high-performance computing resources provided through the University of Zagreb University Computing Centre (SRCE), subject to the project's applicable acknowledgements and citation requirements.
