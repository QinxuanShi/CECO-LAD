# Overview

In this repository, you will find a Python implementation of CECO-LAD: a Cloud-Edge Collaboration Framework for Unsupervised Log Anomaly Detection.

# Introduction to CECO-LAD

Artificial intelligence (AI)-driven Log Anomaly Detection (LAD) is a critical component for maintaining the security and reliability of cyber infrastructure. However, deploying an effective LAD system in real-world environments presents a significant challenge, the cloud-edge dilemma, where accurate deep learning models favor centralized cloud resources, but operational constraints (e.g., latency, bandwidth, privacy, and energy) favor edge-local analysis. To address these challenges, we propose CECO-LAD, a cloud-edge collaborative framework for unsupervised log anomaly detection that balances detection accuracy with resource efficiency. In CECO-LAD, we propose an enhanced version of Anomaly Transformer (AT) as a base learner. Building on enhanced AT, we further proposed a novel ensemble learning approach as the core of CECO-LAD: the BAT for cloud deployment and Q-BAT for resource-constrained edge environments. A Mahalanobis distance-based routing policy enables cloud-edge collaboration by selectively forwarding only uncertain samples to the cloud and retaining confident cases at the edge, thereby minimizing resource consumption while maximizing detection accuracy. Additionally, we propose Green-LADE, a Green AI-inspired method to enable holistic evaluation.

<p align="center">
  <img src="pictures/framework.png" width="700">
</p>

# Project Analysis

## Problem Being Solved

Modern systems generate large volumes of log data. Automatically detecting anomalies in these logs is essential for system reliability and security. Two deployment extremes exist:

| Deployment | Pros | Cons |
|---|---|---|
| **Cloud only** | High model accuracy, powerful hardware | High latency, bandwidth cost, privacy risk, energy usage |
| **Edge only** | Low latency, local privacy, low energy | Limited compute, reduced model accuracy |

CECO-LAD resolves this **cloud-edge dilemma** by combining both: a full-precision ensemble runs in the cloud for difficult cases, while a compact quantized model runs on the edge for straightforward ones, with an intelligent router deciding which path each sample takes.

## System Architecture

```
Raw Logs (HDFS / BGL / OpenStack)
          │
          ▼
  Log Preprocessor
  (context windows, event-to-ID mapping)
          │
          ▼
  ┌───────────────────────────────────────┐
  │           INFERENCE PATH              │
  │                                       │
  │  ┌─────────────────────────────────┐  │
  │  │        Edge Device              │  │
  │  │   Q-BAT (Quantized BAT)         │  │
  │  │   A8W4 quantization via TorchAO │  │
  │  │   ExecuTorch C++ runtime (.pte) │  │
  │  └──────────────┬──────────────────┘  │
  │                 │ uncertain samples    │
  │                 ▼                     │
  │  ┌─────────────────────────────────┐  │
  │  │   Mahalanobis Distance Router   │  │
  │  │   (confidence-based routing)    │  │
  │  └──────────────┬──────────────────┘  │
  │                 │                     │
  │                 ▼                     │
  │  ┌─────────────────────────────────┐  │
  │  │        Cloud Server             │  │
  │  │   BAT (Bagging Anomaly Transf.) │  │
  │  │   81 EMAT models, GPU-optimized │  │
  │  └─────────────────────────────────┘  │
  └───────────────────────────────────────┘
          │
          ▼
  Anomaly / Normal prediction
  + Green-LADE holistic evaluation
```

## Key Components

### 1. Enhanced Anomaly Transformer (EMAT)
The base learner for both cloud and edge deployments. EMAT is built on the [Anomaly Transformer](https://arxiv.org/abs/2110.02642) and extends it with:
- **Triangular causal masking** to model temporal order in log sequences.
- **Learned sigma parameter** that captures per-sequence uncertainty.
- **Gaussian prior associations** paired with series temporal associations to measure how far each time step deviates from expected patterns.
- **Mahalanobis distance scoring** as the anomaly score, making the model sensitive to covariance structure rather than only per-feature variance.

### 2. BAT — Bagging Anomaly Transformer (Cloud)
An ensemble of 81 independently trained EMAT models produced by a full grid search over 3 values for each of 4 hyperparameters (3⁴ = 81 combinations):

| Hyperparameter | Values |
|---|---|
| `num_epochs` | 3, 6, 10 |
| `k` (loss weight) | 3, 4, 5 |
| `e_layer_num` (encoder layers) | 3, 6, 8 |
| `batch_size` | 32, 64, 96 |

Each base model is trained on a bootstrap sample of the training data. At inference time the ensemble uses **majority voting** (≥51% of models vote anomaly) as the default decision rule, with "at-least-one" and "consensus" available as alternatives. Anomaly thresholds for each model are set automatically via a Gaussian Mixture Model (GMM) fitted on reconstruction errors.

### 3. Q-BAT — Quantized BAT (Edge)
The same 81-model ensemble, but with each EMAT model:
1. **Quantized** to A8W4 (int8 activations, int4 weights) using [TorchAO](https://github.com/pytorch/ao).
2. **Exported** to the ExecuTorch `.pte` binary format via `convert_torchao.py`.
3. **Executed** on the edge device by a customized C++ `executor_runner`, with no Python runtime required.

This reduces model size and inference latency dramatically while retaining most of the detection accuracy.

### 4. Mahalanobis Distance-Based Routing
After Q-BAT scores a sample on the edge, the router measures how "confident" that prediction is by computing a Mahalanobis distance against the distribution of training-set scores. If the distance exceeds a calibrated threshold the sample is considered uncertain and forwarded to the cloud BAT ensemble for a higher-quality decision. This keeps bandwidth consumption low: only the hardest cases travel to the cloud.

### 5. Green-LADE Evaluation
A holistic evaluation methodology inspired by Green AI that jointly measures:
- **Detection quality**: Precision, Recall, F1-score, Accuracy.
- **Resource efficiency**: Inference latency, energy consumption (measured via `psutil`).

This lets practitioners compare systems on both axes rather than optimizing accuracy at the expense of sustainability.

## Data Flow

```
1. Raw log files
        │  logPreprocess_helper.py
        ▼
2. Structured sequences  (context windows of preceding log events)
        │  data_loader.py  (StandardScaler, train/test split,
        │                   bootstrap resampling for ensemble)
        ▼
3. PyTorch DataLoader tensors
        │
        ├─► Cloud: solver_ensemble.py  →  EMAT forward pass
        │         reconstruction error  →  GMM threshold  →  label
        │
        └─► Edge:  executor_runner (C++)  →  EMAT forward pass (.pte)
                  reconstruction error  →  YAML threshold  →  label
```

## Design Trade-offs and Decisions

| Decision | Rationale |
|---|---|
| Unsupervised learning | Log anomaly labels are rarely available in practice; training on normal logs only removes the labelling bottleneck. |
| Bagging over boosting | Bootstrap sampling with diverse hyperparameters reduces variance without requiring sequential training, enabling full parallelism. |
| A8W4 quantization | int4 weights halve memory compared to int8 while int8 activations preserve numeric range, striking a practical accuracy-vs-size balance. |
| ExecuTorch over ONNX Runtime | ExecuTorch is PyTorch-native, supports the custom operator set used in EMAT, and integrates directly with TorchAO quantization. |
| Mahalanobis routing over raw score thresholding | Mahalanobis distance accounts for feature correlations, making the routing decision more robust to distribution shift across datasets. |

# Get Started

## Configuration

### Cloud

- Ubuntu 24.04
- NVIDIA driver 580.126.09
- CUDA 12.4
- Python version >= 3.8
- PyTorch 2.4.0+cu124

### Edge

- Ubuntu 20.04
- Python 3.10
- PyTorch 2.4.0

## Installation

This code requires the packages listed in environment.yml and requirements.txt. The conda environments are recommended to run this code:

```bash
conda env create -f ./environment/cloud/environment.yml
conda activate ceco-lad-cloud
pip install -r ./environment/cloud/requirements.txt
```

```bash
conda env create -f ./environment/edge/environment.yml
conda activate ceco-lad-edge
pip install -r ./environment/edge/requirements.txt
```

## Download Data

CECO-LAD and other baseline methods are implemented on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [OpenStack](https://github.com/logpai/loghub/tree/master/OpenStack), and [BGL](https://github.com/logpai/loghub/tree/master/BGL) datasets. These datasets are available on [LogHub](https://github.com/logpai/loghub). Here we directly put the well-processed datasets here in ./Cloud/dataset.

## Download Trained Model

The trained BAT models can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1dh_pSu5M7fZVIWpdwfyBa4OLC4MKO1N0?usp=drive_link). The trained Q-BAT models are included in the ./Edge/executorch/checkpoints.

# Experiment

## BAT Model

For BAT, it is bagging based ensemble, we use 81 base models for bagging in CECO-LAD. To ensure robustness, we use four parameters:num_epochs, k (loss weight), e_layer_num (number of encoder layer), and batch_size. The detailed configs for BAT are provided in ./model_config/bat_config.

```bash
cd Cloud

# To train BAT model
python train_ensemble.py

# To test BAT model
python test_ensemble.py --voting majority
```

## Edge-based Q-BAT

Here we use [ExecuTorch](https://docs.pytorch.org/executorch/0.3/) (version 0.3) for lowering the model for Q-BAT at the edge. We have already included the executorch locally in our project.

If you want to try by your self for the start, according to the guideline of ExecuTorch, you can clone and install ExecuTorch locally.

```bash
cd Edge

git clone --branch v0.3.0 https://github.com/pytorch/executorch.git
cd executorch

# Update and pull submodules
git submodule sync
git submodule update --init

# Install ExecuTorch pip package and its dependencies, as well as
# development tools like CMake.
# If developing on a Mac, make sure to install the Xcode Command Line Tools first.
./install_requirements.sh

# Clean and configure the CMake build system. Compiled programs will appear in the executorch/cmake-out directory we create here.
(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)

# Build the executor_runner target
cmake --build cmake-out --target executor_runner -j9
```

### Runner Customize

After downloading and setting up the executorch, we need to customize the executor_runner to enable the customized input data. By replacing the executor_runner.cpp in the folder ./executorch/examples/portable/executor_runner to enable user customized runner. (We have already updated the file in the current version)

After updating the executor_runner file, build the the executor_runner target again:

```bash
# Build the executor_runner target
cmake --build cmake-out --target executor_runner -j9
```

### Model conversion

For Q-BAT model, we utilize executorch and torchao for edge optimization and quantization. To quantize the model and convert from .pth to .pte file, please run the following:

```bash
python convert_torchao.py
```

### Model Inference

To save time, both the trained Q-BAT models and the preprocessed datasets for executing at the edge can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1pBNMsucvw1eypn5gC_QvOzeRnMgTzLq2?usp=drive_link).

To execute the Q-BAT model accross all the datasets, run the scripts:

```bash
./edge_scripts/edge_execute.sh
```

## Demo

We provide the experiment scripts of all benchmarks under the folder ./scripts. You can reproduce the experiment results as follows:

```bash
bash ./scripts/run.sh
```
