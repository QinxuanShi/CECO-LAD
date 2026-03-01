# Overview

In this repository, you will find a Python implementation of CECO-LAD: a Cloud-Edge Collaboration Framework for Unsupervised Log Anomaly Detection.

# Introduction to CECO-LAD

Artificial intelligence (AI)-driven Log Anomaly Detection (LAD) is a critical component for maintaining the security and reliability of cyber infrastructure. However, deploying an effective LAD system in real-world environments presents a significant challenge, the cloud-edge dilemma, where accurate deep learning models favor centralized cloud resources, but operational constraints (e.g., latency, bandwidth, privacy, and energy) favor edge-local analysis. To address these challenges, we propose CECO-LAD, a cloud-edge collaborative framework for unsupervised log anomaly detection that balances detection accuracy with resource efficiency. In CECO-LAD, we propose an enhanced version of Anomaly Transformer (AT) as a base learner. Building on enhanced AT, we further proposed a novel ensemble learning approach as the core of CECO-LAD: the BAT for cloud deployment and Q-BAT for resource-constrained edge environments. A Mahalanobis distance-based routing policy enables cloud-edge collaboration by selectively forwarding only uncertain samples to the cloud and retaining confident cases at the edge, thereby minimizing resource consumption while maximizing detection accuracy. Additionally, we propose Green-LADE, a Green AI-inspired method to enable holistic evaluation.

# Get Started

## Configuration

- Ubuntu 20.04
- NVIDIA driver 460.73.01
- CUDA 11.2
- Python 3.10
- PyTorch

## Installation

This code requires the packages listed in requirements.txt. A conda environment is recommended to run this code:

```bash
conda create -f ./environment/environment.yml
conda activate ceco-lad
pip install -r ./environment/requirements.txt
```

## Download Data

CECO-LAD and other baseline methods are implemented on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [OpenStack](https://github.com/logpai/loghub/tree/master/OpenStack), and [BGL](https://github.com/logpai/loghub/tree/master/BGL) dataset. These datasets are available on [LogHub](https://github.com/logpai/loghub). You can also obtain these datasets that are well pre-processed from [Google Drive](https://drive.google.com/drive/folders/1iBBTYIx1DaEV5lDh6dO7L6Klb15wUHcI?usp=drive_link).

## Download Trained Model

The trained BAT and Q-BAT models can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1dh_pSu5M7fZVIWpdwfyBa4OLC4MKO1N0?usp=drive_link).

# Experiment

## BAT Model Training

For BAT, it is bagging based ensemble, we use 81 base models for bagging in CECO-LAD. To ensure robustness, we use four parameters:num_epochs, k (loss weight), e_layer_num (number of encoder layer), and batch_size. The detailed configs for BAT are provided in ./bat_config.

```
# To train BAT model
python train_bat.py

```

## Edge-based Q-BAT

Here we use [ExecuTorch](https://docs.pytorch.org/executorch/0.3/) (version 0.3) for lowering the model for Q-BAT at the edge.

According to the guideline of ExecuTorch, clone and install ExecuTorch locally. (The executorch folder has already been downloaded here)

```bash
git clone --branch v0.3.0 https://github.com/pytorch/executorch.git
```

```
cd executorch

# Update and pull submodules
git submodule sync
git submodule update --init

# Install ExecuTorch pip package and its dependencies, as well as
# development tools like CMake.
# If developing on a Mac, make sure to install the Xcode Command Line Tools first.
./install_requirements.sh
```

```
# Clean and configure the CMake build system. Compiled programs will appear in the executorch/cmake-out directory we create here.
(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)

# Build the executor_runner target
cmake --build cmake-out --target executor_runner -j9
```

## Demo

We provide the experiment scripts of all benchmarks under the folder ./scripts. You can reproduce the experiment results as follows:

```bash
bash ./scripts/run.sh
```

or you can directly run the command in the python console:

```bash
# BAT training
python ensemble_train.py

python test.py
```
